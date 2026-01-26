# scripts/preprocess_hands.py
"""
Extract hand crops from PHOENIX image sequences using MediaPipe Hands.

Aligned to RGB sampling by reading frame_indices from:
  data_cache/<dataset>/manifests/<split>_rgb_manifest.json

Outputs:
  data_cache/<dataset>/hands/<split>/<video_id>.pt  with shape [T, 2, 3, H, W]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm
import mediapipe as mp
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")


def phoenix_frame_sort_key(p: Path) -> int:
    # Phoenix format: images0001.png, images0002.png, etc.
    m = re.search(r'images(\d+)', p.stem)
    if m:
        return int(m.group(1))
    # Fallback for other formats
    m = re.search(r'(\d+)', p.stem)
    return int(m.group(1)) if m else 0


class TemporalSampler:
    @staticmethod
    def uniform_sample(num_frames: int, target_frames: int) -> List[int]:
        if num_frames <= 0:
            return [0] * target_frames
        if num_frames <= target_frames:
            idx = list(range(num_frames))
            idx += [num_frames - 1] * (target_frames - len(idx))
            return idx
        return np.linspace(0, num_frames - 1, target_frames).astype(int).tolist()


class HandTracker:
    def __init__(self, master_config):
        self.cfg = master_config.hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.cfg.min_hand_conf,
            min_tracking_confidence=self.cfg.min_hand_conf,
        )

    def detect_hands(self, frame_rgb: np.ndarray) -> Dict[str, Optional[dict]]:
        results = self.hands.process(frame_rgb)
        det = {"left": None, "right": None}

        if not results.multi_hand_landmarks or not results.multi_handedness:
            return det

        h, w = frame_rgb.shape[:2]

        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label.lower()  # 'left' or 'right'

            pts = []
            for lm in hand_landmarks.landmark:
                pts.append([lm.x * w, lm.y * h, lm.z])
            pts = np.array(pts)

            xs = pts[:, 0]
            ys = pts[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            pad = self.cfg.padding_factor
            bw = max(1.0, (x_max - x_min) * pad)
            bh = max(1.0, (y_max - y_min) * pad)

            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2

            x1 = int(max(0, cx - bw / 2))
            y1 = int(max(0, cy - bh / 2))
            x2 = int(min(w, cx + bw / 2))
            y2 = int(min(h, cy + bh / 2))

            det[label] = {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "center": (float(cx), float(cy)),
            }

        return det

    def smooth_bboxes(self, bbox_seq: List[Dict[str, Optional[dict]]]) -> List[Dict[str, Optional[dict]]]:
        if len(bbox_seq) < 2:
            return bbox_seq

        for hand in ["left", "right"]:
            valid_t = []
            valid = []
            for i, d in enumerate(bbox_seq):
                if d.get(hand) is not None:
                    valid_t.append(i)
                    b = d[hand]
                    valid.append([b["x1"], b["y1"], b["x2"], b["y2"]])

            if len(valid_t) < 2:
                continue

            valid_t = np.array(valid_t)
            valid = np.array(valid)

            funcs = [interp1d(valid_t, valid[:, j], kind="linear", fill_value="extrapolate") for j in range(4)]
            T = len(bbox_seq)

            for i in range(T):
                if bbox_seq[i].get(hand) is None and self.cfg.interpolate_missing:
                    dist = np.min(np.abs(valid_t - i))
                    if dist <= self.cfg.max_interpolation_frames:
                        bb = [int(f(i)) for f in funcs]
                        bbox_seq[i][hand] = {"x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3], "interpolated": True}

            # moving average smooth
            w = self.cfg.smoothing_window
            for i in range(T):
                if bbox_seq[i].get(hand) is None:
                    continue
                s = max(0, i - w // 2)
                e = min(T, i + w // 2 + 1)
                coords = []
                for j in range(s, e):
                    if bbox_seq[j].get(hand) is not None:
                        b = bbox_seq[j][hand]
                        coords.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                if coords:
                    sm = np.mean(coords, axis=0).astype(int)
                    bbox_seq[i][hand]["x1"] = int(sm[0])
                    bbox_seq[i][hand]["y1"] = int(sm[1])
                    bbox_seq[i][hand]["x2"] = int(sm[2])
                    bbox_seq[i][hand]["y2"] = int(sm[3])

        return bbox_seq

    def close(self):
        self.hands.close()


class HandCropPreprocessor:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.dataset / "hands"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = HandTracker(config)
        self.crop_size = config.hands.crop_size  # (H,W)

    def crop_and_resize(self, frame_rgb: np.ndarray, bbox: Optional[dict]) -> np.ndarray:
        H, W = self.crop_size
        if bbox is None:
            return np.zeros((H, W, 3), dtype=np.uint8)

        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
        crop = frame_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)
        return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)

    def process_sequence(
        self,
        sequence_dir: Path,
        video_id: str,
        split: str,
        frame_indices: Optional[List[int]],
        target_frames: int,
    ) -> Dict[str, Any]:
        try:
            image_files = sorted(sequence_dir.glob("*.png"), key=phoenix_frame_sort_key)
            if not image_files:
                raise ValueError(f"No PNG images in {sequence_dir}")

            if frame_indices is None:
                frame_indices = TemporalSampler.uniform_sample(len(image_files), target_frames)

            selected = []
            for idx in frame_indices:
                if 0 <= idx < len(image_files):
                    selected.append(image_files[idx])
                else:
                    selected.append(None)

            frames = []
            det_seq = []

            for img_path in selected:
                if img_path is None:
                    # black frame + no det
                    frame_rgb = np.zeros((260, 210, 3), dtype=np.uint8)  # fallback size
                    frames.append(frame_rgb)
                    det_seq.append({"left": None, "right": None})
                    continue

                frame_bgr = cv2.imread(str(img_path))
                if frame_bgr is None:
                    frame_rgb = np.zeros((260, 210, 3), dtype=np.uint8)
                    frames.append(frame_rgb)
                    det_seq.append({"left": None, "right": None})
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                det_seq.append(self.tracker.detect_hands(frame_rgb))

            det_seq = self.tracker.smooth_bboxes(det_seq)

            left_crops, right_crops = [], []
            for frame_rgb, det in zip(frames, det_seq):
                left_crops.append(self.crop_and_resize(frame_rgb, det.get("left")))
                right_crops.append(self.crop_and_resize(frame_rgb, det.get("right")))

            left = torch.from_numpy(np.stack(left_crops)).permute(0, 3, 1, 2)  # [T,3,H,W]
            right = torch.from_numpy(np.stack(right_crops)).permute(0, 3, 1, 2)
            hands = torch.stack([left, right], dim=1).float() / 255.0  # [T,2,3,H,W]

            if self.config.hands.save_fp16:
                hands = hands.half()

            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / f"{video_id}.pt"
            torch.save(hands, out_path)

            left_valid = sum(1 for d in det_seq if d.get("left") is not None)
            right_valid = sum(1 for d in det_seq if d.get("right") is not None)
            T = len(det_seq)

            return {
                "video_id": video_id,
                "success": True,
                "split": split,
                "sequence_dir": str(sequence_dir),
                "num_frames": T,
                "frame_indices": [int(i) for i in frame_indices], 
                "left_hand_ratio": float(left_valid / T) if T else 0.0,
                "right_hand_ratio": float(right_valid / T) if T else 0.0,
                "output_path": str(out_path),
            }

        except Exception as e:
            return {"video_id": video_id, "success": False, "split": split, "error": str(e)}

    def close(self):
        self.tracker.close()


def load_phoenix_image_sequences(data_root: str, split: str) -> Dict[str, str]:
    sequences_dir = Path(data_root) / "features" / "fullFrame-210x260px" / split
    paths = {}
    for d in sequences_dir.iterdir():
        if d.is_dir():
            paths[d.name] = str(d)
    return paths


def load_rgb_manifest_indices(manifest_path: Path) -> Dict[str, List[int]]:
    if not manifest_path.exists():
        return {}
    data = json.load(open(manifest_path, "r", encoding="utf-8"))
    out = {}
    for r in data:
        if r.get("success") and "frame_indices" in r:
            out[r["video_id"]] = r["frame_indices"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="phoenix2014t")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data_cache")
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=64)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from preprocessing.config import phoenix2014t_config
    config = phoenix2014t_config(args.data_root)
    config.output_dir = args.output_dir
    config.rgb.num_frames = args.num_frames

    print(f"Loading {args.dataset} {args.split} split...")
    seq_paths = load_phoenix_image_sequences(args.data_root, args.split)
    print(f"Found {len(seq_paths)} sequences")

    rgb_manifest_path = Path(args.output_dir) / args.dataset / "manifests" / f"{args.split}_rgb_manifest.json"
    rgb_indices = load_rgb_manifest_indices(rgb_manifest_path)
    if not rgb_indices:
        print(f"⚠️  RGB manifest indices not found at {rgb_manifest_path}. Falling back to uniform sampling.")

    pre = HandCropPreprocessor(config)

    try:
        results = []
        for vid, seq in tqdm(seq_paths.items(), desc=f"Extracting HANDS {args.split}"):
            idx = rgb_indices.get(vid, None)
            results.append(pre.process_sequence(Path(seq), vid, args.split, idx, args.num_frames))

        ok = sum(r["success"] for r in results)
        print(f"\n{args.split} summary: {ok}/{len(results)} ok")

        manifest_dir = Path(args.output_dir) / args.dataset / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = manifest_dir / f"{args.split}_hands_manifest.json"
        json.dump(results, open(out_manifest, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"✅ Saved manifest: {out_manifest}")

    finally:
        pre.close()


if __name__ == "__main__":
    main()
