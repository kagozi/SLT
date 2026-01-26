# scripts/preprocess_kpts.py
"""
Extract pose/hand/face keypoints using MediaPipe Holistic from PHOENIX image sequences.

IMPORTANT: This script aligns to RGB sampling by reading frame_indices from:
  data_cache/<dataset>/manifests/<split>_rgb_manifest.json

Outputs:
  data_cache/<dataset>/kpts/<split>/<video_id>.pt   with shape [num_frames, D]
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm
import mediapipe as mp

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


class MediaPipeExtractor:
    FACE_REGIONS = {
        "lips_outer": list(range(61, 68)) + [291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
        "lips_inner": list(range(78, 82)) + [13, 312, 311, 310, 415],
        "left_eye": list(range(133, 145)) + [33, 160, 158, 133, 153, 144],
        "right_eye": list(range(362, 374)) + [263, 387, 385, 362, 380, 373],
        "left_eyebrow": list(range(70, 80)) + [46, 52, 53, 63, 66, 65],
        "right_eyebrow": list(range(300, 310)) + [276, 282, 283, 293, 296, 295],
        "nose": [1, 2, 98, 327] + list(range(195, 199)),
    }

    def __init__(self, master_config):
        self.cfg = master_config.keypoints
        self.master = master_config

        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=self.cfg.static_image_mode,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        )

        face_indices = []
        for region in self.FACE_REGIONS.values():
            face_indices.extend(region)
        self.face_indices = sorted(set(face_indices))[:80]

    def extract_frame(self, frame_rgb: np.ndarray) -> Dict[str, np.ndarray]:
        results = self.holistic.process(frame_rgb)

        # Pose (33 * 4)
        if self.cfg.use_pose and results and results.pose_landmarks:
            pose = []
            for lm in results.pose_landmarks.landmark:
                pose.extend([lm.x, lm.y, lm.z, lm.visibility])
            pose = np.array(pose, dtype=np.float32)
        else:
            pose = np.zeros(33 * 4, dtype=np.float32)

        # Hands (2 * 21 * 3)
        if self.cfg.use_hands and results and results.left_hand_landmarks:
            lh = []
            for lm in results.left_hand_landmarks.landmark:
                lh.extend([lm.x, lm.y, lm.z])
            lh = np.array(lh, dtype=np.float32)
        else:
            lh = np.zeros(21 * 3, dtype=np.float32)

        if self.cfg.use_hands and results and results.right_hand_landmarks:
            rh = []
            for lm in results.right_hand_landmarks.landmark:
                rh.extend([lm.x, lm.y, lm.z])
            rh = np.array(rh, dtype=np.float32)
        else:
            rh = np.zeros(21 * 3, dtype=np.float32)

        # Face (80 * 3)
        if self.cfg.use_face and results and results.face_landmarks:
            face = []
            for idx in self.face_indices:
                lm = results.face_landmarks.landmark[idx]
                face.extend([lm.x, lm.y, lm.z])
            face = np.array(face, dtype=np.float32)
        else:
            face = np.zeros(80 * 3, dtype=np.float32)

        return {"pose": pose, "left_hand": lh, "right_hand": rh, "face": face}

    def smooth_keypoints(self, seq: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        if not self.cfg.temporal_smoothing:
            return seq

        w = self.cfg.smoothing_window
        out = []
        for i in range(len(seq)):
            s = max(0, i - w // 2)
            e = min(len(seq), i + w // 2 + 1)
            window = seq[s:e]

            sm = {}
            for k in seq[i].keys():
                sm[k] = np.stack([x[k] for x in window]).mean(axis=0)
            out.append(sm)
        return out

    def close(self):
        self.holistic.close()


class KeypointsPreprocessor:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.dataset / "kpts"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = MediaPipeExtractor(config)

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

            # select frames by indices (pad out-of-range with None)
            selected = []
            for idx in frame_indices:
                if 0 <= idx < len(image_files):
                    selected.append(image_files[idx])
                else:
                    selected.append(None)

            frames_kpts = []
            for img_path in selected:
                if img_path is None:
                    frames_kpts.append({
                        "pose": np.zeros(33 * 4, dtype=np.float32),
                        "left_hand": np.zeros(21 * 3, dtype=np.float32),
                        "right_hand": np.zeros(21 * 3, dtype=np.float32),
                        "face": np.zeros(80 * 3, dtype=np.float32),
                    })
                    continue

                frame = cv2.imread(str(img_path))
                if frame is None:
                    frames_kpts.append({
                        "pose": np.zeros(33 * 4, dtype=np.float32),
                        "left_hand": np.zeros(21 * 3, dtype=np.float32),
                        "right_hand": np.zeros(21 * 3, dtype=np.float32),
                        "face": np.zeros(80 * 3, dtype=np.float32),
                    })
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_kpts.append(self.extractor.extract_frame(frame_rgb))

            frames_kpts = self.extractor.smooth_keypoints(frames_kpts)

            stacked = []
            for k in frames_kpts:
                stacked.append(np.concatenate([k["pose"], k["left_hand"], k["right_hand"], k["face"]]))
            kpts_tensor = torch.from_numpy(np.stack(stacked))  # [T, D]

            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            out_path = split_dir / f"{video_id}.pt"
            torch.save(kpts_tensor, out_path)

            non_zero_ratio = (kpts_tensor != 0).float().mean().item()

            return {
                "video_id": video_id,
                "success": True,
                "split": split,
                "sequence_dir": str(sequence_dir),
                "num_frames": len(frame_indices),
                "keypoints_dim": int(kpts_tensor.shape[1]),
                "quality_score": float(non_zero_ratio),
                "frame_indices": [int(i) for i in frame_indices], 
                "output_path": str(out_path),
            }

        except Exception as e:
            return {"video_id": video_id, "success": False, "split": split, "error": str(e)}

    def close(self):
        self.extractor.close()


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
    config.rgb.num_frames = args.num_frames  # used for fallback sampling

    print(f"Loading {args.dataset} {args.split} split...")
    seq_paths = load_phoenix_image_sequences(args.data_root, args.split)
    print(f"Found {len(seq_paths)} sequences")

    rgb_manifest_path = Path(args.output_dir) / args.dataset / "manifests" / f"{args.split}_rgb_manifest.json"
    rgb_indices = load_rgb_manifest_indices(rgb_manifest_path)
    if not rgb_indices:
        print(f"⚠️  RGB manifest indices not found at {rgb_manifest_path}. Falling back to uniform sampling.")

    pre = KeypointsPreprocessor(config)

    try:
        results = []
        for vid, seq in tqdm(seq_paths.items(), desc=f"Extracting KPTS {args.split}"):
            idx = rgb_indices.get(vid, None)
            results.append(pre.process_sequence(Path(seq), vid, args.split, idx, args.num_frames))

        ok = sum(r["success"] for r in results)
        avgq = float(np.mean([r["quality_score"] for r in results if r.get("success")])) if ok else 0.0
        print(f"\n{args.split} summary: {ok}/{len(results)} ok | avg quality: {avgq:.3f}")

        manifest_dir = Path(args.output_dir) / args.dataset / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        out_manifest = manifest_dir / f"{args.split}_kpts_manifest.json"
        json.dump(results, open(out_manifest, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"✅ Saved manifest: {out_manifest}")

    finally:
        pre.close()


if __name__ == "__main__":
    main()
