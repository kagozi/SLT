# scripts/preprocess_rgb.py
"""
Extract and preprocess RGB frames from PHOENIX-2014-T image sequences.

PHOENIX-2014-T provides pre-extracted frames in folders like:
features/fullFrame-210x260px/train/<video_id>/images0001.png ...

This script:
- Loads PNG sequences
- Samples fixed num_frames (uniform/adaptive)
- Resizes + normalizes (ImageNet)
- Saves tensors to: data_cache/phoenix2014t/rgb/<split>/<video_id>.pt
- Writes manifest with frame_indices for alignment with kpts/hands
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms

warnings.filterwarnings("ignore")


def phoenix_frame_sort_key(p: Path) -> int:
    # Phoenix format: images0001.png, images0002.png, etc.
    m = re.search(r'images(\d+)', p.stem)
    if m:
        return int(m.group(1))
    # Fallback for other formats
    m = re.search(r'(\d+)', p.stem)
    return int(m.group(1)) if m else 0


class ImageSequenceReader:
    """Read image sequences from PHOENIX-2014-T format."""

    def __init__(self, sequence_dir: Path):
        self.sequence_dir = sequence_dir

        image_files = sorted(sequence_dir.glob("*.png"), key=phoenix_frame_sort_key)
        if not image_files:
            raise ValueError(f"No .png images found in {sequence_dir}")

        self.image_paths = image_files
        self.num_frames = len(image_files)

        first = cv2.imread(str(image_files[0]))
        if first is None:
            raise ValueError(f"Cannot read first frame: {image_files[0]}")
        self.height, self.width = first.shape[:2]

    def read_frames(self, frame_indices: Optional[List[int]] = None) -> List[np.ndarray]:
        if frame_indices is None:
            frame_indices = list(range(self.num_frames))

        frames = []
        for idx in frame_indices:
            if idx < 0 or idx >= self.num_frames:
                frames.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
                continue

            frame = cv2.imread(str(self.image_paths[idx]))
            if frame is None:
                frames.append(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames


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

    @staticmethod
    def adaptive_sample(frames: List[np.ndarray], target_frames: int) -> List[int]:
        # Deterministic: pick top-motion frames
        n = len(frames)
        if n <= 0:
            return [0] * target_frames
        if n <= target_frames:
            idx = list(range(n))
            idx += [n - 1] * (target_frames - len(idx))
            return idx

        motion_scores = []
        for i in range(n - 1):
            diff = np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))
            motion_scores.append(diff.mean())
        motion_scores = np.array(motion_scores)  # length n-1

        selected = {0, n - 1}
        remaining = target_frames - 2
        if remaining > 0:
            top = np.argsort(motion_scores)[-remaining:]  # indices into motion_scores
            selected.update((top + 1).tolist())  # +1 for frame index offset

        out = sorted(selected)
        if len(out) > target_frames:
            out = out[:target_frames]
        # If due to duplicates/edge cases it's short, pad with last
        while len(out) < target_frames:
            out.append(out[-1])
        return out


class RGBPreprocessor:
    def __init__(self, config):
        self.config = config
        self.rgb_config = config.rgb
        self.output_dir = Path(config.output_dir) / config.dataset / "rgb"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.rgb_config.target_size),
            transforms.ToTensor(),
        ])

        self.normalize = None
        if self.rgb_config.normalize:
            self.normalize = transforms.Normalize(mean=self.rgb_config.mean, std=self.rgb_config.std)

    def process_sequence(self, sequence_dir: Path, video_id: str, split: str) -> Dict[str, Any]:
        try:
            reader = ImageSequenceReader(sequence_dir)

            if self.rgb_config.sampling_strategy == "uniform":
                frame_indices = TemporalSampler.uniform_sample(reader.num_frames, self.rgb_config.num_frames)
                frames = reader.read_frames(frame_indices)
            else:
                all_frames = reader.read_frames()
                frame_indices = TemporalSampler.adaptive_sample(all_frames, self.rgb_config.num_frames)
                frames = [all_frames[i] for i in frame_indices]

            processed = []
            for frame in frames:
                x = self.transform(frame)
                if self.normalize:
                    x = self.normalize(x)
                processed.append(x)

            video_tensor = torch.stack(processed)  # [T,C,H,W]

            if self.rgb_config.save_fp16:
                video_tensor = video_tensor.half()

            split_dir = self.output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            output_path = split_dir / f"{video_id}.pt"
            torch.save(video_tensor, output_path)

            return {
                "video_id": video_id,
                "success": True,
                "split": split,
                "sequence_dir": str(sequence_dir),
                "original_frames": reader.num_frames,
                "sampled_frames": self.rgb_config.num_frames,
                "frame_indices": [int(i) for i in frame_indices], 
                "original_size": f"{reader.width}x{reader.height}",
                "output_size": f"{self.rgb_config.target_size[0]}x{self.rgb_config.target_size[1]}",
                "output_path": str(output_path),
            }

        except Exception as e:
            return {"video_id": video_id, "success": False, "split": split, "error": str(e)}

    def process_split(self, split_dir: Path, split: str) -> List[Dict[str, Any]]:
        sequence_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        print(f"Found {len(sequence_dirs)} sequences in {split}")

        results = []
        for seq_dir in tqdm(sequence_dirs, desc=f"Processing RGB {split}"):
            vid = seq_dir.name
            results.append(self.process_sequence(seq_dir, vid, split))

        ok = sum(r["success"] for r in results)
        print(f"\n{split} summary: {ok}/{len(results)} ok")
        return results


def load_phoenix_annotations(data_root: str, split: str) -> Dict[str, Dict[str, str]]:
    annotations_dir = Path(data_root) / "annotations" / "manual"
    # Phoenix v3 has:
    # dev/test: PHOENIX-2014-T.dev.corpus.csv, PHOENIX-2014-T.test.corpus.csv
    # train: PHOENIX-2014-T.train.corpus.csv (also train-complex-annotation exists)
    anno_file = annotations_dir / f"PHOENIX-2014-T.{split}.corpus.csv"

    annotations = {}
    if not anno_file.exists():
        print(f"⚠️  Annotation file not found: {anno_file}")
        return annotations

    with open(anno_file, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split("|")
            # expected: name|signer|gloss|translation
            if len(parts) >= 4:
                vid = parts[0]
                annotations[vid] = {"signer": parts[1], "gloss": parts[2], "translation": parts[3]}
    return annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="phoenix2014t")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data_cache")
    parser.add_argument("--split", type=str, choices=["train", "dev", "test"], required=True)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--sampling", type=str, choices=["uniform", "adaptive"], default="uniform")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    from preprocessing.config import PreprocessingConfig, RGBConfig

    config = PreprocessingConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        output_dir=args.output_dir,
        rgb=RGBConfig(
            target_size=(args.target_size, args.target_size),
            num_frames=args.num_frames,
            sampling_strategy=args.sampling,
            phoenix_centered=True,
            use_person_detection=False,
            save_fp16=args.fp16,
        ),
    )

    split_dir = Path(args.data_root) / "features" / "fullFrame-210x260px" / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    print(f"Loading {args.dataset} {args.split} split from {split_dir}")
    annotations = load_phoenix_annotations(args.data_root, args.split)
    print(f"Loaded {len(annotations)} annotations")

    pre = RGBPreprocessor(config)
    results = pre.process_split(split_dir, args.split)

    # merge annotations into results
    for r in results:
        vid = r["video_id"]
        if r.get("success") and vid in annotations:
            r.update(annotations[vid])

    manifest_dir = Path(args.output_dir) / args.dataset / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.split}_rgb_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ RGB preprocessing complete for {args.split}")
    print(f"📄 Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
