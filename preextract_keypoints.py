"""
Pre-extract and cache all keypoints BEFORE training.

Run this once:
    python preextract_keypoints.py --root_dir /path/to/PHOENIX-2014-T --max_frames 250

This avoids segfaults from MediaPipe running inside DataLoader worker processes.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from preprocessing import PhoenixKeypointExtractor


def preextract_split(root_dir: Path, split: str, max_frames: int):
    csv_path = root_dir / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
    df = pd.read_csv(csv_path, sep='|')

    kps_dir = root_dir / 'features' / 'keypoints' / split
    kps_dir.mkdir(parents=True, exist_ok=True)

    # Count how many already cached
    already_cached = sum(1 for _, row in df.iterrows()
                         if (kps_dir / f"{row['name']}.npy").exists())
    remaining = len(df) - already_cached
    print(f"[{split}] {len(df)} sequences, {already_cached} cached, {remaining} to extract")

    if remaining == 0:
        return

    # Single MediaPipe instance — no forking
    extractor = PhoenixKeypointExtractor()

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split}"):
            name = row['name']
            out_path = kps_dir / f"{name}.npy"

            if out_path.exists():
                continue

            frame_dir = root_dir / 'features' / 'fullFrame-210x260px' / split / name
            if not frame_dir.exists():
                print(f"  Warning: missing {frame_dir}")
                np.save(out_path, np.zeros((max_frames, extractor.feature_dim)))
                continue

            keypoints = extractor.extract_from_frames(frame_dir, max_frames)
            np.save(out_path, keypoints)
    finally:
        extractor.close()

    print(f"[{split}] Done. Feature dim = {extractor.feature_dim}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to PHOENIX-2014-T root')
    parser.add_argument('--max_frames', type=int, default=250)
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    for split in args.splits:
        preextract_split(root_dir, split, args.max_frames)

    print("\n✅ All keypoints cached. You can now run train.py with num_workers > 0.")


if __name__ == '__main__':
    main()