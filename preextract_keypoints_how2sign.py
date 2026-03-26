"""
Pre-extract and cache keypoints for How2Sign RGB clips.

Each CSV row defines a sentence segment within a full video clip via
START / END timestamps (seconds).  We seek to the window, run MediaPipe
Holistic on those frames, and save a (T, 225) .npy array keyed by
SENTENCE_NAME.

Directory layout expected:
    <root_dir>/
        annotations/
            how2sign_train.csv
            how2sign_val.csv
            how2sign_test.csv
        train/   <-- mp4 files named {VIDEO_NAME}.mp4
        val/
        test/

Output (written next to the video splits):
    <root_dir>/keypoints/{split}/{SENTENCE_NAME}.npy

Usage:
    python preextract_keypoints_how2sign.py \\
        --root_dir /data/how2sign_rgb \\
        --max_frames 300 \\
        --splits train val test
"""

import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

from preprocessing import PhoenixKeypointExtractor


# CSV column names as they appear in how2sign annotation files
COL_VIDEO_NAME    = 'VIDEO_NAME'
COL_SENTENCE_NAME = 'SENTENCE_NAME'
COL_START         = 'START'
COL_END           = 'END'

SPLIT_CSV = {
    'train': 'how2sign_train.csv',
    'val':   'how2sign_val.csv',
    'test':  'how2sign_test.csv',
}


def extract_segment(extractor: PhoenixKeypointExtractor,
                    video_path: Path,
                    start_sec: float,
                    end_sec: float,
                    max_frames: int) -> np.ndarray:
    """
    Open video_path, seek to start_sec, read frames up to end_sec,
    and return a (T, 225) keypoint array.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((max_frames, extractor.feature_dim), dtype=np.float32)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    # Seek to start frame
    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pose_data, left_hand_data, right_hand_data, face_data = [], [], [], []

    frame_idx = start_frame
    while cap.isOpened() and frame_idx < end_frame:
        success, image = cap.read()
        if not success:
            break

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = extractor.holistic.process(image)

        pose_data.append(extractor._extract_pose(results))
        left_hand_data.append(extractor._extract_left_hand(results))
        right_hand_data.append(extractor._extract_right_hand(results))
        face_data.append(extractor._extract_face(results))

        frame_idx += 1

    cap.release()

    if not pose_data:
        return np.zeros((max_frames, extractor.feature_dim), dtype=np.float32)

    return extractor._process_keypoints(
        pose_data, left_hand_data, right_hand_data, face_data, max_frames
    )


def preextract_split(root_dir: Path, split: str, max_frames: int) -> None:
    csv_name = SPLIT_CSV.get(split)
    if csv_name is None:
        print(f"Unknown split '{split}', skipping.")
        return

    csv_path = root_dir / 'annotations' / csv_name
    if not csv_path.exists():
        print(f"[{split}] CSV not found: {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path, sep='\t')

    video_dir = root_dir / split
    kps_dir   = root_dir / 'keypoints' / split
    kps_dir.mkdir(parents=True, exist_ok=True)

    already = sum(1 for _, row in df.iterrows()
                  if (kps_dir / f"{row[COL_SENTENCE_NAME]}.npy").exists())
    remaining = len(df) - already
    print(f"[{split}] {len(df)} sentences, {already} cached, {remaining} to extract")

    if remaining == 0:
        print(f"[{split}] Nothing to do.")
        return

    extractor = PhoenixKeypointExtractor()
    errors = 0

    try:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split}"):
            sentence_name = row[COL_SENTENCE_NAME]
            out_path = kps_dir / f"{sentence_name}.npy"

            if out_path.exists():
                continue

            video_name = row[COL_VIDEO_NAME]
            video_path = video_dir / f"{video_name}.mp4"

            if not video_path.exists():
                tqdm.write(f"  WARNING: video not found: {video_path}")
                np.save(out_path,
                        np.zeros((max_frames, extractor.feature_dim), dtype=np.float32))
                errors += 1
                continue

            start_sec = float(row[COL_START])
            end_sec   = float(row[COL_END])

            keypoints = extract_segment(
                extractor, video_path, start_sec, end_sec, max_frames
            )
            np.save(out_path, keypoints)

    finally:
        extractor.close()

    print(f"[{split}] Done. Feature dim=225, errors={errors}")
    print(f"[{split}] Keypoints saved to: {kps_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Pre-extract MediaPipe keypoints from How2Sign RGB clips.'
    )
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to how2sign_rgb root directory')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Max frames per sentence (default: 300)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    for split in args.splits:
        preextract_split(root_dir, split, args.max_frames)

    print("\n✅ All How2Sign keypoints cached.")
    print("   Update --root_dir in How2SignDataset to point at this directory.")


if __name__ == '__main__':
    main()
