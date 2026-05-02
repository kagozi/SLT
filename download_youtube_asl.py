"""
YouTube-ASL download + MediaPipe Holistic extraction pipeline.

Downloads YouTube-ASL videos in shards using yt-dlp, extracts raw
(T, 543, 3) MediaPipe Holistic keypoints per sentence segment, saves
as .npy, then deletes the video.  The output format is identical to
the How2Sign_Holistic raw arrays so dataset_youtube_asl.py can reuse
process_holistic() verbatim.

MediaPipe Holistic layout (543 landmarks, matching How2Sign convention):
    Pose      :   0 -  32  (33 landmarks)
    Face mesh :  33 - 500  (468 landmarks)
    Left hand : 501 - 521  (21 landmarks)
    Right hand: 522 - 542  (21 landmarks)

Input metadata CSV (tab-separated, downloaded from Google Research):
    video_id  start_time  end_time  caption
    (column names are normalised case-insensitively)

Output layout:
    <out_dir>/
        metadata/
            youtube_asl_train.csv    (tab-sep, with PSEUDOGLOSS column)
            youtube_asl_val.csv
            youtube_asl_test.csv
        keypoints/
            train/{video_id}_{seg_idx:05d}.npy   # (T, 543, 3) float32
            val/
            test/

Usage — single shard:
    python download_youtube_asl.py \\
        --metadata /data/youtube_asl/raw_metadata.csv \\
        --out_dir  /data/youtube_asl \\
        --shard_idx 0 --total_shards 20

Usage — prepare split CSVs only (no download):
    python download_youtube_asl.py \\
        --metadata /data/youtube_asl/raw_metadata.csv \\
        --out_dir  /data/youtube_asl \\
        --prepare_only
"""

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm


# ── Metadata helpers ─────────────────────────────────────────────────────────

_COL_ALIASES = {
    'video_id':   ['video_id', 'yt_id', 'youtube_id', 'id'],
    'start_time': ['start_time', 'start', 'start_sec', 'begin'],
    'end_time':   ['end_time',   'end',   'end_sec',   'finish'],
    'caption':    ['caption',    'text',  'sentence',  'translation', 'label'],
}


def _resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw CSV columns to canonical names."""
    cols = {c.strip().lower(): c for c in df.columns}
    rename = {}
    for canonical, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in cols and canonical not in rename.values():
                rename[cols[alias]] = canonical
                break
    df = df.rename(columns=rename)
    missing = [c for c in _COL_ALIASES if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find columns {missing} in CSV.  "
            f"Available: {list(df.columns)}.  "
            f"Expected one of: {_COL_ALIASES}"
        )
    return df


def _make_pseudogloss_fn():
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
                       check=True)
        nlp = spacy.load('en_core_web_sm')

    POS_KEEP = {'NOUN', 'VERB', 'ADJ', 'ADV'}

    def _fn(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ''
        doc = nlp(text)
        tokens = [t.text.upper() for t in doc
                  if t.pos_ in POS_KEEP and not t.is_stop and t.is_alpha]
        return ' '.join(tokens) if tokens else text.upper()

    return _fn


def prepare_metadata(raw_csv: Path, out_dir: Path,
                     val_pct: float = 0.10, test_pct: float = 0.10,
                     seed: int = 42) -> dict:
    """
    Load raw metadata, assign SENTENCE_NAME keys, split train/val/test,
    generate PSEUDOGLOSS, and write the three metadata CSVs.

    Returns a dict: {split: DataFrame}.
    """
    meta_out = out_dir / 'metadata'
    meta_out.mkdir(parents=True, exist_ok=True)

    # ── Check if already done ──
    splits_done = all(
        (meta_out / f'youtube_asl_{s}.csv').exists()
        for s in ('train', 'val', 'test')
    )
    if splits_done:
        print('  Metadata CSVs already exist — loading.')
        return {s: pd.read_csv(meta_out / f'youtube_asl_{s}.csv', sep='\t')
                for s in ('train', 'val', 'test')}

    # ── Load + normalise ──
    print(f'  Loading metadata: {raw_csv}')
    sep = '\t' if str(raw_csv).endswith('.tsv') else ','
    df = pd.read_csv(raw_csv, sep=sep)
    df = _resolve_columns(df)
    df = df.dropna(subset=['video_id', 'start_time', 'end_time', 'caption'])
    df['start_time'] = df['start_time'].astype(float)
    df['end_time']   = df['end_time'].astype(float)
    df['caption']    = df['caption'].astype(str).str.strip()
    # Drop degenerate segments
    df = df[df['end_time'] > df['start_time'] + 0.5].copy()

    # ── Assign SENTENCE_NAME: {video_id}_{seg_idx:05d} ──
    # Sort within each video by start time before numbering
    df = df.sort_values(['video_id', 'start_time']).reset_index(drop=True)
    df['seg_idx'] = df.groupby('video_id').cumcount()
    df['SENTENCE_NAME'] = (df['video_id'].astype(str) + '_'
                           + df['seg_idx'].apply(lambda i: f'{i:05d}'))
    df = df.rename(columns={
        'video_id':   'VIDEO_ID',
        'start_time': 'START',
        'end_time':   'END',
        'caption':    'SENTENCE',
    })
    df['SEGMENT_ID'] = df['seg_idx'].astype(int)
    df = df.drop(columns=['seg_idx'])

    # ── Train / val / test split (by video, not by sentence) ──
    rng = np.random.default_rng(seed)
    video_ids = df['VIDEO_ID'].unique()
    rng.shuffle(video_ids)
    n      = len(video_ids)
    n_val  = max(1, int(n * val_pct))
    n_test = max(1, int(n * test_pct))
    val_ids  = set(video_ids[:n_val])
    test_ids = set(video_ids[n_val:n_val + n_test])

    def _split(vid):
        if vid in val_ids:
            return 'val'
        if vid in test_ids:
            return 'test'
        return 'train'

    df['_split'] = df['VIDEO_ID'].map(_split)

    # ── Generate PSEUDOGLOSS ──
    print('  Generating PSEUDOGLOSS via spaCy …')
    pg_fn = _make_pseudogloss_fn()
    df['PSEUDOGLOSS'] = df['SENTENCE'].apply(pg_fn)

    # ── Write CSVs ──
    col_order = ['VIDEO_ID', 'SEGMENT_ID', 'SENTENCE_NAME',
                 'START', 'END', 'SENTENCE', 'PSEUDOGLOSS']
    dfs = {}
    for split in ('train', 'val', 'test'):
        sub = df[df['_split'] == split][col_order].copy()
        sub.to_csv(meta_out / f'youtube_asl_{split}.csv', sep='\t', index=False)
        dfs[split] = sub
        print(f'  [{split}] {len(sub)} segments → {meta_out}/youtube_asl_{split}.csv')

    return dfs


# ── MediaPipe extraction ─────────────────────────────────────────────────────

def extract_holistic_raw(video_path: str) -> np.ndarray:
    """
    Extract raw (T, 543, 3) MediaPipe Holistic landmarks from a video clip.

    Layout:
        [0:33]    pose (33 landmarks)
        [33:501]  face mesh (468 landmarks)
        [501:522] left hand (21 landmarks)
        [522:543] right hand (21 landmarks)
    """
    cap = cv2.VideoCapture(video_path)
    rows = []

    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        model_complexity=1,
    ) as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = holistic.process(rgb)

            row = np.zeros((543, 3), dtype=np.float32)

            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks.landmark):
                    row[i] = (lm.x, lm.y, lm.z)

            if res.face_landmarks:
                for i, lm in enumerate(res.face_landmarks.landmark):
                    if 33 + i < 501:
                        row[33 + i] = (lm.x, lm.y, lm.z)

            if res.left_hand_landmarks:
                for i, lm in enumerate(res.left_hand_landmarks.landmark):
                    row[501 + i] = (lm.x, lm.y, lm.z)

            if res.right_hand_landmarks:
                for i, lm in enumerate(res.right_hand_landmarks.landmark):
                    row[522 + i] = (lm.x, lm.y, lm.z)

            rows.append(row)

    cap.release()
    if not rows:
        return np.zeros((1, 543, 3), dtype=np.float32)
    return np.stack(rows, axis=0)  # (T, 543, 3)


# ── Download helpers ─────────────────────────────────────────────────────────

def _yt_download(video_id: str, out_path: Path, retries: int = 3) -> bool:
    """Download a YouTube video at ≤360p using yt-dlp. Returns True on success."""
    url = f'https://www.youtube.com/watch?v={video_id}'
    cmd = [
        'yt-dlp',
        '--format', 'bestvideo[height<=360][ext=mp4]/bestvideo[height<=360]/best[height<=360]',
        '--no-playlist',
        '--no-audio',
        '--quiet',
        '--output', str(out_path),
        url,
    ]
    for attempt in range(retries):
        try:
            result = subprocess.run(cmd, timeout=300, capture_output=True)
            if result.returncode == 0 and out_path.exists():
                return True
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                time.sleep(10)
    return False


def _ffmpeg_trim(video_path: Path, start: float, end: float,
                 tmp_dir: Path) -> Path:
    """Trim [start, end] seconds from video_path into a temp file."""
    h = hashlib.md5(f'{video_path.name}_{start}_{end}'.encode()).hexdigest()[:8]
    clip_path = tmp_dir / f'clip_{h}.mp4'
    dur = end - start
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', str(start), '-i', str(video_path),
        '-t', str(dur),
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '28',
        '-an',
        str(clip_path),
    ]
    result = subprocess.run(cmd, timeout=120, capture_output=True)
    if result.returncode != 0 or not clip_path.exists():
        return None
    return clip_path


# ── Main pipeline ────────────────────────────────────────────────────────────

def process_shard(dfs: dict, out_dir: Path,
                  shard_idx: int, total_shards: int) -> None:
    """Download + extract keypoints for videos assigned to this shard."""
    videos_tmp = out_dir / 'videos_tmp'
    videos_tmp.mkdir(parents=True, exist_ok=True)

    # Collect all rows across all splits
    all_rows = []
    for split, df in dfs.items():
        for _, row in df.iterrows():
            all_rows.append((split, row))

    # Group by video_id to minimise downloads
    from collections import defaultdict
    by_video = defaultdict(list)
    for split, row in all_rows:
        by_video[row['VIDEO_ID']].append((split, row))

    video_ids = sorted(by_video.keys())
    # Assign shards by video so each video is handled by exactly one shard
    shard_videos = [v for i, v in enumerate(video_ids) if i % total_shards == shard_idx]
    print(f'  Shard {shard_idx}/{total_shards}: '
          f'{len(shard_videos)} videos / {sum(len(by_video[v]) for v in shard_videos)} segments')

    # Ensure output dirs exist
    for split in dfs:
        (out_dir / 'keypoints' / split).mkdir(parents=True, exist_ok=True)

    done = skipped = errors = 0

    for video_id in tqdm(shard_videos, desc=f'shard-{shard_idx}', unit='video'):
        segments = by_video[video_id]

        # Check which segments still need extraction
        pending = []
        for split, row in segments:
            npy_path = out_dir / 'keypoints' / split / f'{row["SENTENCE_NAME"]}.npy'
            if not npy_path.exists():
                pending.append((split, row, npy_path))

        if not pending:
            skipped += len(segments)
            continue

        # Download video
        video_path = videos_tmp / f'{video_id}.mp4'
        if not video_path.exists():
            ok = _yt_download(video_id, video_path)
            if not ok:
                tqdm.write(f'  SKIP (download failed): {video_id}')
                errors += len(pending)
                continue

        # Process each pending segment
        with tempfile.TemporaryDirectory(dir=videos_tmp) as tmp_dir:
            tmp_dir = Path(tmp_dir)
            for split, row, npy_path in pending:
                try:
                    clip_path = _ffmpeg_trim(
                        video_path,
                        float(row['START']),
                        float(row['END']),
                        tmp_dir,
                    )
                    if clip_path is None:
                        tqdm.write(f'  SKIP (ffmpeg failed): {row["SENTENCE_NAME"]}')
                        errors += 1
                        continue

                    arr = extract_holistic_raw(str(clip_path))   # (T, 543, 3)
                    np.save(npy_path, arr)
                    clip_path.unlink(missing_ok=True)
                    done += 1
                except Exception as exc:
                    tqdm.write(f'  ERROR {row["SENTENCE_NAME"]}: {exc}')
                    errors += 1

        # Delete video once all segments are processed
        video_path.unlink(missing_ok=True)

    print(f'  Shard {shard_idx} complete: done={done}, skipped={skipped}, errors={errors}')


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube-ASL videos and extract MediaPipe keypoints.'
    )
    parser.add_argument('--metadata',      type=str, required=True,
                        help='Path to raw YouTube-ASL metadata CSV/TSV')
    parser.add_argument('--out_dir',       type=str, default='/data/youtube_asl',
                        help='Root output directory')
    parser.add_argument('--shard_idx',     type=int, default=0,
                        help='0-based shard index for this worker')
    parser.add_argument('--total_shards',  type=int, default=1,
                        help='Total number of parallel shards')
    parser.add_argument('--val_pct',       type=float, default=0.10)
    parser.add_argument('--test_pct',      type=float, default=0.10)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--prepare_only',  action='store_true',
                        help='Only prepare metadata CSVs, do not download videos')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=== Step 1: Metadata ===')
    dfs = prepare_metadata(
        Path(args.metadata), out_dir,
        val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed,
    )

    if args.prepare_only:
        print('--prepare_only set, stopping after metadata.')
        return

    print(f'\n=== Step 2: Download + extract (shard {args.shard_idx}/{args.total_shards}) ===')
    process_shard(dfs, out_dir, args.shard_idx, args.total_shards)

    print('\nDone.')
    print(f'  Metadata  : {out_dir}/metadata/')
    print(f'  Keypoints : {out_dir}/keypoints/{{train,val,test}}/')


if __name__ == '__main__':
    main()
