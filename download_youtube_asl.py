"""
YouTube-ASL download + MediaPipe Holistic extraction pipeline.

The YouTube-ASL dataset (gs://gresearch/youtube-asl) provides only a
plain-text list of YouTube video IDs.  Captions/timings are fetched
directly from each video's subtitle track using yt-dlp, then the video
is downloaded, trimmed per caption segment, keypoints are extracted,
and the video is deleted.

MediaPipe Holistic layout saved (543 landmarks, same as How2Sign_Holistic):
    Pose      :   0 -  32  (33 landmarks)
    Face mesh :  33 - 500  (468 landmarks)
    Left hand : 501 - 521  (21 landmarks)
    Right hand: 522 - 542  (21 landmarks)

Input:
    --video_ids  plain-text file, one YouTube video ID per line

Output layout:
    <out_dir>/
        metadata/
            youtube_asl_train.csv    (tab-sep, with PSEUDOGLOSS)
            youtube_asl_val.csv
            youtube_asl_test.csv
        keypoints/
            train/{video_id}_{seg_idx:05d}.npy   # (T, 543, 3) float32
            val/
            test/

Usage — fetch captions + build metadata only (no video download):
    python download_youtube_asl.py \\
        --video_ids /data/youtube_asl/video_ids.txt \\
        --out_dir   /data/youtube_asl \\
        --prepare_only

Usage — full pipeline, shard 0 of 20:
    python download_youtube_asl.py \\
        --video_ids /data/youtube_asl/video_ids.txt \\
        --out_dir   /data/youtube_asl \\
        --shard_idx 0 --total_shards 20
"""

import argparse
import hashlib
import re
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

from dataset_how2sign import process_holistic


# ── VTT / subtitle parsing ───────────────────────────────────────────────────

_TIMESTAMP_RE = re.compile(
    r'(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*'
    r'(\d{2}):(\d{2}):(\d{2})[.,](\d{3})'
)
_TAG_RE = re.compile(r'<[^>]+>')


def _ts_to_sec(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(vtt_text: str, min_dur: float = 0.5) -> list:
    """
    Parse a WebVTT subtitle string into a list of
    {'start': float, 'end': float, 'text': str} dicts.

    Merges cue continuations (same timestamp block, multiple lines).
    Skips cues shorter than min_dur seconds or with empty text.
    """
    segments = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        m = _TIMESTAMP_RE.search(lines[i])
        if m:
            start = _ts_to_sec(*m.group(1, 2, 3, 4))
            end   = _ts_to_sec(*m.group(5, 6, 7, 8))
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip() and not _TIMESTAMP_RE.search(lines[i]):
                clean = _TAG_RE.sub('', lines[i]).strip()
                if clean:
                    text_lines.append(clean)
                i += 1
            text = ' '.join(text_lines).strip()
            if text and (end - start) >= min_dur:
                segments.append({'start': start, 'end': end, 'text': text})
        else:
            i += 1
    return segments


# ── Caption fetching ─────────────────────────────────────────────────────────

def fetch_captions(video_id: str, tmp_dir: Path,
                   retries: int = 3) -> list:
    """
    Download subtitle/caption VTT for a YouTube video using yt-dlp.
    Tries manual subs first (en), falls back to auto-generated (en).
    Returns a list of segment dicts from parse_vtt(), or [] on failure.
    """
    url = f'https://www.youtube.com/watch?v={video_id}'

    for sub_flag in (['--write-subs'], ['--write-auto-subs']):
        vtt_glob = list(tmp_dir.glob(f'{video_id}*.vtt'))
        if vtt_glob:
            break  # already downloaded on a previous attempt

        cmd = [
            'yt-dlp',
            *sub_flag,
            '--sub-langs', 'en',
            '--skip-download',
            '--no-playlist',
            '--quiet',
            '--output', str(tmp_dir / video_id),
            url,
        ]
        for attempt in range(retries):
            try:
                subprocess.run(cmd, timeout=60, capture_output=True)
                break
            except subprocess.TimeoutExpired:
                if attempt < retries - 1:
                    time.sleep(5)

        vtt_glob = list(tmp_dir.glob(f'{video_id}*.vtt'))
        if vtt_glob:
            break

    vtt_files = list(tmp_dir.glob(f'{video_id}*.vtt'))
    if not vtt_files:
        return []

    text = vtt_files[0].read_text(encoding='utf-8', errors='replace')
    return parse_vtt(text)


# ── Metadata preparation ─────────────────────────────────────────────────────

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


def prepare_metadata(video_ids_file: Path, out_dir: Path,
                     val_pct: float = 0.10, test_pct: float = 0.10,
                     seed: int = 42) -> dict:
    """
    For every video ID:
      1. Fetch captions via yt-dlp (cached — skips if already done).
      2. Build a DataFrame with one row per caption segment.
      3. Split train/val/test by video (not by segment).
      4. Generate PSEUDOGLOSS with spaCy.
      5. Write youtube_asl_{train,val,test}.csv.

    Returns {split: DataFrame}.
    """
    meta_out = out_dir / 'metadata'
    meta_out.mkdir(parents=True, exist_ok=True)

    splits_done = all(
        (meta_out / f'youtube_asl_{s}.csv').exists()
        for s in ('train', 'val', 'test')
    )
    if splits_done:
        print('  Metadata CSVs already exist — loading.')
        return {s: pd.read_csv(meta_out / f'youtube_asl_{s}.csv', sep='\t')
                for s in ('train', 'val', 'test')}

    # ── Load video IDs ──
    raw_ids = [l.strip() for l in video_ids_file.read_text().splitlines()
               if l.strip() and not l.startswith('#')]
    print(f'  {len(raw_ids)} video IDs loaded from {video_ids_file}')

    # ── Fetch captions (resumable via captions_cache/) ──
    cache_dir = out_dir / 'captions_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    failed_caption = 0

    for video_id in tqdm(raw_ids, desc='fetch captions', unit='video'):
        cache_file = cache_dir / f'{video_id}.json'
        if cache_file.exists():
            cached = pd.read_json(cache_file, orient='records')
            if len(cached):
                rows.append(cached)
            continue

        with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
            segs = fetch_captions(video_id, Path(tmp))

        if not segs:
            failed_caption += 1
            # Write empty JSON so we don't retry this video
            cache_file.write_text('[]')
            continue

        seg_df = pd.DataFrame([
            {'video_id': video_id, 'seg_idx': i,
             'start': s['start'], 'end': s['end'], 'caption': s['text']}
            for i, s in enumerate(segs)
        ])
        cache_file.write_text(seg_df.to_json(orient='records'))
        rows.append(seg_df)

    print(f'  Captions fetched. Failed: {failed_caption}/{len(raw_ids)} videos')

    if not rows:
        raise RuntimeError('No captions found for any video. '
                           'Check that video IDs are valid and public.')

    df = pd.concat([r for r in rows if len(r)], ignore_index=True)
    df = df.dropna(subset=['caption'])
    df['caption'] = df['caption'].astype(str).str.strip()
    df = df[df['caption'] != ''].copy()

    # ── SENTENCE_NAME ──
    df = df.sort_values(['video_id', 'seg_idx']).reset_index(drop=True)
    df['SENTENCE_NAME'] = (df['video_id'].astype(str) + '_'
                           + df['seg_idx'].apply(lambda i: f'{i:05d}'))
    df = df.rename(columns={
        'video_id': 'VIDEO_ID',
        'seg_idx':  'SEGMENT_ID',
        'start':    'START',
        'end':      'END',
        'caption':  'SENTENCE',
    })

    # ── Train / val / test split by video ──
    rng = np.random.default_rng(seed)
    video_ids_all = df['VIDEO_ID'].unique()
    rng.shuffle(video_ids_all)
    n      = len(video_ids_all)
    n_val  = max(1, int(n * val_pct))
    n_test = max(1, int(n * test_pct))
    val_ids  = set(video_ids_all[:n_val])
    test_ids = set(video_ids_all[n_val:n_val + n_test])

    def _assign_split(vid):
        if vid in val_ids:
            return 'val'
        if vid in test_ids:
            return 'test'
        return 'train'

    df['_split'] = df['VIDEO_ID'].map(_assign_split)

    # ── PSEUDOGLOSS ──
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
        '-c:v', 'copy',
        '-an',
        '-avoid_negative_ts', 'make_zero',
        str(clip_path),
    ]
    result = subprocess.run(cmd, timeout=120, capture_output=True)
    if result.returncode != 0 or not clip_path.exists():
        stderr = result.stderr.decode('utf-8', errors='replace').strip()
        if stderr:
            tqdm.write(f'  ffmpeg error for {video_path.name} [{start}-{end}]: {stderr[:200]}')
        return None
    return clip_path


# ── Main extraction pipeline ─────────────────────────────────────────────────

def process_shard(dfs: dict, out_dir: Path,
                  shard_idx: int, total_shards: int) -> None:
    """Download + extract keypoints for videos assigned to this shard."""
    videos_tmp = out_dir / 'videos_tmp'
    videos_tmp.mkdir(parents=True, exist_ok=True)

    from collections import defaultdict
    by_video = defaultdict(list)
    for split, df in dfs.items():
        for _, row in df.iterrows():
            by_video[row['VIDEO_ID']].append((split, row))

    video_ids = sorted(by_video.keys())
    shard_videos = [v for i, v in enumerate(video_ids) if i % total_shards == shard_idx]
    print(f'  Shard {shard_idx}/{total_shards}: '
          f'{len(shard_videos)} videos / '
          f'{sum(len(by_video[v]) for v in shard_videos)} segments')

    for split in dfs:
        (out_dir / 'keypoints' / split).mkdir(parents=True, exist_ok=True)

    done = skipped = errors = 0

    for video_id in tqdm(shard_videos, desc=f'shard-{shard_idx}', unit='video'):
        segments = by_video[video_id]

        pending = []
        for split, row in segments:
            npy_path = out_dir / 'keypoints' / split / f'{row["SENTENCE_NAME"]}.npy'
            if not npy_path.exists():
                pending.append((split, row, npy_path))

        if not pending:
            skipped += len(segments)
            continue

        video_path = videos_tmp / f'{video_id}.mp4'
        if not video_path.exists():
            ok = _yt_download(video_id, video_path)
            if not ok:
                tqdm.write(f'  SKIP (download failed): {video_id}')
                errors += len(pending)
                continue

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

                    raw = extract_holistic_raw(str(clip_path))   # (T, 543, 3)
                    T = raw.shape[0]
                    arr = process_holistic(raw, T)               # (T, 225) no padding
                    np.save(npy_path, arr)
                    clip_path.unlink(missing_ok=True)
                    done += 1
                except Exception as exc:
                    tqdm.write(f'  ERROR {row["SENTENCE_NAME"]}: {exc}')
                    errors += 1

        video_path.unlink(missing_ok=True)

    print(f'  Shard {shard_idx} complete: done={done}, skipped={skipped}, errors={errors}')


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download YouTube-ASL videos and extract MediaPipe keypoints.'
    )
    parser.add_argument('--video_ids',     type=str, required=True,
                        help='Path to plain-text file with one YouTube video ID per line')
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
                        help='Only fetch captions and build metadata CSVs; skip video download')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=== Step 1: Fetch captions + build metadata ===')
    dfs = prepare_metadata(
        Path(args.video_ids), out_dir,
        val_pct=args.val_pct, test_pct=args.test_pct, seed=args.seed,
    )

    if args.prepare_only:
        print('--prepare_only set, stopping after metadata.')
        return

    print(f'\n=== Step 2: Download + extract keypoints '
          f'(shard {args.shard_idx}/{args.total_shards}) ===')
    process_shard(dfs, out_dir, args.shard_idx, args.total_shards)

    print('\nDone.')
    print(f'  Metadata  : {out_dir}/metadata/')
    print(f'  Keypoints : {out_dir}/keypoints/{{train,val,test}}/')


if __name__ == '__main__':
    main()
