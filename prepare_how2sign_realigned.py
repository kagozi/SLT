"""
Prepare How2Sign with manually re-aligned clips.

Pipeline:
  1. Download re-aligned annotation CSVs (tiny, ~1 MB each)
  2. Download full Green Screen RGB videos (frontal) via gdown
     Train: 10 split-zip parts → progressive-cat → unzip → delete zips
     Val/Test: single zip each
  3. Re-segment per-sentence clips with ffmpeg at re-aligned timestamps
  4. Delete full videos to reclaim ~50-80 GB
  5. Wipe stale keypoints cache
  6. Re-run MediaPipe extraction on new clips

NOTE: This script is OBSOLETE. The training pipeline now reads directly from
the HF cache at /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features/.
No re-segmentation or keypoint re-extraction is needed.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ── Google Drive file IDs ────────────────────────────────────────────────────
REALIGNED_IDS = {
    'train': '1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok',
    'val':   '1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI',
    'test':  '1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu',
}

# Train = 10 split-zip parts (z01–z09 + .zip); val & test = single zip
FULL_VIDEO_IDS = {
    'train_parts': [
        ('1xWlMM2O3Gbp_8LK5FefoH0TVEmae6jIf', 'train_raw.z01'),
        ('1krtYdpK_LQFgEUCnHxoYAW7EyhLMLWq0', 'train_raw.z02'),
        ('1fXpWRNFhpuVm3ym7lT9vF_bnDjHkvP_K', 'train_raw.z03'),
        ('1IFetFt4AzsxNCMZ0VVpX7YRgFAm58X48', 'train_raw.z04'),
        ('1ZHuuun6Ae-AOLBns3LmuH7w8C9YCB4gH', 'train_raw.z05'),
        ('1FQQIPblk-oLH_vu7h2tDO0oJaZ3xkp5N', 'train_raw.z06'),
        ('19XNgERcolGAMPPgX-Gx_GebSTx3W4o0r', 'train_raw.z07'),
        ('1YN-SA9uzrogEdKeT6UdQUIcuGEyYJILg', 'train_raw.z08'),
        ('1SZQ2GzPLCkRqvsImAjULAPBiuAKi9DE9', 'train_raw.z09'),
        ('1Xe1T5okJiopMXUiH3sc0mdCWNDYSBopd', 'train_raw.zip'),
    ],
    'val':  ('1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g', 'val_raw.zip'),
    'test': ('1z0i6BBGHQ12ChY63hZH56QnczvQ0JfTb', 'test_raw.zip'),
}

ROOT        = Path('/data/hf_cache/How2Sign_Holistic/how2sign_holistic_features')
FULL_VIDEOS = Path('/data/how2sign_full_videos')   # temp; deleted after re-segmentation
ANNO_DIR    = ROOT / 'annotations'
KPS_DIR     = ROOT / 'keypoints'


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd, **kw):
    print(f'  $ {cmd}', flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def gdown_file(file_id: str, out_path: Path) -> None:
    """Download a single Google Drive file with gdown (no --fuzzy with --id)."""
    if out_path.exists():
        print(f'  already exists: {out_path.name}', flush=True)
        return
    # Use Python gdown API directly — more reliable than CLI for large files
    import gdown
    url = f'https://drive.google.com/uc?id={file_id}'
    print(f'  gdown {url} → {out_path.name}', flush=True)
    gdown.download(url, str(out_path), quiet=False, fuzzy=False)


def df_space_gb(path: Path) -> float:
    r = shutil.disk_usage(path)
    return r.free / 1e9


# ── Stage 1: Re-aligned CSVs ─────────────────────────────────────────────────

def download_realigned_csvs(tmp: Path) -> dict:
    """Download the three re-aligned CSV files; return {split: Path}."""
    print('\n=== Stage 1: Download re-aligned annotation CSVs ===')
    tmp.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, fid in REALIGNED_IDS.items():
        out = tmp / f'how2sign_realigned_{split}.csv'
        gdown_file(fid, out)
        paths[split] = out
    return paths


# ── Stage 2: Full video download + extraction ─────────────────────────────────

def download_and_extract_val_test(tmp: Path) -> None:
    """Download val and test full videos (single zip each) and extract."""
    for split in ('val', 'test'):
        fid, fname = FULL_VIDEO_IDS[split]
        zip_path = tmp / fname
        out_dir  = FULL_VIDEOS / split
        done_flag = out_dir / '.done'
        if done_flag.exists():
            print(f'  [{split}] already extracted, skipping')
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n--- Downloading {split} full videos ({fname}) ---')
        gdown_file(fid, zip_path)
        print(f'  Extracting {zip_path.name} → {out_dir}')
        run(f'unzip -q "{zip_path}" -d "{out_dir}"')
        zip_path.unlink()
        done_flag.touch()
        print(f'  [{split}] free space: {df_space_gb(Path("/data")):.1f} GB')


def download_and_extract_train(tmp: Path) -> None:
    """
    Download 10 train split-zip parts, progressively cat into one archive,
    then unzip.  Peak extra space = combined.zip + 1 part at a time.
    """
    out_dir   = FULL_VIDEOS / 'train'
    done_flag = out_dir / '.done'
    if done_flag.exists():
        print('  [train] already extracted, skipping')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    combined = tmp / 'train_raw_combined.zip'

    print('\n--- Downloading train full videos (10 parts, progressive cat) ---')
    with open(combined, 'wb') as cf:
        for fid, fname in FULL_VIDEO_IDS['train_parts']:
            part = tmp / fname
            print(f'  downloading part: {fname}')
            gdown_file(fid, part)
            print(f'  appending {fname} to combined archive...')
            with open(part, 'rb') as pf:
                shutil.copyfileobj(pf, cf, length=64 * 1024 * 1024)  # 64 MB chunks
            part.unlink()
            print(f'  free space: {df_space_gb(Path("/data")):.1f} GB')

    print(f'  Extracting combined archive ({combined.stat().st_size / 1e9:.1f} GB) → {out_dir}')
    run(f'unzip -q "{combined}" -d "{out_dir}"')
    combined.unlink()
    done_flag.touch()
    print(f'  [train] free space after extraction: {df_space_gb(Path("/data")):.1f} GB')


# ── Stage 3: Re-segment clips with ffmpeg ─────────────────────────────────────

def resegment_split(split: str, realigned_csv: Path) -> int:
    """
    For every row in the re-aligned CSV, cut the sentence clip from the
    full (unsegmented) video and save to ROOT/{split}/{SENTENCE_NAME}.mp4.
    Returns number of clips written.
    """
    print(f'\n=== Stage 3: Re-segment {split} clips ===')

    df = pd.read_csv(realigned_csv, sep='\t')
    # Normalise column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    video_src_dir = FULL_VIDEOS / split
    clip_out_dir  = ROOT / split
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    written = skipped = errors = 0

    for _, row in df.iterrows():
        video_name    = str(row['VIDEO_NAME']).strip()
        sentence_name = str(row['SENTENCE_NAME']).strip()
        start         = float(row['START'])
        end           = float(row['END'])

        out_clip = clip_out_dir / f'{sentence_name}.mp4'
        if out_clip.exists():
            skipped += 1
            continue

        # Full video could be directly in split dir or in a sub-dir
        candidates = list(video_src_dir.rglob(f'{video_name}.mp4'))
        if not candidates:
            print(f'  WARNING: full video not found: {video_name}.mp4')
            errors += 1
            continue

        full_video = candidates[0]
        duration   = end - start
        # -avoid_negative_ts 1 ensures proper pts for downstream tools
        cmd = (
            f'ffmpeg -y -ss {start:.3f} -i "{full_video}" '
            f'-t {duration:.3f} -c copy -avoid_negative_ts 1 '
            f'"{out_clip}" -loglevel error'
        )
        result = subprocess.run(cmd, shell=True)
        if result.returncode == 0:
            written += 1
        else:
            errors += 1

    print(f'  [{split}] written={written}, skipped={skipped}, errors={errors}')
    return written


# ── Stage 4: Delete full videos ───────────────────────────────────────────────

def delete_full_videos() -> None:
    print('\n=== Stage 4: Delete full videos to reclaim space ===')
    if FULL_VIDEOS.exists():
        shutil.rmtree(FULL_VIDEOS)
        print(f'  Deleted {FULL_VIDEOS}')
    print(f'  Free space: {df_space_gb(Path("/data")):.1f} GB')


# ── Stage 5: Update annotation CSVs ──────────────────────────────────────────

def update_annotations(realigned_paths: dict) -> None:
    """
    Replace the how2sign_{split}.csv files with re-aligned versions.
    Regenerate PSEUDOGLOSS column using spaCy POS extraction.
    """
    print('\n=== Stage 5: Update annotation CSVs ===')
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        run('python -m spacy download en_core_web_sm')
        nlp = spacy.load('en_core_web_sm')

    POS_KEEP = {'NOUN', 'VERB', 'ADJ', 'ADV'}

    def pseudogloss(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ''
        doc = nlp(text)
        tokens = [t.text.upper() for t in doc
                  if t.pos_ in POS_KEEP and not t.is_stop and t.is_alpha]
        return ' '.join(tokens) if tokens else text.upper()

    for split, csv_path in realigned_paths.items():
        df = pd.read_csv(csv_path, sep='\t')
        df.columns = [c.strip() for c in df.columns]
        if 'PSEUDOGLOSS' not in df.columns:
            print(f'  [{split}] generating PSEUDOGLOSS column ...')
            df['PSEUDOGLOSS'] = df['SENTENCE'].apply(pseudogloss)
        out = ANNO_DIR / f'how2sign_{split}.csv'
        df.to_csv(out, sep='\t', index=False)
        print(f'  [{split}] saved {len(df)} rows → {out}')


# ── Stage 6: Wipe stale keypoints ─────────────────────────────────────────────

def wipe_keypoints() -> None:
    print('\n=== Stage 6: Delete stale keypoint cache ===')
    if KPS_DIR.exists():
        shutil.rmtree(KPS_DIR)
        print(f'  Deleted {KPS_DIR}')
    print(f'  Free space: {df_space_gb(Path("/data")):.1f} GB')


# ── Stage 7: Re-extract keypoints ─────────────────────────────────────────────

def reextract_keypoints(max_frames: int) -> None:
    print('\n=== Stage 7: Re-extract MediaPipe keypoints ===')
    run(
        f'python preextract_keypoints_how2sign.py '
        f'--root_dir "{ROOT}" --max_frames {max_frames} '
        f'--splits train val test'
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Download How2Sign full videos and re-segment with re-aligned timestamps.'
    )
    parser.add_argument('--max_frames', type=int, default=300)
    parser.add_argument('--tmp_dir',    type=str, default='/data/how2sign_tmp')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip video download (full videos already in /data/how2sign_full_videos)')
    parser.add_argument('--skip_resegment', action='store_true',
                        help='Skip re-segmentation (clips already in ROOT/{split}/)')
    parser.add_argument('--skip_keypoints', action='store_true',
                        help='Skip keypoint re-extraction')
    args = parser.parse_args()

    tmp = Path(args.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    ANNO_DIR.mkdir(parents=True, exist_ok=True)

    # Install / upgrade gdown
    print('Installing/upgrading gdown ...', flush=True)
    run('pip install --upgrade gdown -q')

    print(f'\nPVC free space at start: {df_space_gb(Path("/data")):.1f} GB')

    # Stage 1
    realigned_paths = download_realigned_csvs(tmp)

    if not args.skip_download:
        # Stage 2
        download_and_extract_val_test(tmp)
        download_and_extract_train(tmp)
    else:
        print('\n[skip_download] Assuming full videos already in /data/how2sign_full_videos')

    if not args.skip_resegment:
        # Stage 3
        for split in ('train', 'val', 'test'):
            resegment_split(split, realigned_paths[split])
        # Stage 4
        delete_full_videos()
    else:
        print('\n[skip_resegment] Using existing clips in ROOT/{split}/')

    # Stage 5
    update_annotations(realigned_paths)

    # Clean up tmp dir
    if tmp.exists():
        shutil.rmtree(tmp)

    if not args.skip_keypoints:
        # Stage 6
        wipe_keypoints()
        # Stage 7
        reextract_keypoints(args.max_frames)

    print('\n✅ How2Sign re-aligned pipeline complete.')
    print(f'   Clips     : {ROOT}/{{train,val,test}}/')
    print(f'   Keypoints : {KPS_DIR}/{{train,val,test}}/')
    print(f'   PVC free  : {df_space_gb(Path("/data")):.1f} GB')


if __name__ == '__main__':
    main()
