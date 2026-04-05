"""
Dataset analysis and W&B logging for PHOENIX-2014-T and How2Sign.

Logs to W&B (project="slt", run="dataset-analysis"):
  - Samples table: RGB frame, keypoint visualisation, sentence, gloss/pseudogloss
  - Per-split statistics: frame counts, gloss/sentence lengths, vocab size,
    missing-keypoint rate, zero-frame rate

Usage (NRP job):
    kubectl apply -f nautilius/analyze-datasets-job.yaml

Usage (local):
    python analyze_datasets.py \
        --phoenix_dir /data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --how2sign_dir /data/how2sign_hf
"""

import argparse
import random
import io
from pathlib import Path
from collections import Counter

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from PIL import Image


# ── Joint connectivity for skeleton drawing ──────────────────────────────────
# Indices into the 75-joint layout:
#   0-12  : pose (13 joints)
#   13-33 : left hand (21 joints)
#   34-54 : right hand (21 joints)
#   55-74 : face (20 joints)

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head/neck
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders / arms
    (5, 11), (6, 12), (11, 12),                # torso
]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
]
FACE_CONNECTIONS = [
    (0,1),(2,3),(4,5),(6,7),   # corners/lips
]

N_SAMPLES  = 20   # visual samples per dataset
N_STAT_CAP = None  # use all rows for statistics


# ── Keypoint visualisation ────────────────────────────────────────────────────

def draw_keypoints(kps_225: np.ndarray, frame_hw=(256, 256)) -> np.ndarray:
    """
    Render a single frame of 225-d keypoints as a skeleton image.
    kps_225: (225,) float32 in [0, 1] (normalised x/y, plus z ignored for 2-D vis)
    Returns (H, W, 3) uint8 BGR.
    """
    H, W = frame_hw
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    pts = kps_225.reshape(75, 3)  # (75, 3) — x, y, z in [0,1]

    def to_px(idx):
        x, y = pts[idx, 0], pts[idx, 1]
        return int(x * W), int(y * H)

    def draw_links(connections, offset, color, thickness=1):
        for a, b in connections:
            pa = to_px(offset + a)
            pb = to_px(offset + b)
            if all(0 <= v < W for v in [pa[0], pb[0]]) and \
               all(0 <= v < H for v in [pa[1], pb[1]]):
                cv2.line(canvas, pa, pb, color, thickness)

    def draw_dots(start, end, color, radius=3):
        for i in range(start, end):
            px = to_px(i)
            if 0 <= px[0] < W and 0 <= px[1] < H:
                cv2.circle(canvas, px, radius, color, -1)

    # Pose (green), left hand (cyan), right hand (orange), face (yellow)
    draw_links(POSE_CONNECTIONS,  0,  (0, 200, 0),   2)
    draw_links(HAND_CONNECTIONS,  13, (0, 255, 255), 1)
    draw_links(HAND_CONNECTIONS,  34, (0, 165, 255), 1)
    draw_links(FACE_CONNECTIONS,  55, (0, 220, 220), 1)

    draw_dots(0,  13, (0, 230, 0),   4)   # pose
    draw_dots(13, 34, (0, 255, 255), 3)   # left hand
    draw_dots(34, 55, (0, 165, 255), 3)   # right hand
    draw_dots(55, 75, (255, 220, 0), 2)   # face

    return canvas


def best_frame(kps: np.ndarray) -> np.ndarray:
    """Return the frame with the most non-zero keypoints."""
    nonzero = (np.abs(kps).sum(axis=1) > 0)  # (T,)
    norms   = np.linalg.norm(kps, axis=1)
    idx = int(np.argmax(norms * nonzero)) if nonzero.any() else 0
    return kps[idx]


def kps_to_wandb_image(kps: np.ndarray) -> wandb.Image:
    frame = best_frame(kps)
    canvas = draw_keypoints(frame)
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return wandb.Image(img_rgb)


def video_frame_to_wandb_image(video_path: str, target_frame: int = 0) -> wandb.Image:
    """Extract one frame from an mp4 and return as wandb.Image (or blank on failure)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seek  = min(target_frame, max(0, total // 2))   # middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, seek)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return wandb.Image(rgb)


# ── Statistics helpers ────────────────────────────────────────────────────────

def gloss_lengths(texts):
    return [len(str(t).split()) for t in texts if pd.notna(t) and str(t).strip()]


def build_vocab(texts):
    vocab = Counter()
    for t in texts:
        if pd.notna(t):
            vocab.update(str(t).upper().split())
    return vocab


def frame_count_stats(npy_dir: Path):
    """Return list of frame counts (non-zero rows) for every .npy in dir."""
    counts = []
    zero_files = 0
    for f in npy_dir.glob("*.npy"):
        kps = np.load(f)
        nonzero = int((np.linalg.norm(kps, axis=-1) > 0).sum())
        counts.append(nonzero)
        if nonzero == 0:
            zero_files += 1
    return counts, zero_files


def hist_image(values, title, xlabel, color='steelblue') -> wandb.Image:
    """Return a wandb.Image of a matplotlib histogram."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(values, bins=40, color=color, edgecolor='white', linewidth=0.4)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(Image.open(buf).copy())


# ── PHOENIX analysis ──────────────────────────────────────────────────────────

def analyze_phoenix(root_dir: Path):
    print("\n── PHOENIX-2014-T ──")
    splits = ['train', 'dev', 'test']
    csv_map = {
        'train': root_dir / 'annotations/manual/PHOENIX-2014-T.train.corpus.csv',
        'dev':   root_dir / 'annotations/manual/PHOENIX-2014-T.dev.corpus.csv',
        'test':  root_dir / 'annotations/manual/PHOENIX-2014-T.test.corpus.csv',
    }
    kps_root = root_dir / 'features' / 'keypoints'

    sample_table = wandb.Table(columns=[
        'dataset', 'split', 'name',
        'rgb_frame', 'keypoints_vis',
        'sentence', 'gloss',
        'gloss_len', 'sentence_len', 'n_frames',
    ])

    stat_table = wandb.Table(columns=[
        'dataset', 'split', 'n_samples',
        'avg_gloss_len', 'max_gloss_len',
        'avg_sentence_len', 'max_sentence_len',
        'vocab_size_gloss', 'vocab_size_sentence',
        'avg_frames', 'median_frames', 'max_frames',
        'zero_frame_files', 'missing_npy',
    ])

    charts = {}

    for split in splits:
        csv_path = csv_map[split]
        if not csv_path.exists():
            print(f"  [{split}] CSV not found: {csv_path}, skipping.")
            continue

        df = pd.read_csv(csv_path, sep='|')
        # Standard PHOENIX CSV columns: name, video, start, end, speaker, orth, translation
        name_col   = 'name'
        gloss_col  = 'orth'
        sent_col   = 'translation'

        print(f"  [{split}] {len(df)} samples")

        frame_dir = root_dir / 'features' / 'fullFrame-210x260px' / split

        # ── Samples ──
        indices = random.sample(range(len(df)), min(N_SAMPLES, len(df)))
        for i in indices:
            row  = df.iloc[i]
            name = str(row[name_col])
            gloss_text = str(row[gloss_col]) if pd.notna(row.get(gloss_col)) else ''
            sent_text  = str(row[sent_col])  if pd.notna(row.get(sent_col))  else ''

            npy_path = kps_root / split / f"{name}.npy"
            kps_img  = None
            n_frames = 0
            if npy_path.exists():
                kps = np.load(npy_path)
                kps_img  = kps_to_wandb_image(kps)
                n_frames = int((np.linalg.norm(kps, axis=-1) > 0).sum())

            # Try to find an RGB frame from the original images
            vid_subdir = frame_dir / name
            rgb_img    = None
            if vid_subdir.exists():
                frames = sorted(vid_subdir.glob('*.png'))
                if frames:
                    mid   = frames[len(frames) // 2]
                    img   = cv2.imread(str(mid))
                    if img is not None:
                        rgb_img = wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            sample_table.add_data(
                'phoenix', split, name,
                rgb_img, kps_img,
                sent_text, gloss_text,
                len(gloss_text.split()), len(sent_text.split()), n_frames,
            )

        # ── Statistics ──
        gloss_lens    = gloss_lengths(df[gloss_col].tolist())
        sent_lens     = gloss_lengths(df[sent_col].tolist())
        gloss_vocab   = build_vocab(df[gloss_col].tolist())
        sent_vocab    = build_vocab(df[sent_col].tolist())

        frame_counts, zero_files = [], 0
        missing_npy = 0
        for _, row in df.iterrows():
            p = kps_root / split / f"{row[name_col]}.npy"
            if p.exists():
                kps = np.load(p)
                fc  = int((np.linalg.norm(kps, axis=-1) > 0).sum())
                frame_counts.append(fc)
                if fc == 0:
                    zero_files += 1
            else:
                missing_npy += 1

        if frame_counts:
            charts[f'phoenix/{split}/frame_dist'] = hist_image(
                frame_counts, f'PHOENIX {split} — frame counts', 'frames', 'steelblue')
        if gloss_lens:
            charts[f'phoenix/{split}/gloss_len_dist'] = hist_image(
                gloss_lens, f'PHOENIX {split} — gloss lengths', 'tokens', 'seagreen')

        stat_table.add_data(
            'phoenix', split, len(df),
            round(np.mean(gloss_lens), 2) if gloss_lens else 0,
            max(gloss_lens) if gloss_lens else 0,
            round(np.mean(sent_lens), 2) if sent_lens else 0,
            max(sent_lens) if sent_lens else 0,
            len(gloss_vocab), len(sent_vocab),
            round(np.mean(frame_counts), 1) if frame_counts else 0,
            int(np.median(frame_counts)) if frame_counts else 0,
            max(frame_counts) if frame_counts else 0,
            zero_files, missing_npy,
        )
        avg_frames_str = f"{np.mean(frame_counts):.1f}" if frame_counts else "N/A"
        print(f"  [{split}] gloss vocab={len(gloss_vocab)}, "
              f"avg_gloss_len={np.mean(gloss_lens):.1f}, "
              f"avg_frames={avg_frames_str}")

    return sample_table, stat_table, charts


# ── How2Sign analysis ─────────────────────────────────────────────────────────

def analyze_how2sign(root_dir: Path):
    print("\n── How2Sign ──")
    splits = ['train', 'val', 'test']

    sample_table = wandb.Table(columns=[
        'dataset', 'split', 'name',
        'rgb_frame', 'keypoints_vis',
        'sentence', 'pseudogloss',
        'gloss_len', 'sentence_len', 'n_frames',
    ])

    stat_table = wandb.Table(columns=[
        'dataset', 'split', 'n_samples',
        'avg_gloss_len', 'max_gloss_len',
        'avg_sentence_len', 'max_sentence_len',
        'vocab_size_gloss', 'vocab_size_sentence',
        'avg_frames', 'median_frames', 'max_frames',
        'zero_frame_files', 'missing_npy',
        'missing_pseudogloss',
    ])

    charts = {}

    for split in splits:
        csv_path = root_dir / 'annotations' / f'how2sign_{split}.csv'
        if not csv_path.exists():
            print(f"  [{split}] CSV not found: {csv_path}, skipping.")
            continue

        df = pd.read_csv(csv_path, sep='\t')
        kps_dir   = root_dir / 'keypoints' / split
        video_dir = root_dir / split

        print(f"  [{split}] {len(df)} samples")

        has_pseudogloss = 'PSEUDOGLOSS' in df.columns

        # ── Samples ──
        indices = random.sample(range(len(df)), min(N_SAMPLES, len(df)))
        for i in indices:
            row  = df.iloc[i]
            name = str(row['SENTENCE_NAME'])
            sent_text  = str(row['SENTENCE']) if pd.notna(row.get('SENTENCE')) else ''
            gloss_text = str(row['PSEUDOGLOSS']) \
                if has_pseudogloss and pd.notna(row.get('PSEUDOGLOSS')) else ''

            npy_path = kps_dir / f"{name}.npy"
            kps_img  = None
            n_frames = 0
            if npy_path.exists():
                kps = np.load(npy_path)
                kps_img  = kps_to_wandb_image(kps)
                n_frames = int((np.linalg.norm(kps, axis=-1) > 0).sum())

            video_path = video_dir / f"{name}.mp4"
            rgb_img = None
            if video_path.exists():
                rgb_img = video_frame_to_wandb_image(str(video_path))

            sample_table.add_data(
                'how2sign', split, name,
                rgb_img, kps_img,
                sent_text, gloss_text,
                len(gloss_text.split()) if gloss_text else 0,
                len(sent_text.split()),
                n_frames,
            )

        # ── Statistics ──
        gloss_col_data = df['PSEUDOGLOSS'].tolist() if has_pseudogloss else []
        sent_col_data  = df['SENTENCE'].tolist()

        gloss_lens  = gloss_lengths(gloss_col_data)
        sent_lens   = gloss_lengths(sent_col_data)
        gloss_vocab = build_vocab(gloss_col_data)
        sent_vocab  = build_vocab(sent_col_data)

        missing_pg = int(df['PSEUDOGLOSS'].isna().sum()) if has_pseudogloss else len(df)

        frame_counts, zero_files, missing_npy = [], 0, 0
        for _, row in df.iterrows():
            p = kps_dir / f"{row['SENTENCE_NAME']}.npy"
            if p.exists():
                kps = np.load(p)
                fc  = int((np.linalg.norm(kps, axis=-1) > 0).sum())
                frame_counts.append(fc)
                if fc == 0:
                    zero_files += 1
            else:
                missing_npy += 1

        if frame_counts:
            charts[f'how2sign/{split}/frame_dist'] = hist_image(
                frame_counts, f'How2Sign {split} — frame counts', 'frames', 'coral')
        if gloss_lens:
            charts[f'how2sign/{split}/gloss_len_dist'] = hist_image(
                gloss_lens, f'How2Sign {split} — pseudogloss lengths', 'tokens', 'mediumorchid')
        if sent_lens:
            charts[f'how2sign/{split}/sent_len_dist'] = hist_image(
                sent_lens, f'How2Sign {split} — sentence lengths', 'tokens', 'goldenrod')

        stat_table.add_data(
            'how2sign', split, len(df),
            round(np.mean(gloss_lens), 2) if gloss_lens else 0,
            max(gloss_lens) if gloss_lens else 0,
            round(np.mean(sent_lens), 2) if sent_lens else 0,
            max(sent_lens) if sent_lens else 0,
            len(gloss_vocab), len(sent_vocab),
            round(np.mean(frame_counts), 1) if frame_counts else 0,
            int(np.median(frame_counts)) if frame_counts else 0,
            max(frame_counts) if frame_counts else 0,
            zero_files, missing_npy, missing_pg,
        )
        avg_frames_str = f"{np.mean(frame_counts):.1f}" if frame_counts else "N/A"
        print(f"  [{split}] sent_vocab={len(sent_vocab)}, "
              f"avg_sent_len={np.mean(sent_lens):.1f}, "
              f"avg_frames={avg_frames_str}, "
              f"missing_npy={missing_npy}")

    return sample_table, stat_table, charts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Analyze SLT datasets and log to W&B')
    parser.add_argument('--phoenix_dir',  type=str, default=None,
                        help='Root of PHOENIX-2014-T (contains annotations/ and features/)')
    parser.add_argument('--how2sign_dir', type=str, default=None,
                        help='Root of how2sign_hf (contains annotations/ keypoints/ train/ val/ test/)')
    parser.add_argument('--n_samples',   type=int, default=20,
                        help='Visual samples per split (default: 20)')
    args = parser.parse_args()

    global N_SAMPLES
    N_SAMPLES = args.n_samples

    if args.phoenix_dir is None and args.how2sign_dir is None:
        parser.error('Provide at least one of --phoenix_dir or --how2sign_dir')

    wandb.init(
        project='slt',
        name='dataset-analysis',
        config={
            'phoenix_dir':  args.phoenix_dir,
            'how2sign_dir': args.how2sign_dir,
            'n_samples':    N_SAMPLES,
        },
    )

    log_payload = {}

    if args.phoenix_dir:
        s_table, stat_table, charts = analyze_phoenix(Path(args.phoenix_dir))
        log_payload['phoenix/samples']    = s_table
        log_payload['phoenix/statistics'] = stat_table
        log_payload.update(charts)

    if args.how2sign_dir:
        s_table, stat_table, charts = analyze_how2sign(Path(args.how2sign_dir))
        log_payload['how2sign/samples']    = s_table
        log_payload['how2sign/statistics'] = stat_table
        log_payload.update(charts)

    wandb.log(log_payload)
    wandb.finish()
    print('\nDone. Results saved to W&B run "dataset-analysis" in project "slt".')


if __name__ == '__main__':
    main()
