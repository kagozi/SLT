"""
visualize_phoenix.py — Manuscript figure: PHOENIX-2014-T raw frames vs keypoint skeletons.

Produces a publication-quality 2-row figure:
  Row 1: raw video frames (cropped to signer region)
  Row 2: 75-landmark skeleton drawn on a clean background

Saves PNG + PDF to --output_dir.

Usage (cluster):
    python visualize_phoenix.py \
        --root_dir /data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --output_dir /data/experiments/figures \
        --split dev --n_frames 5 --n_sequences 2

Target a specific cached keypoint clip:
    python visualize_phoenix.py \
        --root_dir /data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
        --sequence_npy 01April_2010_Thursday_heute-6697.npy \
        --output_dir /data/experiments/figures
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from PIL import Image


# ─── Landmark layout in the 75-joint (225-d) PHOENIX keypoint array ──
#
# Extraction order (see preprocessing.py / PhoenixKeypointExtractor):
#   Pose:        joints  0–12   (13 × 3 = 39 dims)
#   Left hand:   joints 13–33   (21 × 3 = 63 dims)
#   Right hand:  joints 34–54   (21 × 3 = 63 dims)
#   Face:        joints 55–74   (20 × 3 = 60 dims)

P, LH, RH, F = 13, 21, 21, 20

POSE_R  = slice(0,          P)
LHAND_R = slice(P,          P + LH)
RHAND_R = slice(P + LH,     P + LH + RH)
FACE_R  = slice(P + LH + RH, P + LH + RH + F)


# ─── Skeleton connectivity ────────────────────────────────────────────

# Upper-body pose:
#  idx 0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear
#  5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow
#  9=l_wrist, 10=r_wrist, 11=l_hip, 12=r_hip
POSE_CONNS = [
    (0, 1), (0, 2), (1, 3), (2, 4),   # face centre → eyes → ears
    (5, 6),                             # shoulder span
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 11), (6, 12), (11, 12),         # torso
]


def _hand_conns(offset: int):
    """MediaPipe 21-point hand skeleton connections at array offset."""
    c = []
    for base in [1, 5, 9, 13, 17]:          # wrist → each finger root
        c.append((offset, offset + base))
    for start in [1, 5, 9, 13, 17]:         # finger chains
        for i in range(3):
            c.append((offset + start + i, offset + start + i + 1))
    for a, b in [(5, 9), (9, 13), (13, 17)]:  # palm arcs
        c.append((offset + a, offset + b))
    return c


LHAND_CONNS = _hand_conns(P)
RHAND_CONNS = _hand_conns(P + LH)

FO = P + LH + RH   # face offset = 55
# Face: lips (0-7), l-brow (8-10), r-brow (11-13), l-eye (14-15), r-eye (16-17), nose (18-19)
FACE_CONNS = [
    (FO+0, FO+4), (FO+4, FO+2), (FO+2, FO+5), (FO+5, FO+1),
    (FO+0, FO+6), (FO+6, FO+3), (FO+3, FO+7), (FO+7, FO+1),
    (FO+8, FO+9), (FO+9, FO+10),
    (FO+11, FO+12), (FO+12, FO+13),
    (FO+14, FO+15),
    (FO+16, FO+17),
    (FO+18, FO+19),
]


# ─── Colour palette (matches sample.ipynb) ───────────────────────────

JOINT_COLORS = {
    'pose':       '#2196F3',  # blue
    'left_hand':  '#FF9800',  # orange
    'right_hand': '#4CAF50',  # green
    'face':       '#E91E63',  # pink
}
CONN_COLORS = {
    'pose':       '#64B5F6',
    'left_hand':  '#FFB74D',
    'right_hand': '#81C784',
    'face':       '#F48FB1',
}
JOINT_SIZES = {'pose': 35, 'left_hand': 18, 'right_hand': 18, 'face': 14}


# ─── Hand repositioning ───────────────────────────────────────────────

def _reattach_hands(xy: np.ndarray) -> np.ndarray:
    """
    Translate both hand groups so the hand wrist (joint 0 of each hand)
    sits exactly on the corresponding pose wrist joint.

      Left  hand wrist = idx P     (13) → pose left  wrist = idx 9
      Right hand wrist = idx P+LH  (34) → pose right wrist = idx 10
    """
    xy = xy.copy()

    def _translate(hand_start, hand_len, pose_wrist_idx):
        hw = xy[hand_start]
        pw = xy[pose_wrist_idx]
        if np.allclose(hw, 0) or np.allclose(pw, 0):
            return
        offset = pw - hw
        for k in range(hand_start, hand_start + hand_len):
            if not np.allclose(xy[k], 0):
                xy[k] += offset

    _translate(P,      LH, 9)
    _translate(P + LH, RH, 10)
    return xy


# ─── Core skeleton plot ───────────────────────────────────────────────

def plot_skeleton(joints_75x3: np.ndarray, ax, title: str = '',
                  dark_bg: bool = True):
    """Draw 75-joint skeleton on ax.  joints_75x3 shape: (75, 3)."""
    if dark_bg:
        ax.set_facecolor('#1a1a2e')

    xy = _reattach_hands(joints_75x3[:, :2])   # x, y with hands repositioned

    def draw_conns(conns, color, lw=1.6, alpha=0.7):
        segs = []
        for i, j in conns:
            if np.allclose(xy[i], 0) or np.allclose(xy[j], 0):
                continue
            segs.append([xy[i], xy[j]])
        if segs:
            ax.add_collection(
                LineCollection(segs, colors=color, linewidths=lw, alpha=alpha,
                               zorder=2))

    draw_conns(POSE_CONNS,  CONN_COLORS['pose'],       lw=2.2)
    draw_conns(LHAND_CONNS, CONN_COLORS['left_hand'],  lw=1.4)
    draw_conns(RHAND_CONNS, CONN_COLORS['right_hand'], lw=1.4)
    draw_conns(FACE_CONNS,  CONN_COLORS['face'],       lw=1.0)

    groups = {
        'pose':       (xy[POSE_R],  JOINT_COLORS['pose'],       JOINT_SIZES['pose']),
        'left_hand':  (xy[LHAND_R], JOINT_COLORS['left_hand'],  JOINT_SIZES['left_hand']),
        'right_hand': (xy[RHAND_R], JOINT_COLORS['right_hand'], JOINT_SIZES['right_hand']),
        'face':       (xy[FACE_R],  JOINT_COLORS['face'],       JOINT_SIZES['face']),
    }
    for name, (pts, col, sz) in groups.items():
        mask = ~np.all(pts == 0, axis=1)
        if mask.any():
            ax.scatter(pts[mask, 0], pts[mask, 1], s=sz, c=col, zorder=3,
                       alpha=0.95, edgecolors='white', linewidths=0.3)

    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, color='white' if dark_bg else 'black',
                 pad=4)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ─── Frame loading ────────────────────────────────────────────────────

def load_frames(frame_dir: Path, frame_indices):
    """Load specific frame images from a PHOENIX sequence directory."""
    # PHOENIX stores frames as images0001.png … imagesNNNN.png
    all_pngs = sorted(frame_dir.glob('images*.png'))
    if not all_pngs:
        all_pngs = sorted(frame_dir.glob('*.png'))
    if not all_pngs:
        all_pngs = sorted(frame_dir.glob('*.jpg'))

    imgs = []
    for idx in frame_indices:
        if idx < len(all_pngs):
            imgs.append(np.array(Image.open(all_pngs[idx]).convert('RGB')))
        else:
            imgs.append(None)
    return imgs, all_pngs


# ─── Pick frames with good hand visibility ────────────────────────────

def pick_active_frames(kps: np.ndarray, n: int) -> list:
    """
    Choose n frames with maximum hand-joint activity (non-zero norm).
    Spreads selected frames evenly across the active portion of the sequence.
    """
    T = kps.shape[0]
    # Hand columns: joints 13-54 → dims 39..165
    hand_kps = kps[:, 39:165]
    activity = np.linalg.norm(hand_kps.reshape(T, -1, 3), axis=-1).mean(axis=-1)
    active = np.where(activity > 0.01)[0]
    if len(active) < n:
        active = np.arange(T)
    # Evenly spaced within the active span
    indices = np.linspace(active[0], active[-1], n, dtype=int)
    return indices.tolist()


# ─── Main figure function ─────────────────────────────────────────────

def make_figure(sequences: list, root_dir: Path, split: str,
                n_frames: int, output_dir: Path, dpi: int = 200):
    """
    sequences: list of sequence name strings from the CSV.
    Produces one figure per sequence and a combined figure.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    all_figs = []

    for seq_name in sequences:
        kps_path  = root_dir / 'features' / 'keypoints' / split / f'{seq_name}.npy'
        frame_dir = root_dir / 'features' / 'fullFrame-210x260px' / split / seq_name

        if not kps_path.exists():
            print(f'  [SKIP] keypoints not found: {kps_path}')
            continue
        if not frame_dir.exists():
            print(f'  [SKIP] frame dir not found: {frame_dir}')
            continue

        kps = np.load(kps_path)           # (T, 225)
        if kps.ndim == 2 and kps.shape[1] == 225:
            kps_3d = kps.reshape(-1, 75, 3)
        else:
            print(f'  [SKIP] unexpected kps shape: {kps.shape}')
            continue

        frame_indices = pick_active_frames(kps, n_frames)
        raw_imgs, all_pngs = load_frames(frame_dir, frame_indices)

        # ── figure layout ──────────────────────────────────────────
        fig = plt.figure(figsize=(n_frames * 2.8, 5.8),
                         facecolor='#0f0f1a')
        fig.patch.set_facecolor('#0f0f1a')

        gs = fig.add_gridspec(2, n_frames,
                              hspace=0.06, wspace=0.04,
                              top=0.88, bottom=0.02,
                              left=0.01, right=0.99)

        # Row labels
        fig.text(0.005, 0.72, 'Raw\nframe', va='center', ha='left',
                 fontsize=9, color='#aaaaaa', rotation=90)
        fig.text(0.005, 0.28, 'Keypoint\nskeleton', va='center', ha='left',
                 fontsize=9, color='#aaaaaa', rotation=90)

        for col, (fidx, raw_img) in enumerate(zip(frame_indices, raw_imgs)):
            # ── top row: raw frame ────────────────────────────────
            ax_raw = fig.add_subplot(gs[0, col])
            ax_raw.set_facecolor('#0f0f1a')
            if raw_img is not None:
                ax_raw.imshow(raw_img)
            else:
                ax_raw.text(0.5, 0.5, 'frame\nnot found',
                            ha='center', va='center', color='#666666',
                            transform=ax_raw.transAxes, fontsize=8)
            ax_raw.set_xticks([]); ax_raw.set_yticks([])
            ax_raw.set_title(f'frame {fidx}', fontsize=8, color='#888888', pad=3)
            for spine in ax_raw.spines.values():
                spine.set_edgecolor('#333344'); spine.set_linewidth(0.5)

            # ── bottom row: skeleton ──────────────────────────────
            ax_sk = fig.add_subplot(gs[1, col])
            plot_skeleton(kps_3d[fidx], ax_sk, dark_bg=True)

        # Legend (bottom-right)
        patches = [
            mpatches.Patch(color=JOINT_COLORS['pose'],       label='Pose'),
            mpatches.Patch(color=JOINT_COLORS['left_hand'],  label='Left hand'),
            mpatches.Patch(color=JOINT_COLORS['right_hand'], label='Right hand'),
            mpatches.Patch(color=JOINT_COLORS['face'],       label='Face'),
        ]
        fig.legend(handles=patches, loc='lower right', ncol=4,
                   fontsize=8, framealpha=0.15,
                   facecolor='#1a1a2e', edgecolor='#333344',
                   labelcolor='white', bbox_to_anchor=(0.99, 0.0))

        # Title
        fig.suptitle(
            f'PHOENIX-2014-T  ·  {seq_name}  '
            f'[75 landmarks · 225-d features]',
            fontsize=10, color='#ccccdd', y=0.97,
        )

        out_stem = output_dir / f'phoenix_vis_{seq_name}'
        fig.savefig(f'{out_stem}.png', dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        fig.savefig(f'{out_stem}.pdf', bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f'  Saved: {out_stem}.png / .pdf')
        all_figs.append(fig)

    return all_figs


# ─── Sequence selection ───────────────────────────────────────────────

def pick_sequences(root_dir: Path, split: str, n: int) -> list:
    """
    Pick n sequences that have both keypoints and raw frames available.
    Prefer mid-length sequences (not too short, not too long).
    """
    csv = root_dir / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
    df = pd.read_csv(csv, sep='|')

    kps_dir   = root_dir / 'features' / 'keypoints'   / split
    frame_dir = root_dir / 'features' / 'fullFrame-210x260px' / split

    candidates = []
    for _, row in df.iterrows():
        name = row['name']
        if (kps_dir / f'{name}.npy').exists() and (frame_dir / name).exists():
            kps = np.load(kps_dir / f'{name}.npy')
            T = int((np.linalg.norm(kps, axis=-1) > 0).sum())
            # Prefer sequences of 40–120 active frames (long enough to sample 5)
            if 40 <= T <= 120:
                candidates.append((name, T))

    if not candidates:
        # Relax length constraint
        candidates = [(row['name'], 0) for _, row in df.iterrows()
                      if (kps_dir / f'{row["name"]}.npy').exists()
                      and (frame_dir / row['name']).exists()]

    if not candidates:
        sys.exit('No sequences found with both keypoints and raw frames.')

    # Sort by length and pick evenly from the distribution
    candidates.sort(key=lambda x: x[1])
    step = max(1, len(candidates) // n)
    picked = [candidates[i * step][0] for i in range(min(n, len(candidates)))]
    print(f'  Selected {len(picked)} sequences: {picked}')
    return picked


def resolve_sequence_and_split(root_dir: Path, split: Optional[str],
                               sequence_npy: Optional[str],
                               sequences: Optional[list]):
    """
    Resolve a specific PHOENIX sequence and split.

    Priority:
      1. --sequence_npy : auto-detect split from cached keypoints / frame dir
      2. --sequences    : use provided sequence names with explicit split
      3. fallback       : caller will use pick_sequences()
    """
    if sequence_npy:
        seq_name = Path(sequence_npy).stem
        candidate_splits = [split] if split else ['train', 'dev', 'test']
        candidate_splits = [s for s in candidate_splits if s is not None]
        if not candidate_splits:
            candidate_splits = ['train', 'dev', 'test']

        for cand in candidate_splits:
            kps_path = root_dir / 'features' / 'keypoints' / cand / f'{seq_name}.npy'
            frame_dir = root_dir / 'features' / 'fullFrame-210x260px' / cand / seq_name
            if kps_path.exists() and frame_dir.exists():
                print(f'  Resolved {seq_name} in split={cand}')
                return cand, [seq_name]

        sys.exit(f'Could not resolve {sequence_npy} in splits {candidate_splits}')

    if sequences:
        if split is None:
            sys.exit('--split is required when using --sequences')
        return split, sequences

    if split is None:
        split = 'dev'
    return split, []


# ─── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='PHOENIX-2014-T manuscript figure: raw frames vs keypoints')
    parser.add_argument('--root_dir', default=
        '/data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T')
    parser.add_argument('--split', default=None,
                        choices=['train', 'dev', 'test'])
    parser.add_argument('--output_dir', default='/data/experiments/figures')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='Frames to show per sequence')
    parser.add_argument('--n_sequences', type=int, default=3,
                        help='Number of sequences to visualise')
    parser.add_argument('--sequences', nargs='+', default=None,
                        help='Specific sequence names (overrides --n_sequences)')
    parser.add_argument('--sequence_npy', default=None,
                        help='Specific cached .npy clip name, e.g. 01April_2010_Thursday_heute-6697.npy')
    parser.add_argument('--dpi', type=int, default=200)
    args = parser.parse_args()

    root = Path(args.root_dir)
    out  = Path(args.output_dir)

    print(f'Root:   {root}')
    print(f'Output: {out}')

    resolved_split, seqs = resolve_sequence_and_split(
        root, args.split, args.sequence_npy, args.sequences
    )
    print(f'Split:  {resolved_split}')
    if not seqs:
        seqs = pick_sequences(root, resolved_split, args.n_sequences)

    make_figure(seqs, root, resolved_split,
                n_frames=args.n_frames,
                output_dir=out,
                dpi=args.dpi)

    print('Done.')


if __name__ == '__main__':
    main()
