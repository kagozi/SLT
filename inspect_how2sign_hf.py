"""
Inspect /data/how2sign_hf and log dataset statistics + sample keypoint
visualizations to Weights & Biases.

Run:
    python inspect_how2sign_hf.py --root_dir /data/how2sign_hf
"""

import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


SPLITS = ('train', 'val', 'test')


def check_split(root: Path, split: str) -> dict:
    ann_path = root / 'annotations' / f'how2sign_{split}.csv'
    kps_dir  = root / 'keypoints' / split

    stats = {'split': split}

    if not ann_path.exists():
        print(f'  [{split}] ERROR: annotation CSV missing')
        stats['error'] = 'missing CSV'
        return stats

    df = pd.read_csv(ann_path, sep='\t')
    stats['n_rows'] = len(df)
    stats['has_sentence']    = 'SENTENCE' in df.columns
    stats['has_pseudogloss'] = 'PSEUDOGLOSS' in df.columns

    npy_files = list(kps_dir.glob('*.npy'))
    stats['n_npy_files'] = len(npy_files)
    stats['missing_npy']  = stats['n_rows'] - len(npy_files)

    print(f'  [{split}] CSV rows={stats["n_rows"]}, '
          f'npy files={stats["n_npy_files"]}, '
          f'missing={stats["missing_npy"]}')

    if not npy_files:
        stats['error'] = 'no npy files'
        return stats

    sample_paths = random.sample(npy_files, min(200, len(npy_files)))
    shapes, zero_count, nan_count, frame_lens = [], 0, 0, []

    for p in sample_paths:
        arr = np.load(p)
        shapes.append(arr.shape)
        if np.all(arr == 0):
            zero_count += 1
        if np.any(np.isnan(arr)):
            nan_count += 1
        frame_lens.append(int((np.abs(arr).sum(axis=1) > 0).sum()))

    stats['shape_sample']  = str(list(set(shapes))[:3])
    stats['all_zero_pct']  = round(zero_count / len(sample_paths) * 100, 2)
    stats['nan_pct']       = round(nan_count  / len(sample_paths) * 100, 2)
    stats['mean_frames']   = round(float(np.mean(frame_lens)), 1)
    stats['median_frames'] = round(float(np.median(frame_lens)), 1)
    stats['min_frames']    = int(np.min(frame_lens))
    stats['max_frames']    = int(np.max(frame_lens))
    stats['frame_lens']    = frame_lens
    stats['npy_files']     = npy_files

    print(f'  [{split}] shapes={set(shapes)}, '
          f'zero={stats["all_zero_pct"]}%, nan={stats["nan_pct"]}%')
    print(f'  [{split}] frames: mean={stats["mean_frames"]}, '
          f'median={stats["median_frames"]}, '
          f'min={stats["min_frames"]}, max={stats["max_frames"]}')
    return stats


def make_keypoint_fig(arr: np.ndarray, title: str) -> plt.Figure:
    """(T, 225) heatmap + per-frame L2 norm."""
    real_len = int((np.abs(arr).sum(axis=1) > 0).sum())
    disp = arr[:real_len] if real_len > 0 else arr[:1]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    im = axes[0].imshow(disp.T, aspect='auto', cmap='RdBu_r',
                        vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=axes[0], fraction=0.02)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Feature dim (225)')
    axes[0].set_title(f'{title}  [{real_len} frames]')
    for label, y in [('Pose', 0), ('L.Hand', 39), ('R.Hand', 102), ('Face', 165)]:
        axes[0].axhline(y, color='yellow', linewidth=0.8, linestyle='--')
        axes[0].text(disp.shape[0] * 0.01, y + 2, label,
                     fontsize=7, color='yellow', va='bottom')

    axes[1].plot(np.linalg.norm(disp, axis=1), linewidth=0.8)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('L2 norm')
    axes[1].set_title('Per-frame keypoint magnitude')

    fig.suptitle(title[:80], fontsize=9)
    fig.tight_layout()
    return fig


def make_hist_fig(frame_lens: list, split: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(frame_lens, bins=40, edgecolor='black', color='steelblue', alpha=0.8)
    ax.axvline(np.mean(frame_lens), color='red', linestyle='--',
               label=f'mean={np.mean(frame_lens):.0f}')
    ax.axvline(np.median(frame_lens), color='orange', linestyle='--',
               label=f'median={np.median(frame_lens):.0f}')
    ax.set_xlabel('Real frames (non-padded)')
    ax.set_ylabel('Count')
    ax.set_title(f'{split} — clip length distribution (n={len(frame_lens)} sampled)')
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',   type=str, default='/data/how2sign_hf')
    parser.add_argument('--wandb_proj', type=str, default='slt-how2sign')
    parser.add_argument('--n_samples',  type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root_dir)
    print(f'Inspecting {root}\n')

    run = wandb.init(project=args.wandb_proj, name='dataset-inspection-hf-v2',
                     tags=['inspection', 'how2sign-hf'])

    all_stats = {}
    log_dict  = {}          # collected into one wandb.log call at the end

    for split in SPLITS:
        print(f'\n--- {split} ---')
        stats = check_split(root, split)
        all_stats[split] = stats

        if 'error' in stats:
            continue

        # Scalar stats
        log_dict.update({
            f'{split}/n_rows':       stats['n_rows'],
            f'{split}/n_npy_files':  stats['n_npy_files'],
            f'{split}/missing_npy':  stats['missing_npy'],
            f'{split}/zero_pct':     stats['all_zero_pct'],
            f'{split}/nan_pct':      stats['nan_pct'],
            f'{split}/mean_frames':  stats['mean_frames'],
            f'{split}/median_frames':stats['median_frames'],
            f'{split}/min_frames':   stats['min_frames'],
            f'{split}/max_frames':   stats['max_frames'],
        })

        # Frame-length histogram
        hist_fig = make_hist_fig(stats['frame_lens'], split)
        log_dict[f'{split}/frame_length_hist'] = wandb.Image(hist_fig,
                                                              caption=f'{split} clip lengths')
        plt.close(hist_fig)

        # Keypoint samples — logged as a WandB Table so all are visible at once
        sample_paths = random.sample(stats['npy_files'],
                                     min(args.n_samples, len(stats['npy_files'])))
        table = wandb.Table(columns=['clip_name', 'real_frames', 'keypoint_heatmap'])
        for p in sample_paths:
            arr = np.load(p)
            real_len = int((np.abs(arr).sum(axis=1) > 0).sum())
            fig = make_keypoint_fig(arr, p.stem)
            table.add_data(p.stem, real_len, wandb.Image(fig, caption=p.stem))
            plt.close(fig)
        log_dict[f'{split}/keypoint_samples'] = table

    # Dataset summary table
    rows = []
    for split, s in all_stats.items():
        if 'error' not in s:
            rows.append([split, s['n_rows'], s['n_npy_files'],
                         s['missing_npy'], s['all_zero_pct'],
                         s['mean_frames'], s['median_frames'],
                         s['shape_sample']])

    log_dict['dataset_summary'] = wandb.Table(
        columns=['split', 'csv_rows', 'npy_files', 'missing',
                 'zero_pct', 'mean_frames', 'median_frames', 'shapes'],
        data=rows
    )

    # Single log call — all media visible together
    run.log(log_dict)

    print('\n=== Summary ===')
    for split, s in all_stats.items():
        if 'error' not in s:
            print(f'  {split:5s}: {s["n_npy_files"]:6d} files, '
                  f'zero={s["all_zero_pct"]}%, '
                  f'mean_frames={s["mean_frames"]}')

    run.finish()
    print('\n✅ Inspection complete. Results logged to WandB.')


if __name__ == '__main__':
    main()
