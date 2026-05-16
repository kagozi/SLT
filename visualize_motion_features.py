"""
visualize_motion_features.py — Manuscript figure: motion feature augmentation.

Shows how raw right-hand keypoints are augmented with velocity and acceleration
to triple the per-stream feature dimension (63 → 189 for right hand).

Layout (4 rows × N_FRAMES columns + bottom schematic):
  Row 0  – Position  x_t       : skeleton at each frame
  Row 1  – Velocity  Δx_t      : ghost skeleton + arrows colored by magnitude
  Row 2  – Acceleration Δ²x_t  : ghost skeleton + arrows colored by magnitude
  Row 3  – Concatenation panel  : [pos | vel | acc] = 63+63+63 = 189-d schematic

Reads a local How2Sign (T, 543, 3) raw-holistic .npy file.
Saves PNG + PDF to figures/.

Usage:
    python visualize_motion_features.py
    python visualize_motion_features.py --npy data/test/frontal/dIIMHOX5AD8_6-8-rgb_front_holistic.npy
"""

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np


# ─── Raw-holistic landmark layout (How2Sign / YouTube-ASL) ───────────────────
#   [0:33]    pose
#   [33:501]  face mesh
#   [501:522] left hand
#   [522:543] right hand  ← we use this

RHAND_START = 522
RHAND_END   = 543   # exclusive

# MediaPipe 21-point hand connections
_HAND_CONNS = []
for _base in [1, 5, 9, 13, 17]:       # wrist → finger roots
    _HAND_CONNS.append((0, _base))
for _start in [1, 5, 9, 13, 17]:      # finger chains
    for _i in range(3):
        _HAND_CONNS.append((_start + _i, _start + _i + 1))
for _a, _b in [(5, 9), (9, 13), (13, 17)]:   # palm arcs
    _HAND_CONNS.append((_a, _b))


# ─── Colour palette ───────────────────────────────────────────────────────────
BG          = '#0f0f1a'
JOINT_COL   = '#4CAF50'   # green — right hand
CONN_COL    = '#81C784'
GHOST_JOINT = '#2a4a2e'
GHOST_CONN  = '#1e3320'
CMAP_VEL    = plt.cm.plasma      # velocity  magnitude
CMAP_ACC    = plt.cm.cool        # accel magnitude


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hand_conns_offset(offset_xy):
    """Return (N,2,2) array of line segments for hand connections."""
    segs = []
    for i, j in _HAND_CONNS:
        if np.allclose(offset_xy[i], 0) or np.allclose(offset_xy[j], 0):
            continue
        segs.append([offset_xy[i], offset_xy[j]])
    return segs


def _draw_hand(ax, xy21, color_joint, color_conn, lw=1.4, s=22, alpha=1.0,
               dark_bg=True):
    """Draw a 21-joint hand skeleton on ax.  xy21: (21,2)."""
    if dark_bg:
        ax.set_facecolor(BG)
    segs = _hand_conns_offset(xy21)
    if segs:
        ax.add_collection(LineCollection(segs, colors=color_conn,
                                         linewidths=lw, alpha=alpha * 0.8,
                                         zorder=2))
    mask = ~np.all(xy21 == 0, axis=1)
    if mask.any():
        ax.scatter(xy21[mask, 0], xy21[mask, 1], s=s,
                   c=color_joint, edgecolors='white', linewidths=0.3,
                   alpha=alpha, zorder=4)


def _set_hand_limits(ax, xy21, pad_frac=0.30):
    """Auto-zoom to the hand bounding box."""
    valid = xy21[~np.all(xy21 == 0, axis=1)]
    if len(valid) == 0:
        return
    x0, x1 = valid[:, 0].min(), valid[:, 0].max()
    y0, y1 = valid[:, 1].min(), valid[:, 1].max()
    px = max((x1 - x0) * pad_frac, 0.02)
    py = max((y1 - y0) * pad_frac, 0.02)
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(y1 + py, y0 - py)   # y inverted (image coords)


def _clean_ax(ax, title='', title_color='#aaaaaa'):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')
    if title:
        ax.set_title(title, fontsize=8, color=title_color, pad=3)


def _draw_motion_arrows(ax, xy_base, delta, cmap, vmin, vmax,
                        scale=None, ghost=True):
    """
    Overlay motion arrows on ax.

    xy_base : (21,2)  — anchor positions
    delta   : (21,2)  — displacement vectors (vel or accel)
    cmap    : colormap keyed by |delta| magnitude
    vmin/max: color normalization range
    scale   : arrow scale factor (auto if None)
    ghost   : draw a ghost skeleton behind arrows
    """
    ax.set_facecolor(BG)
    if ghost:
        segs = _hand_conns_offset(xy_base)
        if segs:
            ax.add_collection(LineCollection(segs, colors=GHOST_CONN,
                                             linewidths=1.2, alpha=0.45, zorder=2))
        mask = ~np.all(xy_base == 0, axis=1)
        if mask.any():
            ax.scatter(xy_base[mask, 0], xy_base[mask, 1], s=14,
                       c=GHOST_JOINT, edgecolors='none', alpha=0.55, zorder=3)

    norm = Normalize(vmin=vmin, vmax=vmax)
    mag  = np.linalg.norm(delta, axis=-1)   # (21,)

    # Auto-scale: longest arrow ≈ 15 % of bounding-box span
    if scale is None:
        valid = xy_base[~np.all(xy_base == 0, axis=1)]
        if len(valid) > 1:
            span = max(valid[:, 0].ptp(), valid[:, 1].ptp())
            max_mag = mag.max() if mag.max() > 0 else 1.0
            scale = (span * 0.18) / max_mag
        else:
            scale = 1.0

    for i, (pt, dv, mg) in enumerate(zip(xy_base, delta, mag)):
        if np.allclose(pt, 0) or mg < 1e-6:
            continue
        col = cmap(norm(mg))
        ax.annotate('', xy=pt + dv * scale, xytext=pt,
                    arrowprops=dict(arrowstyle='->', color=col,
                                   lw=1.8, mutation_scale=8),
                    zorder=5)
        ax.scatter(*pt, s=16, c=[col], edgecolors='white',
                   linewidths=0.2, zorder=6)

    return norm, scale


def _find_active_window(rhand, n_frames=5, min_motion=0.005):
    """
    Return start index of an n_frames-long window with good hand visibility
    and meaningful inter-frame motion.
    """
    T = rhand.shape[0]
    if T < n_frames + 2:
        return 0

    # visibility: fraction of non-zero joints per frame
    vis = (~np.all(rhand == 0, axis=-1)).mean(axis=-1)   # (T,)
    # motion: mean velocity magnitude per frame
    vel  = np.diff(rhand[:, :, :2], axis=0)              # (T-1,21,2)
    motion = np.linalg.norm(vel, axis=-1).mean(axis=-1)   # (T-1,)
    motion = np.append(motion, motion[-1])                # align to T

    # score = visibility × motion
    score = vis * motion
    # Slide a window of n_frames+2 (need extra for accel)
    best_start, best_score = 0, -1.0
    for t in range(T - n_frames - 2):
        w = score[t: t + n_frames + 2].mean()
        if w > best_score and vis[t: t + n_frames + 2].min() > 0.5:
            best_score, best_start = w, t
    return best_start


# ─── Main figure ─────────────────────────────────────────────────────────────

def make_motion_figure(npy_path: str, output_dir: Path,
                       n_frames: int = 5, dpi: int = 220):
    """
    Build a 4-row motion-feature figure and save to output_dir.
    Rows: Position | Velocity | Acceleration | Feature-concatenation schematic
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load + extract right-hand ──────────────────────────────────────
    raw = np.load(npy_path)                                # (T, 543, 3)
    if raw.ndim == 2:
        # Already (T, 225) processed format — reshape and fall back to right-hand slice
        T = raw.shape[0]
        raw_3d = raw.reshape(T, 75, 3)
        # right hand: joints 34-54 in 75-joint array
        rhand_all = raw_3d[:, 34:55, :2]                   # (T,21,2)
    else:
        rhand_all = raw[:, RHAND_START:RHAND_END, :2]      # (T,21,2)

    T = rhand_all.shape[0]
    start = _find_active_window(rhand_all, n_frames=n_frames)

    # Grab n_frames + 2 consecutive frames (extras needed for accel at last vis frame)
    end = min(start + n_frames + 2, T)
    frames = rhand_all[start:end]   # (≤n+2, 21, 2)

    # We'll display columns 0 … n_frames-1
    disp_frames = min(n_frames, len(frames) - 2)

    # ── Pre-compute velocity + acceleration ───────────────────────────
    # vel[t]  = x[t+1] - x[t]
    # acc[t]  = vel[t+1] - vel[t] = x[t+2] - 2*x[t+1] + x[t]
    vel = np.diff(frames, axis=0)                          # (n+1, 21, 2)
    acc = np.diff(vel,    axis=0)                          # (n,   21, 2)

    vel_mag = np.linalg.norm(vel, axis=-1)                 # (n+1, 21)
    acc_mag = np.linalg.norm(acc, axis=-1)                 # (n,   21)

    vel_vmax = np.percentile(vel_mag, 95) if vel_mag.max() > 0 else 1.0
    acc_vmax = np.percentile(acc_mag, 95) if acc_mag.max() > 0 else 1.0
    vel_vmin = 0.0
    acc_vmin = 0.0

    # ── Figure setup ──────────────────────────────────────────────────
    # 3 data rows + 1 schematic row
    n_rows  = 4
    heights = [1.0, 1.0, 1.0, 0.55]
    fig = plt.figure(figsize=(disp_frames * 2.6 + 0.6, 9.8),
                     facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = fig.add_gridspec(n_rows, disp_frames,
                          height_ratios=heights,
                          hspace=0.08, wspace=0.04,
                          top=0.93, bottom=0.03,
                          left=0.09, right=0.98)

    # ── Row labels ────────────────────────────────────────────────────
    total_h  = sum(heights[:-1])   # label only data rows
    cum, row_mids = 0, []
    for h in heights:
        row_mids.append(1.0 - (cum + h / 2) / sum(heights))
        cum += h

    label_x = 0.01
    row_labels = [
        ('Position\n$\\mathbf{x}_t$',           '#4CAF50'),
        ('Velocity\n$\\Delta\\mathbf{x}_t$',     '#FF9800'),
        ('Acceleration\n$\\Delta^2\\mathbf{x}_t','#03A9F4'),
        ('',                                     '#aaaaaa'),
    ]
    for ri, (lbl, lcol) in enumerate(row_labels):
        fig.text(label_x, row_mids[ri], lbl, va='center', ha='left',
                 fontsize=8, color=lcol, rotation=90, fontweight='bold')

    # ── Per-column axis limits (shared for rows 0-2) ──────────────────
    # We'll compute limits per column from the position frames.
    col_lims = []
    for ci in range(disp_frames):
        xy = frames[ci]
        valid = xy[~np.all(xy == 0, axis=1)]
        if len(valid) == 0:
            col_lims.append(None)
            continue
        x0, x1 = valid[:, 0].min(), valid[:, 0].max()
        y0, y1 = valid[:, 1].min(), valid[:, 1].max()
        px = max((x1 - x0) * 0.32, 0.025)
        py = max((y1 - y0) * 0.32, 0.025)
        col_lims.append((x0 - px, x1 + px, y1 + py, y0 - py))

    def _apply_lim(ax, ci):
        if col_lims[ci]:
            xl0, xl1, yl0, yl1 = col_lims[ci]
            ax.set_xlim(xl0, xl1)
            ax.set_ylim(yl0, yl1)

    # ── Row 0: Position ───────────────────────────────────────────────
    for ci in range(disp_frames):
        ax = fig.add_subplot(gs[0, ci])
        _draw_hand(ax, frames[ci], JOINT_COL, CONN_COL, lw=1.6, s=22)
        _clean_ax(ax, title=f'$t_{{{ci}}}$' if ci == 0 else f'$t_{{{ci}}}$',
                  title_color='#888888')
        _apply_lim(ax, ci)

    # ── Row 1: Velocity ───────────────────────────────────────────────
    for ci in range(disp_frames):
        ax = fig.add_subplot(gs[1, ci])
        if ci < len(vel):
            norm_v, _ = _draw_motion_arrows(ax, frames[ci], vel[ci],
                                            CMAP_VEL, vel_vmin, vel_vmax)
        else:
            # No more velocity to show — draw ghosted skeleton
            _draw_hand(ax, frames[ci], GHOST_JOINT, GHOST_CONN,
                       lw=1.0, s=12, alpha=0.3)
        _clean_ax(ax)
        _apply_lim(ax, ci)

    # ── Row 2: Acceleration ───────────────────────────────────────────
    for ci in range(disp_frames):
        ax = fig.add_subplot(gs[2, ci])
        if ci < len(acc):
            _draw_motion_arrows(ax, frames[ci], acc[ci],
                                CMAP_ACC, acc_vmin, acc_vmax)
        else:
            _draw_hand(ax, frames[ci], GHOST_JOINT, GHOST_CONN,
                       lw=1.0, s=12, alpha=0.3)
        _clean_ax(ax)
        _apply_lim(ax, ci)

    # ── Row 3: Feature-concatenation schematic ────────────────────────
    ax_schem = fig.add_subplot(gs[3, :])
    ax_schem.set_facecolor('#12121f')
    ax_schem.set_xlim(0, 1)
    ax_schem.set_ylim(0, 1)
    ax_schem.set_xticks([]); ax_schem.set_yticks([])
    for sp in ax_schem.spines.values():
        sp.set_visible(False)

    # Three equal blocks representing the feature streams
    blocks = [
        (0.04, 0.29, '$\\mathbf{x}_t$\n(position)',     '#4CAF50',  '63-d',  'right-hand\nx, y, z'),
        (0.37, 0.29, '$\\Delta\\mathbf{x}_t$\n(velocity)',   '#FF9800',  '63-d',  '1st difference'),
        (0.70, 0.29, '$\\Delta^2\\mathbf{x}_t$\n(accel.)',   '#03A9F4',  '63-d',  '2nd difference'),
    ]
    bw, bh = 0.27, 0.68
    for bx, by, label, col, dim_label, sublabel in blocks:
        rect = mpatches.FancyBboxPatch((bx, by), bw, bh,
                                       boxstyle='round,pad=0.01',
                                       facecolor=col + '22',
                                       edgecolor=col, linewidth=2,
                                       transform=ax_schem.transAxes,
                                       clip_on=False)
        ax_schem.add_patch(rect)
        ax_schem.text(bx + bw / 2, by + bh * 0.72, label,
                      ha='center', va='center', fontsize=9,
                      color=col, fontweight='bold',
                      transform=ax_schem.transAxes)
        ax_schem.text(bx + bw / 2, by + bh * 0.38, dim_label,
                      ha='center', va='center', fontsize=10,
                      color='white', fontweight='bold',
                      transform=ax_schem.transAxes)
        ax_schem.text(bx + bw / 2, by + bh * 0.16, sublabel,
                      ha='center', va='center', fontsize=7.5,
                      color='#aaaaaa',
                      transform=ax_schem.transAxes)

    # Plus signs between blocks
    for px in [0.33, 0.66]:
        ax_schem.text(px, 0.29 + bh / 2, '⊕', ha='center', va='center',
                      fontsize=14, color='#888888',
                      transform=ax_schem.transAxes)

    # Arrow + label on the right
    ax_schem.annotate('', xy=(0.985, 0.29 + bh / 2),
                      xytext=(0.975, 0.29 + bh / 2),
                      xycoords='axes fraction',
                      arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=2))
    ax_schem.text(0.995, 0.29 + bh / 2,
                  '189-d\nper stream',
                  ha='left', va='center', fontsize=8,
                  color='white', transform=ax_schem.transAxes)

    ax_schem.text(0.5, 0.03,
                  'Right-hand stream  (21 joints × 3 coords × 3 features = 189-d)',
                  ha='center', va='bottom', fontsize=8,
                  color='#666677', transform=ax_schem.transAxes)

    # ── Colourbars ────────────────────────────────────────────────────
    cb_ax_v = fig.add_axes([0.91, 0.72, 0.012, 0.18])
    cb_ax_a = fig.add_axes([0.91, 0.49, 0.012, 0.18])
    for cb_ax, cmap, vmin, vmax, label, col in [
        (cb_ax_v, CMAP_VEL, vel_vmin, vel_vmax, 'vel. mag.', '#FF9800'),
        (cb_ax_a, CMAP_ACC, acc_vmin, acc_vmax, 'accel. mag.', '#03A9F4'),
    ]:
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
        cb = plt.colorbar(sm, cax=cb_ax)
        cb.set_label(label, fontsize=7, color=col, labelpad=3)
        cb.ax.yaxis.set_tick_params(color='#888888', labelsize=6, labelcolor='#888888')
        cb.outline.set_visible(False)
        cb.ax.set_facecolor(BG)

    # ── Title ─────────────────────────────────────────────────────────
    seq_name = Path(npy_path).stem
    fig.suptitle(
        f'Motion feature augmentation  ·  right-hand stream  '
        f'[{disp_frames} consecutive frames  ·  frames {start}–{start+disp_frames-1}]',
        fontsize=10, color='#ccccdd', y=0.97,
    )

    # ── Legend ────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=JOINT_COL,  label='Position $\\mathbf{x}_t$'),
        mpatches.Patch(color='#FF9800',  label='Velocity $\\Delta\\mathbf{x}_t$'),
        mpatches.Patch(color='#03A9F4',  label='Acceleration $\\Delta^2\\mathbf{x}_t$'),
    ]
    fig.legend(handles=legend_handles, loc='lower right', ncol=3,
               fontsize=8, framealpha=0.12,
               facecolor='#1a1a2e', edgecolor='#333344',
               labelcolor='white', bbox_to_anchor=(0.90, 0.005))

    # ── Save ──────────────────────────────────────────────────────────
    stem = output_dir / 'motion_features_vis'
    fig.savefig(f'{stem}.png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    fig.savefig(f'{stem}.pdf', bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'Saved: {stem}.png / .pdf')
    plt.close(fig)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _pick_best_npy(data_root: str) -> str:
    """
    Among all available .npy files, pick the one with the highest right-hand
    activity so the arrows are clearly visible.
    """
    candidates = glob.glob(f'{data_root}/**/*holistic*.npy', recursive=True)
    if not candidates:
        candidates = glob.glob(f'{data_root}/**/*.npy', recursive=True)
    if not candidates:
        raise FileNotFoundError(f'No .npy files found under {data_root}')

    best_path, best_score = candidates[0], -1.0
    for p in candidates[:40]:      # sample up to 40 files
        try:
            raw = np.load(p)
            if raw.ndim == 3 and raw.shape[1] == 543:
                rh = raw[:, RHAND_START:RHAND_END, :2]
            elif raw.ndim == 2 and raw.shape[1] == 225:
                rh = raw.reshape(-1, 75, 3)[:, 34:55, :2]
            else:
                continue
            vis    = (~np.all(rh == 0, axis=-1)).mean()
            motion = np.linalg.norm(np.diff(rh, axis=0), axis=-1).mean()
            score  = vis * motion
            if score > best_score:
                best_score, best_path = score, p
        except Exception:
            continue

    print(f'Selected: {best_path}  (activity score {best_score:.4f})')
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description='Motion feature augmentation manuscript figure')
    parser.add_argument('--npy', default=None,
                        help='Path to a (T, 543, 3) or (T, 225) .npy file')
    parser.add_argument('--data_root', default='data',
                        help='Root dir to search for .npy files if --npy not given')
    parser.add_argument('--output_dir', default='figures')
    parser.add_argument('--n_frames', type=int, default=5)
    parser.add_argument('--dpi', type=int, default=220)
    args = parser.parse_args()

    npy = args.npy or _pick_best_npy(args.data_root)
    make_motion_figure(npy,
                       output_dir=Path(args.output_dir),
                       n_frames=args.n_frames,
                       dpi=args.dpi)
    print('Done.')


if __name__ == '__main__':
    main()
