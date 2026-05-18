"""
Generate the two remaining paper figures:
  1. figures/transfer_summary.pdf   – How2Sign condition comparison (bar chart)
  2. figures/phoenix_results.pdf    – PHOENIX condition comparison (bar chart)

Both replace the \figurepanel placeholders in paper/main.tex.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'paper', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (white-background friendly)
# ──────────────────────────────────────────────────────────────────────────────
C_PRIOR   = '#90A4AE'   # grey-blue   — prior work
C_BASE    = '#B0BEC5'   # light grey  — our baselines
C_GOOD    = '#42A5F5'   # mid blue    — good results
C_BEST    = '#1565C0'   # dark blue   — best results

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Transfer-learning / How2Sign summary
# ══════════════════════════════════════════════════════════════════════════════

methods = [
    'CTC baseline\n(H2S scratch)',
    'Multi-stream\n(H2S scratch)',
    'PHOENIX\ntransfer',
    'YouTube-ASL\npretrain only',
    r'YouTube-ASL$\to$H2S',
]
kw_wer  = [0.72, 0.68, 0.62, 0.50, 0.55]
bleu4   = [0.00, 0.00, 0.00, 4.00, 6.00]
colors  = [C_BASE, C_BASE, C_GOOD, C_GOOD, C_BEST]

x = np.arange(len(methods))
width = 0.36

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
fig.patch.set_facecolor('white')

# Left panel: KW-WER (lower is better)
ax = axes[0]
ax.set_facecolor('white')
bars = ax.bar(x, kw_wer, width=0.6, color=colors, edgecolor='#333333',
              linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel('KW-WER  ↓', fontsize=10, labelpad=6)
ax.set_ylim(0, 0.85)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Keyword WER', fontsize=11, pad=8, color='#222222')
for bar, v in zip(bars, kw_wer):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
            f'{v:.2f}', ha='center', va='bottom', fontsize=8.5,
            color='#222222')

# Right panel: Trans. BLEU-4 (higher is better)
ax = axes[1]
ax.set_facecolor('white')
bars = ax.bar(x, bleu4, width=0.6, color=colors, edgecolor='#333333',
              linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=8.5)
ax.set_ylabel('Trans. BLEU-4  ↑', fontsize=10, labelpad=6)
ax.set_ylim(0, 8.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_title('Translation BLEU-4', fontsize=11, pad=8, color='#222222')
for bar, v in zip(bars, bleu4):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8.5,
                color='#222222')
    else:
        ax.text(bar.get_x() + bar.get_width() / 2, 0.15,
                '≈0', ha='center', va='bottom', fontsize=8.0,
                color='#666666')

fig.tight_layout(pad=1.5)

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT_DIR, f'transfer_summary.{ext}'),
                dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
plt.close(fig)
print("Saved transfer_summary.pdf / .png")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PHOENIX results comparison (replaces training-curves placeholder)
# ══════════════════════════════════════════════════════════════════════════════

phoenix_methods = [
    'Flat\n(Sign2Gloss)',
    'Flat\n(S2G2T)',
    'Flat\n(Glossless)',
    'Multistream\n(Sign2Gloss)',
    'Multistream\n(S2G2T)',
]
phoenix_wer    = [0.411, 0.464, None,  0.270, 0.290]
phoenix_bleu4g = [15.72, 10.17, None,  17.48, None ]
phoenix_bleu4t = [None,  9.54,  0.00,  None,  16.2 ]
ph_colors      = [C_BASE, C_GOOD, C_BASE, C_GOOD, C_BEST]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
fig.patch.set_facecolor('white')

panels = [
    ('WER  ↓',            phoenix_wer,    'WER', 0, 0.58),
    ('Gloss BLEU-4  ↑',   phoenix_bleu4g, 'Gloss BLEU-4', 0, 21),
    ('Trans. BLEU-4  ↑',  phoenix_bleu4t, 'Trans. BLEU-4', 0, 20),
]

for ax, (ylabel, vals, title, ymin, ymax) in zip(axes, panels):
    ax.set_facecolor('white')
    xpos = np.arange(len(phoenix_methods))
    plot_vals = [v if v is not None else 0 for v in vals]
    hatch_list = ['//' if v is None else '' for v in vals]
    bars = ax.bar(xpos, plot_vals, width=0.6, color=ph_colors,
                  edgecolor='#333333', linewidth=0.8, hatch=None)
    # Apply hatching individually for N/A bars
    for bar, h in zip(bars, hatch_list):
        if h:
            bar.set_hatch(h)
            bar.set_facecolor('#e0e0e0')
    ax.set_xticks(xpos)
    ax.set_xticklabels(phoenix_methods, fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.set_ylim(ymin, ymax)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, fontsize=11, pad=8, color='#222222')
    for bar, v in zip(bars, vals):
        if v is None:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    ymax * 0.05, 'N/A', ha='center', va='bottom',
                    fontsize=8.5, color='#888888', style='italic')
        else:
            offset = ymax * 0.02
            ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                    f'{v:.2f}' if v < 1 else f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8.5, color='#222222')

# Legend patch for N/A
na_patch = mpatches.Patch(facecolor='#e0e0e0', edgecolor='#333333',
                           hatch='//', label='Not applicable / not measured')
fig.legend(handles=[na_patch], loc='lower center', ncol=1,
           framealpha=0.0, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(pad=1.5)
fig.subplots_adjust(bottom=0.18)

for ext in ('pdf', 'png'):
    fig.savefig(os.path.join(OUT_DIR, f'phoenix_results.{ext}'),
                dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
plt.close(fig)
print("Saved phoenix_results.pdf / .png")
