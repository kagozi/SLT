"""
Extract encoder hidden states and visualise with UMAP / t-SNE.

Loads a trained (or pretrained) checkpoint, runs the encoder on samples from
PHOENIX and How2Sign val splits, then plots:
  - UMAP coloured by dataset (DGS vs ASL)
  - UMAP coloured by pseudo-gloss semantic cluster (top-30 most frequent)
  - t-SNE coloured by dataset
  - Attention heatmap for one sample from each dataset
  - Pairwise cosine similarity matrix between languages

All plots are saved to W&B (project="slt", run="feature-extraction-<exp_name>")
and locally to <output_dir>/plots/.

Usage (NRP):
    kubectl apply -f nautilius/extract-features-job.yaml

Usage (local):
    python extract_features.py \\
        --checkpoint /data/experiments/exp1_phoenix_sign2gloss/models/best_model.pt \\
        --phoenix_dir /data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
        --how2sign_dir /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features \\
        --n_samples 500 \\
        --output_dir /data/experiments/feature_analysis
"""

import io
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import wandb
from PIL import Image as PILImage

from dataset import PhoenixSignDataset
from dataset_how2sign import How2SignDataset
from models import SignLanguageTransformer
from utils import GlossTokenizer, collate_fn


# ── Helpers ───────────────────────────────────────────────────────────

def _fig_to_wandb(fig) -> wandb.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(PILImage.open(buf).copy())


def _ax_style(ax, title, xlabel='', ylabel=''):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.spines[['top','right']].set_visible(False)


# ── Feature extraction ────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, loader, device, n_samples=500):
    """
    Extract mean-pooled encoder hidden states.
    Returns:
        features: (N, dim) float32 numpy
        glosses:  list of N gloss strings
        names:    list of N sample names
    """
    model.eval()
    features, glosses, names = [], [], []

    for batch in loader:
        kps  = batch['keypoints'].to(device)
        mask = batch['mask'].to(device)
        hidden, _ = model.encode(kps, mask)           # (B, T, dim)

        # Mean-pool over valid (non-padded) frames
        valid_mask = mask.unsqueeze(-1)                # (B, T, 1)
        pooled = (hidden * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)  # (B, dim)
        features.append(pooled.cpu().numpy())

        gloss_batch = batch.get('gloss_text', [''] * kps.shape[0])
        names_batch = batch.get('name',       [''] * kps.shape[0])
        glosses.extend(gloss_batch)
        names.extend(names_batch)

        if len(glosses) >= n_samples:
            break

    features = np.vstack(features)[:n_samples]
    glosses  = glosses[:n_samples]
    names    = names[:n_samples]
    return features, glosses, names


# ── UMAP / t-SNE plots ────────────────────────────────────────────────

def plot_by_dataset(ph_feats, h2s_feats, method='umap', output_dir=None):
    """
    2-D projection coloured by dataset origin (DGS vs ASL).
    Returns wandb.Image.
    """
    combined = np.vstack([ph_feats, h2s_feats])
    labels   = ['DGS (PHOENIX)'] * len(ph_feats) + ['ASL (How2Sign)'] * len(h2s_feats)

    embedding = _project(combined, method)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {'DGS (PHOENIX)': '#2196F3', 'ASL (How2Sign)': '#F44336'}
    for label, color in colors.items():
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=color, label=label, alpha=0.55, s=18, linewidths=0)
    _ax_style(ax, f'{method.upper()} — Feature Space by Dataset',
              f'{method.upper()} dim 1', f'{method.upper()} dim 2')
    ax.legend(markerscale=2, fontsize=10)
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / f'{method}_by_dataset.png', dpi=150, bbox_inches='tight')
    return _fig_to_wandb(fig)


def plot_by_gloss(ph_feats, ph_glosses, h2s_feats, h2s_glosses,
                  method='umap', top_k=15, output_dir=None):
    """
    2-D projection coloured by most-frequent pseudo-gloss tokens.
    Shows whether same-gloss samples cluster across languages.
    """
    # Find top-k glosses shared or frequent across both
    all_glosses = ph_glosses + h2s_glosses
    gloss_tokens = []
    for g in all_glosses:
        gloss_tokens.extend(str(g).upper().split())
    from collections import Counter
    top_glosses = [g for g, _ in Counter(gloss_tokens).most_common(top_k)]
    top_set = set(top_glosses)

    # Assign colour label: first matching token, else 'other'
    def _label(gloss_str):
        for tok in str(gloss_str).upper().split():
            if tok in top_set:
                return tok
        return 'other'

    ph_labels  = [_label(g) for g in ph_glosses]
    h2s_labels = [_label(g) for g in h2s_glosses]
    all_labels = ph_labels + h2s_labels

    combined = np.vstack([ph_feats, h2s_feats])
    embedding = _project(combined, method)

    unique_labels = top_glosses + ['other']
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    color_map = {l: cmap(i) for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(11, 8))
    for label in unique_labels:
        idx = [i for i, l in enumerate(all_labels) if l == label]
        if not idx:
            continue
        alpha = 0.6 if label != 'other' else 0.15
        size  = 20  if label != 'other' else 8
        ax.scatter(embedding[idx, 0], embedding[idx, 1],
                   c=[color_map[label]], label=label if label != 'other' else '_',
                   alpha=alpha, s=size, linewidths=0)
    _ax_style(ax, f'{method.upper()} — Feature Space by Gloss Token (top {top_k})',
              f'{method.upper()} dim 1', f'{method.upper()} dim 2')
    ax.legend(markerscale=2, fontsize=8, ncol=2, loc='upper right')
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / f'{method}_by_gloss.png', dpi=150, bbox_inches='tight')
    return _fig_to_wandb(fig)


def _project(features, method='umap'):
    """Dimensionality reduction to 2-D."""
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                metric='cosine', random_state=42)
            return reducer.fit_transform(features)
        except ImportError:
            print("  umap-learn not installed, falling back to t-SNE")
            method = 'tsne'
    # t-SNE
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, perplexity=30, n_iter=1000,
                random_state=42, metric='cosine').fit_transform(features)


# ── Cosine similarity analysis ────────────────────────────────────────

def plot_cross_language_similarity(ph_feats, h2s_feats, output_dir=None):
    """
    Plot distribution of cosine similarities:
    - intra-DGS (within PHOENIX)
    - intra-ASL (within How2Sign)
    - cross-lingual (DGS vs ASL)
    If cross-lingual similarity ≈ intra-language, the encoder is language-agnostic.
    """
    def _cosine_sample(A, B, n=2000):
        idx_a = np.random.choice(len(A), min(n, len(A)), replace=False)
        idx_b = np.random.choice(len(B), min(n, len(B)), replace=False)
        a = A[idx_a] / (np.linalg.norm(A[idx_a], axis=1, keepdims=True) + 1e-8)
        b = B[idx_b] / (np.linalg.norm(B[idx_b], axis=1, keepdims=True) + 1e-8)
        return (a * b).sum(axis=1)

    # Sample pairs
    intra_ph   = _cosine_sample(ph_feats,  ph_feats)
    intra_h2s  = _cosine_sample(h2s_feats, h2s_feats)
    cross      = _cosine_sample(ph_feats,  h2s_feats)

    fig, ax = plt.subplots(figsize=(9, 4))
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(intra_ph,  bins=bins, alpha=0.6, color='#2196F3', label='Intra-DGS (PHOENIX)',   density=True)
    ax.hist(intra_h2s, bins=bins, alpha=0.6, color='#F44336', label='Intra-ASL (How2Sign)',  density=True)
    ax.hist(cross,     bins=bins, alpha=0.6, color='#4CAF50', label='Cross-lingual (DGS↔ASL)', density=True)
    ax.axvline(np.mean(intra_ph),  ls='--', color='#1565C0', lw=1.5)
    ax.axvline(np.mean(intra_h2s), ls='--', color='#B71C1C', lw=1.5)
    ax.axvline(np.mean(cross),     ls='--', color='#2E7D32', lw=1.5)
    _ax_style(ax, 'Cosine Similarity Distribution', 'Cosine Similarity', 'Density')
    ax.legend(fontsize=9)

    # Text: mean values
    ax.text(0.02, 0.95,
            f"Mean intra-DGS: {np.mean(intra_ph):.3f}\n"
            f"Mean intra-ASL: {np.mean(intra_h2s):.3f}\n"
            f"Mean cross:     {np.mean(cross):.3f}",
            transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    fig.tight_layout()
    if output_dir:
        fig.savefig(Path(output_dir) / 'cosine_similarity.png', dpi=150, bbox_inches='tight')
    return _fig_to_wandb(fig)


# ── Attention heatmap ────────────────────────────────────────────────

@torch.no_grad()
def extract_attention(model, batch, device, layer_idx=0):
    """
    Extract attention weights from a TransformerBlock by hooking into
    the MultiHeadSelfAttention forward pass.
    Returns (T, T) numpy array averaged over heads.
    """
    attn_weights = {}

    def _hook(module, inp, out):
        B, T, C = inp[0].shape
        qkv = module.qkv(inp[0])
        qkv = qkv.reshape(B, T, 3, module.num_heads, C // module.num_heads).permute(2,0,3,1,4)
        q, k = qkv[0], qkv[1]
        attn = (q @ k.transpose(-2,-1)) * module.scale
        # Apply mask if present
        mask = inp[1] if len(inp) > 1 else None
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn_weights['attn'] = attn.detach().cpu()

    blk = model.transformer_blocks[layer_idx]
    handle = blk.attn.register_forward_hook(_hook)

    kps  = batch['keypoints'][:1].to(device)
    mask = batch['mask'][:1].to(device)
    model.encode(kps, mask)
    handle.remove()

    if 'attn' not in attn_weights:
        return None
    # Average over heads, first sample → (T, T)
    return attn_weights['attn'][0].mean(0).numpy()


def plot_attention_heatmap(attn_matrix, title, max_frames=80, output_dir=None, fname=None):
    """Plot a (T, T) attention heatmap for a single sample."""
    A = attn_matrix[:max_frames, :max_frames]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(A, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _ax_style(ax, title, 'Key Frame', 'Query Frame')
    fig.tight_layout()
    if output_dir and fname:
        fig.savefig(Path(output_dir) / fname, dpi=150, bbox_inches='tight')
    return _fig_to_wandb(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract encoder features and visualise with UMAP/t-SNE')
    parser.add_argument('--checkpoint',   type=str, required=True,
                        help='Path to .pt model checkpoint')
    parser.add_argument('--phoenix_dir',  type=str,
                        default='/data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T')
    parser.add_argument('--how2sign_dir', type=str,
                        default='/data/hf_cache/How2Sign_Holistic/how2sign_holistic_features')
    parser.add_argument('--output_dir',   type=str, default='/data/experiments/feature_analysis')
    parser.add_argument('--n_samples',    type=int, default=500,
                        help='Max samples per dataset for projection (default: 500)')
    parser.add_argument('--max_frames',   type=int, default=300)
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--num_workers',  type=int, default=4)
    parser.add_argument('--seed',         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt_name = Path(args.checkpoint).parent.parent.name  # exp dir name
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project='slt',
        name=f'feature-analysis-{ckpt_name}',
        config=vars(args),
    )

    # ── Load checkpoint ──
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    config = ckpt.get('config', {})
    dim    = config.get('dim', 256)
    n_cls  = config.get('num_classes', 1085)

    # ── Rebuild tokenizers ──
    phoenix_dir  = Path(args.phoenix_dir)
    how2sign_dir = Path(args.how2sign_dir)

    ph_glosses = []
    for split in ['train', 'dev', 'test']:
        csv = phoenix_dir / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
        if csv.exists():
            df = pd.read_csv(csv, sep='|')
            ph_glosses.extend(df['orth'].dropna().tolist())
    ph_tok = GlossTokenizer(ph_glosses, min_freq=1)

    h2s_csv = how2sign_dir / 'annotations' / 'how2sign_train.csv'
    h2s_glosses = []
    if h2s_csv.exists():
        df_h = pd.read_csv(h2s_csv, sep='\t')
        if 'PSEUDOGLOSS' in df_h.columns:
            h2s_glosses = df_h['PSEUDOGLOSS'].dropna().tolist()
    h2s_tok = GlossTokenizer(h2s_glosses or ['DUMMY'], min_freq=2)

    # ── Model ──
    model = SignLanguageTransformer(
        input_dim=225, dim=dim, num_classes=n_cls,
        max_frames=args.max_frames * 2,
    )
    src = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(src, strict=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"  Device: {device}")

    # ── Datasets ──
    ph_val  = PhoenixSignDataset(phoenix_dir, 'dev',  args.max_frames, False, ph_tok,  None)
    h2s_val = How2SignDataset(how2sign_dir,   'val',  args.max_frames, False, h2s_tok, None)

    def _subsample(ds, n):
        idx = random.sample(range(len(ds)), min(n, len(ds)))
        return Subset(ds, idx)

    ph_sub  = _subsample(ph_val,  args.n_samples)
    h2s_sub = _subsample(h2s_val, args.n_samples)

    ph_loader  = DataLoader(ph_sub,  args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)
    h2s_loader = DataLoader(h2s_sub, args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # ── Extract features ──
    print(f"\nExtracting features ({args.n_samples} samples/dataset)...")
    ph_feats,  ph_glosses,  ph_names  = extract_features(model, ph_loader,  device, args.n_samples)
    h2s_feats, h2s_glosses, h2s_names = extract_features(model, h2s_loader, device, args.n_samples)
    print(f"  PHOENIX:  {ph_feats.shape}")
    print(f"  How2Sign: {h2s_feats.shape}")

    # ── Plots ──
    print("\nGenerating visualisations...")
    log_dict = {}

    # UMAP by dataset
    print("  UMAP by dataset...")
    log_dict['representations/umap_by_dataset'] = plot_by_dataset(
        ph_feats, h2s_feats, method='umap', output_dir=str(plots_dir))

    # UMAP by gloss
    print("  UMAP by gloss...")
    log_dict['representations/umap_by_gloss'] = plot_by_gloss(
        ph_feats, ph_glosses, h2s_feats, h2s_glosses,
        method='umap', top_k=15, output_dir=str(plots_dir))

    # t-SNE by dataset
    print("  t-SNE by dataset...")
    log_dict['representations/tsne_by_dataset'] = plot_by_dataset(
        ph_feats, h2s_feats, method='tsne', output_dir=str(plots_dir))

    # Cosine similarity
    print("  Cosine similarity distributions...")
    log_dict['representations/cosine_similarity'] = plot_cross_language_similarity(
        ph_feats, h2s_feats, output_dir=str(plots_dir))

    # Attention heatmaps
    print("  Attention heatmaps...")
    ph_batch  = next(iter(ph_loader))
    h2s_batch = next(iter(h2s_loader))

    for batch, lang_label, fname in [
        (ph_batch,  'PHOENIX (DGS)',   'attn_phoenix.png'),
        (h2s_batch, 'How2Sign (ASL)',  'attn_how2sign.png'),
    ]:
        attn = extract_attention(model, batch, device, layer_idx=-1)
        if attn is not None:
            key = f'representations/attention_{lang_label.split()[0].lower()}'
            log_dict[key] = plot_attention_heatmap(
                attn, f'Attention (last transformer block) — {lang_label}',
                output_dir=str(plots_dir), fname=fname)

    # Attention heatmaps for each layer
    n_layers = len(model.transformer_blocks)
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    for li, ax in enumerate(axes):
        attn = extract_attention(model, ph_batch, device, layer_idx=li)
        if attn is not None:
            A = attn[:60, :60]
            ax.imshow(A, aspect='auto', cmap='viridis')
            ax.set_title(f'Layer {li}', fontsize=10)
            ax.set_xlabel('Key'); ax.set_ylabel('Query')
    fig.suptitle('Attention per Layer (PHOENIX sample)', fontweight='bold')
    fig.tight_layout()
    log_dict['representations/attention_all_layers'] = _fig_to_wandb(fig)

    # Log all to W&B
    wandb.log(log_dict)
    print(f"\n  ✅ {len(log_dict)} plots saved to W&B and {plots_dir}")

    wandb.finish()
    print("\n✨ Feature analysis complete.")


if __name__ == '__main__':
    main()
