"""
Train Sign Language Transformer with configurable modes.

All results auto-save to ../results/<exp_name>/ and ../models/<exp_name>/.
After training, evaluates on both val (dev) and test sets, then runs a beam
size sweep (widths 1–32) on both splits and logs everything to W&B.

PHOENIX-2014-T experiments:
  # Exp 1: Sign2Gloss (CTC-only)
  python train.py --exp_name exp1_phoenix_sign2gloss \\
      --dim 256 --epochs 200 --decode beam --beam_width 10

  # Exp 2: Sign2Gloss2Text (joint CTC + BART)
  python train.py --exp_name exp2_phoenix_sign2gloss2text \\
      --dim 256 --epochs 200 --decode beam --beam_width 10 \\
      --use_bart --ctc_weight 0.5 --freeze_bart_epochs 20

  # Exp 3: Glossless Sign2Text (BART-only)
  python train.py --exp_name exp3_phoenix_glossless \\
      --dim 256 --epochs 200 --decode beam --beam_width 10 \\
      --use_bart --ctc_weight 0.0 --freeze_bart_epochs 0

How2Sign experiments:
  # Exp 4: How2Sign Glossless (BART-only, no pseudo-glosses)
  python train.py --dataset how2sign --exp_name exp4_how2sign_glossless \\
      --root_dir /data/how2sign_rgb \\
      --dim 256 --epochs 200 --max_frames 300 --decode beam --beam_width 10 \\
      --use_bart --ctc_weight 0.0 --freeze_bart_epochs 0

  # Exp 5: How2Sign Sign2Gloss2Text (joint CTC + BART with pseudo-glosses)
  python train.py --dataset how2sign --exp_name exp5_how2sign_sign2gloss2text \\
      --root_dir /data/how2sign_rgb \\
      --dim 256 --epochs 200 --max_frames 300 --decode beam --beam_width 10 \\
      --use_bart --ctc_weight 0.5 --freeze_bart_epochs 20

  # Exp 6: How2Sign Sign2Gloss (CTC-only, pseudo-glosses as labels)
  python train.py --dataset how2sign --exp_name exp6_how2sign_sign2gloss \\
      --root_dir /data/how2sign_rgb \\
      --dim 256 --epochs 200 --max_frames 300 --decode beam --beam_width 10
"""

import io
import time
import argparse
import pandas as pd
from collections import Counter
import math
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb


def _fig_to_wandb(fig) -> wandb.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as _PILImage
    return wandb.Image(_PILImage.open(buf).copy())

from dataset import PhoenixSignDataset
from dataset_how2sign import How2SignDataset
from models import SignLanguageTransformer
from utils import GlossTokenizer, Trainer, collate_fn, ctc_greedy_decode, ctc_beam_decode


# ─── Experiment Naming ───────────────────────────────────────────────

def get_experiment_name(args):
    """Generate a descriptive experiment name from args."""
    if args.exp_name:
        return args.exp_name
    
    dataset_tag = args.dataset  # 'phoenix' or 'how2sign'
    
    if not args.use_bart:
        mode = "sign2gloss"
    elif args.ctc_weight == 0:
        mode = "glossless_sign2text"
    else:
        mode = f"sign2gloss2text_ctc{args.ctc_weight}"
    
    parts = [
        dataset_tag,
        mode,
        f"e{args.epochs}",
        f"d{args.dim}",
        f"{args.decode}_bw{args.beam_width}",
    ]
    if args.use_bart:
        parts.append(f"freeze{args.freeze_bart_epochs}")
    
    return "_".join(parts)


def setup_directories(exp_name, output_dir=None):
    """Create result and model directories."""
    base = Path(output_dir) if output_dir else Path("..")
    results_dir = base / "results" / exp_name
    models_dir = base / "models" / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, models_dir


# ─── Evaluator ───────────────────────────────────────────────────────

class SignLanguageEvaluator:
    def __init__(self, model, test_loader, tokenizer, device='cuda',
                 bart_tokenizer=None, decode_mode='greedy', beam_width=5):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.device = device
        self.decode_mode = decode_mode
        self.beam_width = beam_width

    @torch.no_grad()
    def evaluate(self, verbose=True):
        self.model.eval()
        all_preds, all_targets = [], []
        all_trans_preds, all_trans_targets = [], []
        all_names = []
        inference_times = []

        for batch_idx, batch in enumerate(self.test_loader):
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)

            start = time.time()
            if self.model.use_bart:
                output = self.model(keypoints, mask)
                logits = output['logits']
                mask_out = output['mask']
            else:
                logits, mask_out = self.model(keypoints, mask)
            inference_times.append(time.time() - start)

            logits_cpu = logits.cpu()
            input_lengths = mask_out.sum(dim=1).long().cpu()

            if self.decode_mode == 'beam':
                preds = ctc_beam_decode(logits_cpu, input_lengths, self.beam_width)
            else:
                preds = ctc_greedy_decode(logits_cpu, input_lengths)
            for p in preds:
                all_preds.append(self.tokenizer.decode(p))
            all_targets.extend(batch['gloss_text'])
            all_names.extend(batch['name'])
            
            if self.model.use_bart and self.bart_tokenizer is not None:
                try:
                    token_ids = self.model.translate(keypoints, mask, beam_width=self.beam_width)
                    for i in range(token_ids.shape[0]):
                        text = self.bart_tokenizer.decode(token_ids[i], skip_special_tokens=True)
                        all_trans_preds.append(text)
                    all_trans_targets.extend(batch['translation'])
                except Exception as e:
                    if verbose:
                        print(f"  Translation failed: {e}")

            if verbose and batch_idx % 10 == 0:
                print(f"  Eval batch {batch_idx}/{len(self.test_loader)}")

        metrics = self._compute_metrics(all_preds, all_targets, inference_times)
        if all_trans_preds:
            trans_bleu = self._compute_bleu(all_trans_preds, all_trans_targets)
            metrics['trans_BLEU-1'] = trans_bleu.get('BLEU-1', 0)
            metrics['trans_BLEU-4'] = trans_bleu.get('BLEU-4', 0)
            metrics['trans_BLEU'] = trans_bleu.get('BLEU', 0)

        return metrics, {
            'preds': all_preds, 'targets': all_targets, 'names': all_names,
            'trans_preds': all_trans_preds, 'trans_targets': all_trans_targets,
        }

    def _compute_metrics(self, preds, targets, times):
        metrics = {
            'total_samples': len(preds),
            'avg_inference_ms': float(np.mean(times) * 1000),
            'WER': float(self._compute_wer(preds, targets)),
        }
        bleu = self._compute_bleu(preds, targets)
        metrics.update(bleu)
        exact = sum(1 for p, t in zip(preds, targets) if p.strip() == t.strip())
        metrics['exact_match'] = exact / len(preds) if preds else 0
        return metrics

    def _compute_wer(self, preds, targets):
        def wer_single(p, t):
            pw, tw = p.split(), t.split()
            if not tw: return 0.0
            d = np.zeros((len(tw)+1, len(pw)+1))
            for i in range(len(tw)+1): d[i,0] = i * 3
            for j in range(len(pw)+1): d[0,j] = j * 3
            for i in range(1, len(tw)+1):
                for j in range(1, len(pw)+1):
                    cost = 0 if tw[i-1] == pw[j-1] else 4
                    d[i,j] = min(d[i-1,j]+3, d[i,j-1]+3, d[i-1,j-1]+cost)
            mc = 4 * max(len(tw), len(pw))
            return d[len(tw), len(pw)] / mc if mc > 0 else 0.0
        return float(np.mean([wer_single(p, t) for p, t in zip(preds, targets)]))

    def _compute_bleu(self, preds, targets, max_n=4):
        def ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        scores = {}
        for n in range(1, max_n+1):
            prec_sum, count = 0, 0
            for p, t in zip(preds, targets):
                pt, tt = p.split(), t.split()
                if len(pt) < n or len(tt) < n: continue
                pn, tn = Counter(ngrams(pt, n)), Counter(ngrams(tt, n))
                matches = sum((pn & tn).values())
                total = sum(pn.values())
                if total > 0:
                    prec_sum += matches / total
                    count += 1
            scores[f'BLEU-{n}'] = prec_sum / count if count > 0 else 0.0
        if all(v > 0 for v in scores.values()):
            scores['BLEU'] = math.exp(sum(math.log(v) for v in scores.values()) / len(scores))
        else:
            scores['BLEU'] = 0.0
        return scores

    def print_metrics(self, metrics):
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        print(f"  Decode:  {self.decode_mode} (beam={self.beam_width})")
        print(f"  Samples: {metrics['total_samples']}")
        print(f"  WER:     {metrics['WER']:.2%}")
        print(f"  BLEU-1:  {metrics.get('BLEU-1', 0):.4f}")
        print(f"  BLEU-4:  {metrics.get('BLEU-4', 0):.4f}")
        print(f"  Exact:   {metrics.get('exact_match', 0):.2%}")
        if 'trans_BLEU-1' in metrics:
            print(f"  --- Translation (German) ---")
            print(f"  Trans BLEU-1: {metrics['trans_BLEU-1']:.4f}")
            print(f"  Trans BLEU-4: {metrics['trans_BLEU-4']:.4f}")
        print("="*60)


# ─── Save Results ────────────────────────────────────────────────────

def save_experiment(args, exp_name, results_dir, models_dir,
                    metrics, eval_data, model, tokenizer):
    """Save all experiment artifacts to organized directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ── Metrics JSON ──
    metrics_out = {
        'experiment': exp_name,
        'timestamp': timestamp,
        'args': vars(args),
        'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in metrics.items()},
    }
    metrics_path = results_dir / f"metrics_{exp_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_out, f, indent=2)
    
    # ── Predictions CSV ──
    df = pd.DataFrame({
        'name': eval_data['names'],
        'target_gloss': eval_data['targets'],
        'predicted_gloss': eval_data['preds'],
        'correct': [p.strip() == t.strip() 
                    for p, t in zip(eval_data['preds'], eval_data['targets'])]
    })
    if eval_data['trans_preds']:
        tp = list(eval_data['trans_preds'])
        tt = list(eval_data['trans_targets'])
        while len(tp) < len(df): tp.append('')
        while len(tt) < len(df): tt.append('')
        df['target_translation'] = tt[:len(df)]
        df['predicted_translation'] = tp[:len(df)]
    preds_path = results_dir / f"predictions_{exp_name}.csv"
    df.to_csv(preds_path, index=False)
    
    # ── Final Model ──
    model_path = models_dir / f"final_{exp_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.gloss_to_idx,
        'config': {
            'dataset': args.dataset,
            'input_dim': args.input_dim_detected,
            'dim': args.dim,
            'num_classes': tokenizer.vocab_size,
            'max_frames': args.max_frames,
            'use_bart': args.use_bart,
            'bart_model': args.bart_model,
            'ctc_weight': args.ctc_weight,
        },
        'args': vars(args),
        'metrics': metrics,
    }, model_path)
    
    # ── Copy best_model.pt ──
    if Path('best_model.pt').exists():
        shutil.copy2('best_model.pt', models_dir / f"best_{exp_name}.pt")
    
    # ── Save args for reproducibility ──
    args_path = results_dir / f"args_{exp_name}.json"
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n📁 Experiment '{exp_name}' saved:")
    print(f"   📊 {metrics_path}")
    print(f"   📋 {preds_path}")
    print(f"   🤖 {model_path}")
    if (models_dir / f"best_{exp_name}.pt").exists():
        print(f"   ⭐ {models_dir / f'best_{exp_name}.pt'}")


# ─── Plotting helpers ────────────────────────────────────────────────

def _ax_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)


def _plot_beam_sweep(sweep_rows: list):
    """Plot WER, BLEU-4, trans_BLEU-4 vs beam width for val + test."""
    if not sweep_rows:
        return
    plots = {}
    has_trans = any(r['trans_bleu4'] > 0 for r in sweep_rows)

    for metric, ylabel, title, key in [
        ('WER',     'WER (lower is better)',    'WER vs Beam Width',          'WER'),
        ('bleu4',   'BLEU-4 (higher is better)','BLEU-4 vs Beam Width',       'bleu4'),
        ('bleu1',   'BLEU-1 (higher is better)','BLEU-1 vs Beam Width',       'bleu1'),
    ] + ([('trans_bleu4', 'BLEU-4 (higher is better)',
           'Translation BLEU-4 vs Beam Width', 'trans_bleu4')] if has_trans else []):

        fig, ax = plt.subplots(figsize=(7, 4))
        for split, color, marker in [('val', 'steelblue', 'o'), ('test', 'tomato', 's')]:
            rows = [r for r in sweep_rows if r['split'] == split]
            if rows:
                bws = [r['beam_width'] for r in rows]
                vals = [r[key] for r in rows]
                ax.plot(bws, vals, marker=marker, color=color,
                        linewidth=1.8, markersize=6, label=split)
        _ax_style(ax, title, 'Beam Width', ylabel)
        ax.set_xticks(BEAM_WIDTHS)
        ax.legend()
        fig.tight_layout()
        plots[f'beam_sweep/{key}_vs_beam'] = _fig_to_wandb(fig)

    if wandb.run is not None:
        wandb.log(plots)


def plot_val_test_comparison(val_metrics: dict, test_metrics: dict):
    """Bar chart comparing val vs test for WER, BLEU-1, BLEU-4, trans_BLEU-4."""
    metric_keys = ['WER', 'BLEU-1', 'BLEU-4']
    if val_metrics.get('trans_BLEU-4', 0) > 0 or test_metrics.get('trans_BLEU-4', 0) > 0:
        metric_keys.append('trans_BLEU-4')

    labels = metric_keys
    val_vals  = [val_metrics.get(k, 0) for k in labels]
    test_vals = [test_metrics.get(k, 0) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars_v = ax.bar(x - width/2, val_vals,  width, label='Val',  color='steelblue',  alpha=0.85)
    bars_t = ax.bar(x + width/2, test_vals, width, label='Test', color='tomato',     alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    _ax_style(ax, 'Val vs Test — Final Metrics', 'Metric', 'Score')
    ax.legend()
    # Annotate bars
    for bar in list(bars_v) + list(bars_t):
        h = bar.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    if wandb.run is not None:
        wandb.log({'plots/val_vs_test_bar': _fig_to_wandb(fig)})


# ─── Beam Sweep ──────────────────────────────────────────────────────

BEAM_WIDTHS = [1, 2, 4, 8, 16, 32]


def run_beam_sweep(model, val_loader, test_loader, tokenizer, device,
                   bart_tokenizer, exp_name):
    """
    Evaluate best model at beam widths 1–32 on val and test splits.
    Logs a W&B Table (beam_sweep) and a per-metric line chart.
    """
    print(f"\n🔍 Beam size sweep — {exp_name}")
    cols = ['split', 'beam_width', 'WER',
            'BLEU-1', 'BLEU-2', 'BLEU-4', 'trans_BLEU-4', 'exact_match']
    sweep_table = wandb.Table(columns=cols)
    sweep_rows = []   # for plotting

    for split_name, loader in [('val', val_loader), ('test', test_loader)]:
        for bw in BEAM_WIDTHS:
            ev = SignLanguageEvaluator(
                model, loader, tokenizer, device,
                bart_tokenizer=bart_tokenizer,
                decode_mode='beam', beam_width=bw,
            )
            m, _ = ev.evaluate(verbose=False)
            row = dict(
                split=split_name, beam_width=bw,
                WER=round(m.get('WER', 1.0), 4),
                bleu1=round(m.get('BLEU-1', 0), 4),
                bleu2=round(m.get('BLEU-2', 0), 4),
                bleu4=round(m.get('BLEU-4', 0), 4),
                trans_bleu4=round(m.get('trans_BLEU-4', 0), 4),
                exact=round(m.get('exact_match', 0), 4),
            )
            sweep_rows.append(row)
            sweep_table.add_data(
                split_name, bw,
                row['WER'], row['bleu1'], row['bleu2'], row['bleu4'],
                row['trans_bleu4'], row['exact'],
            )
            print(f"  [{split_name}] beam={bw:2d}  "
                  f"WER={row['WER']:.4f}  "
                  f"BLEU-4={row['bleu4']:.4f}  "
                  f"trans_BLEU-4={row['trans_bleu4']:.4f}")

    wandb.log({'beam_sweep/table': sweep_table})

    # ── Plot beam sweep charts ──
    _plot_beam_sweep(sweep_rows)
    print("  ✅ Beam sweep logged to W&B.")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Sign Language Transformer")
    
    # Data
    parser.add_argument('--dataset', type=str, default='phoenix',
                        choices=['phoenix', 'how2sign'],
                        help='Dataset to use: phoenix or how2sign')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Dataset root (auto-set per dataset if not provided)')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames (default: 250 for phoenix, 300 for how2sign)')
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    
    # Decoding
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_width', type=int, default=10)
    
    # BART translation
    parser.add_argument('--use_bart', action='store_true', help='Enable BART Gloss→Text')
    parser.add_argument('--bart_model', type=str, default='facebook/bart-base')
    parser.add_argument('--ctc_weight', type=float, default=0.3,
                        help='CTC loss weight (1.0=CTC only, 0.0=BART only)')
    parser.add_argument('--freeze_bart_epochs', type=int, default=5,
                        help='Epochs to freeze BART before joint training')
    
    # Experiment management
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Custom experiment name (auto-generated if not set)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base dir for results/ and models/ (default: ..)')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # ─── Dataset defaults ───
    if args.root_dir is None:
        if args.dataset == 'phoenix':
            args.root_dir = '/data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
        else:
            args.root_dir = '/data/how2sign_rgb'
    
    if args.max_frames is None:
        args.max_frames = 250 if args.dataset == 'phoenix' else 300
    
    # ─── Experiment Setup ───
    exp_name = get_experiment_name(args)
    results_dir, models_dir = setup_directories(exp_name, args.output_dir)

    wandb.init(
        project="slt",
        name=exp_name,
        config=vars(args),
        dir=str(results_dir),
    )

    print(f"\n{'='*60}")
    print(f"🧪 Experiment: {exp_name}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Results → {results_dir}")
    print(f"   Models  → {models_dir}")
    print(f"   Config:  epochs={args.epochs}, decode={args.decode}, "
          f"beam={args.beam_width}, bart={args.use_bart}, ctc_w={args.ctc_weight}")
    print(f"{'='*60}")
    
    # ─── Tokenizers ───
    print("\nBuilding tokenizers...")
    
    bart_tokenizer = None
    if args.use_bart:
        from transformers import BartTokenizer
        bart_tokenizer = BartTokenizer.from_pretrained(args.bart_model)
        print(f"  BART tokenizer: {args.bart_model}")
    
    if args.dataset == 'phoenix':
        # Build gloss tokenizer from PHOENIX annotations
        all_gloss = []
        for split in ['train', 'dev', 'test']:
            csv = Path(args.root_dir) / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
            df = pd.read_csv(csv, sep='|')
            all_gloss.extend(df['orth'].dropna().tolist())
        tokenizer = GlossTokenizer(all_gloss, min_freq=1)
    else:
        # How2Sign: build tokenizer from PSEUDOGLOSS column if present.
        pseudo_csv = Path(args.root_dir) / 'annotations' / 'how2sign_train.csv'
        pseudo_glosses = []
        if pseudo_csv.exists():
            df_h2s = pd.read_csv(pseudo_csv, sep='\t')
            if 'PSEUDOGLOSS' in df_h2s.columns:
                pseudo_glosses = df_h2s['PSEUDOGLOSS'].dropna().tolist()
        if pseudo_glosses:
            tokenizer = GlossTokenizer(pseudo_glosses, min_freq=2)
            print(f"  How2Sign pseudogloss tokenizer: {tokenizer.vocab_size} tokens")
        else:
            print("  How2Sign: no PSEUDOGLOSS column found — using dummy tokenizer")
            tokenizer = GlossTokenizer(["DUMMY"], min_freq=1)
    
    # ─── Datasets ───
    print(f"Loading {args.dataset} datasets...")
    
    if args.dataset == 'phoenix':
        train_ds = PhoenixSignDataset(args.root_dir, 'train', args.max_frames, True,
                                       tokenizer, bart_tokenizer)
        val_ds = PhoenixSignDataset(args.root_dir, 'dev', args.max_frames, False,
                                     tokenizer, bart_tokenizer)
        test_ds = PhoenixSignDataset(args.root_dir, 'test', args.max_frames, False,
                                      tokenizer, bart_tokenizer)
    else:
        train_ds = How2SignDataset(args.root_dir, 'train', args.max_frames, True,
                                    tokenizer, bart_tokenizer)
        val_ds = How2SignDataset(args.root_dir, 'val', args.max_frames, False,
                                  tokenizer, bart_tokenizer)
        test_ds = How2SignDataset(args.root_dir, 'test', args.max_frames, False,
                                   tokenizer, bart_tokenizer)
    
    sample = train_ds[0]
    input_dim = sample['keypoints'].shape[-1]
    args.input_dim_detected = input_dim
    print(f"  input_dim={input_dim}, vocab={tokenizer.vocab_size}, "
          f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers,
                             pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True)
    
    # ─── Model ───
    model = SignLanguageTransformer(
        input_dim=input_dim, dim=args.dim,
        num_classes=tokenizer.vocab_size,
        max_frames=args.max_frames * 2,
        dropout=args.dropout,
        use_bart=args.use_bart,
        bart_model=args.bart_model,
        ctc_weight=args.ctc_weight,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🤖 Model: {total_params:,} total, {trainable:,} trainable")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Resumed from {args.resume}")
    
    # ─── Train ───
    if not args.eval_only:
        trainer = Trainer(
            model, train_loader, val_loader, tokenizer, device,
            bart_tokenizer=bart_tokenizer,
            use_bart=args.use_bart,
            ctc_weight=args.ctc_weight,
            freeze_bart_epochs=args.freeze_bart_epochs,
            models_dir=models_dir,
        )
        print(f"\n🏋️ Training for {args.epochs} epochs...")
        trainer.train(num_epochs=args.epochs, decode_mode=args.decode,
                      beam_width=args.beam_width)
    
    # ─── Load best checkpoint for final evaluation ───
    best_ckpt = models_dir / 'best_model.pt'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"\n  📥 Loaded best checkpoint (epoch {ckpt.get('epoch','?')}, "
              f"val_loss={ckpt.get('val_loss',float('nan')):.4f})")

    # ─── Val (dev) evaluation ───
    print(f"\n📊 VAL (DEV) SET EVALUATION ({exp_name})")
    val_evaluator = SignLanguageEvaluator(
        model, val_loader, tokenizer, device,
        bart_tokenizer=bart_tokenizer,
        decode_mode=args.decode, beam_width=args.beam_width,
    )
    val_metrics, val_data = val_evaluator.evaluate(verbose=False)
    val_evaluator.print_metrics(val_metrics)
    wandb.log({f"val_final/{k}": v for k, v in val_metrics.items()})

    val_sample_rows = []
    for i in range(min(50, len(val_data['names']))):
        row = [val_data['names'][i], val_data['targets'][i], val_data['preds'][i]]
        if val_data['trans_preds'] and i < len(val_data['trans_preds']):
            row += [val_data['trans_targets'][i], val_data['trans_preds'][i]]
        val_sample_rows.append(row)
    val_cols = ['name', 'target_gloss', 'pred_gloss']
    if val_data['trans_preds']:
        val_cols += ['target_translation', 'pred_translation']
    wandb.log({'val_final/predictions': wandb.Table(columns=val_cols, data=val_sample_rows)})

    # ─── Test evaluation ───
    print(f"\n🎯 TEST SET EVALUATION ({exp_name})")
    test_evaluator = SignLanguageEvaluator(
        model, test_loader, tokenizer, device,
        bart_tokenizer=bart_tokenizer,
        decode_mode=args.decode, beam_width=args.beam_width,
    )
    test_metrics, test_data = test_evaluator.evaluate(verbose=True)
    test_evaluator.print_metrics(test_metrics)
    wandb.log({f"test_final/{k}": v for k, v in test_metrics.items()})

    test_sample_rows = []
    for i in range(min(50, len(test_data['names']))):
        row = [test_data['names'][i], test_data['targets'][i], test_data['preds'][i]]
        if test_data['trans_preds'] and i < len(test_data['trans_preds']):
            row += [test_data['trans_targets'][i], test_data['trans_preds'][i]]
        test_sample_rows.append(row)
    test_cols = ['name', 'target_gloss', 'pred_gloss']
    if test_data['trans_preds']:
        test_cols += ['target_translation', 'pred_translation']
    wandb.log({'test_final/predictions': wandb.Table(columns=test_cols, data=test_sample_rows)})

    # ─── Val vs Test summary table ───
    summary_cols = ['split', 'WER', 'BLEU-1', 'BLEU-4',
                    'trans_BLEU-1', 'trans_BLEU-4', 'exact_match']
    summary_table = wandb.Table(columns=summary_cols)
    for split_name, m in [('val', val_metrics), ('test', test_metrics)]:
        summary_table.add_data(
            split_name,
            round(m.get('WER', 1.0), 4),
            round(m.get('BLEU-1', 0), 4),
            round(m.get('BLEU-4', 0), 4),
            round(m.get('trans_BLEU-1', 0), 4),
            round(m.get('trans_BLEU-4', 0), 4),
            round(m.get('exact_match', 0), 4),
        )
    wandb.log({'results/val_vs_test': summary_table})
    plot_val_test_comparison(val_metrics, test_metrics)

    # ─── Beam sweep (val + test) ───
    run_beam_sweep(model, val_loader, test_loader, tokenizer, device,
                   bart_tokenizer, exp_name)

    # ─── Save experiment artefacts ───
    save_experiment(args, exp_name, results_dir, models_dir,
                    test_metrics, test_data, model, tokenizer)

    wandb.finish()
    
    # ─── Show test samples ───
    print(f"\n🔍 Sample predictions (test):")
    for i in range(min(10, len(test_data['names']))):
        correct = test_data['preds'][i].strip() == test_data['targets'][i].strip()
        mark = "✅" if correct else "❌"
        print(f"  {mark} Target: {test_data['targets'][i]}")
        print(f"     Pred:   {test_data['preds'][i]}")
        if test_data['trans_preds'] and i < len(test_data['trans_preds']):
            print(f"     Trans:  {test_data['trans_preds'][i]}")
        print()

    print(f"\n✨ Experiment '{exp_name}' complete!")
    return test_metrics


if __name__ == '__main__':
    main()