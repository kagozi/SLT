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
      --root_dir /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features \\
      --dim 256 --epochs 200 --max_frames 300 --decode beam --beam_width 10 \\
      --use_bart --ctc_weight 0.0 --freeze_bart_epochs 0

  # Exp 5: How2Sign Sign2Gloss2Text (joint CTC + BART with pseudo-glosses)
  python train.py --dataset how2sign --exp_name exp5_how2sign_sign2gloss2text \\
      --root_dir /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features \\
      --dim 256 --epochs 300 --max_frames 300 --decode beam --beam_width 10 \\
      --use_bart --ctc_weight 0.3 --freeze_bart_epochs 5

  # Exp 6: How2Sign Sign2Gloss (CTC-only, pseudo-glosses as labels)
  python train.py --dataset how2sign --exp_name exp6_how2sign_sign2gloss \\
      --root_dir /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features \\
      --dim 256 --epochs 200 --max_frames 300 --decode beam --beam_width 10
"""

import io
import re
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
from translation_utils import (
    resolve_translation_config,
    configure_tokenizer_for_target,
    normalize_translation_text,
    normalize_german_text,
    translation_keywords,
)


def _fig_to_wandb(fig) -> wandb.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as _PILImage
    return wandb.Image(_PILImage.open(buf).copy())

from dataset import PhoenixSignDataset
from dataset_how2sign import How2SignDataset
from dataset_youtube_asl import YouTubeASLDataset
from models import SignLanguageTransformer, MultiStreamSignLanguageTransformer, ConformerBlock
from utils import GlossTokenizer, BPETokenizer, Trainer, collate_fn, ctc_greedy_decode, ctc_beam_decode


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
                 bart_tokenizer=None, decode_mode='greedy', beam_width=5,
                 use_ctc=True, forced_bos_token_id=None, length_penalty=1.0):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.device = device
        self.decode_mode = decode_mode
        self.beam_width = beam_width
        self.use_ctc = use_ctc  # False when ctc_weight==0 (glossless mode)
        self.forced_bos_token_id = forced_bos_token_id
        self.length_penalty = float(length_penalty)

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

            all_names.extend(batch['name'])

            if self.use_ctc:
                logits_cpu = logits.cpu()
                input_lengths = mask_out.sum(dim=1).long().cpu()
                if not torch.isfinite(logits).all():
                    preds = [[] for _ in range(logits.size(0))]
                elif self.decode_mode == 'beam':
                    preds = ctc_beam_decode(logits_cpu, input_lengths, self.beam_width,
                                            length_penalty=self.length_penalty)
                else:
                    preds = ctc_greedy_decode(logits_cpu, input_lengths)
                for p in preds:
                    all_preds.append(self.tokenizer.decode(p))
                all_targets.extend(batch['gloss_text'])
            
            if self.model.use_bart and self.bart_tokenizer is not None:
                try:
                    token_ids = self.model.translate(keypoints, mask, beam_width=self.beam_width,
                                                        forced_bos_token_id=self.forced_bos_token_id)
                    for i in range(token_ids.shape[0]):
                        text = self.bart_tokenizer.decode(token_ids[i], skip_special_tokens=True)
                        all_trans_preds.append(text)
                    all_trans_targets.extend(batch['translation'])
                except Exception as e:
                    if verbose:
                        print(f"  Translation failed: {e}")

            if verbose and batch_idx % 10 == 0:
                print(f"  Eval batch {batch_idx}/{len(self.test_loader)}")

        metrics = self._compute_metrics(all_preds, all_targets, inference_times,
                                        use_ctc=self.use_ctc)
        if all_trans_preds:
            trans_bleu = self._compute_bleu(all_trans_preds, all_trans_targets)
            metrics['trans_BLEU-1'] = trans_bleu.get('BLEU-1', 0)
            metrics['trans_BLEU-4'] = trans_bleu.get('BLEU-4', 0)
            metrics['trans_BLEU'] = trans_bleu.get('BLEU', 0)
            trans_rouge = self._compute_rouge(all_trans_preds, all_trans_targets)
            metrics['trans_ROUGE-L'] = trans_rouge.get('ROUGE-L', 0)
            trans_meteor = self._compute_meteor(all_trans_preds, all_trans_targets)
            metrics['trans_METEOR'] = trans_meteor.get('METEOR', 0)

        return metrics, {
            'preds': all_preds, 'targets': all_targets, 'names': all_names,
            'trans_preds': all_trans_preds, 'trans_targets': all_trans_targets,
        }

    def _compute_metrics(self, preds, targets, times, use_ctc=True):
        metrics = {
            'total_samples': len(preds) if use_ctc else len(times),
            'avg_inference_ms': float(np.mean(times) * 1000),
        }
        if use_ctc and preds:
            metrics['WER'] = float(self._compute_wer(preds, targets))
            bleu = self._compute_bleu(preds, targets)
            metrics.update(bleu)
            metrics.update(self._compute_rouge(preds, targets))
            metrics.update(self._compute_meteor(preds, targets))
            exact = sum(1 for p, t in zip(preds, targets) if p.strip() == t.strip())
            metrics['exact_match'] = exact / len(preds)
        else:
            # Glossless mode: CTC metrics are not meaningful — only translation metrics matter
            metrics['WER'] = None
            metrics['exact_match'] = None
        return metrics

    def _compute_wer(self, preds, targets):
        def wer_single(p, t):
            pw, tw = p.split(), t.split()
            if not tw: return 0.0
            d = np.zeros((len(tw)+1, len(pw)+1), dtype=np.float32)
            for i in range(len(tw)+1): d[i,0] = i
            for j in range(len(pw)+1): d[0,j] = j
            for i in range(1, len(tw)+1):
                for j in range(1, len(pw)+1):
                    cost = 0 if tw[i-1] == pw[j-1] else 1
                    d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+cost)
            return d[len(tw), len(pw)] / len(tw)
        return float(np.mean([wer_single(p, t) for p, t in zip(preds, targets)]))

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, remove punctuation, collapse whitespace."""
        text = text.lower()
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _compute_bleu(self, preds, targets, max_n=4):
        """sacrebleu corpus BLEU with standard 13a tokenization + lowercase.
        Returns scores as percentages (sacrebleu native scale, e.g. 17.3)."""
        from sacrebleu.metrics import BLEU
        bleu = BLEU(tokenize='13a', lowercase=True)
        result = bleu.corpus_score(preds, [targets])
        scores = {
            'BLEU-1': result.precisions[0],
            'BLEU-2': result.precisions[1],
            'BLEU-4': result.precisions[3],
            'BLEU':   result.score,
        }
        return scores

    @staticmethod
    def _compute_rouge(preds, targets):
        """Corpus-average ROUGE-1 / ROUGE-2 / ROUGE-L (F1, ×100). Returns {} on failure."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            r1, r2, rl = [], [], []
            for p, t in zip(preds, targets):
                s = scorer.score(t, p)
                r1.append(s['rouge1'].fmeasure)
                r2.append(s['rouge2'].fmeasure)
                rl.append(s['rougeL'].fmeasure)
            return {
                'ROUGE-1': float(np.mean(r1)) * 100,
                'ROUGE-2': float(np.mean(r2)) * 100,
                'ROUGE-L': float(np.mean(rl)) * 100,
            }
        except Exception:
            return {}

    @staticmethod
    def _compute_meteor(preds, targets):
        """Corpus-average METEOR (×100). Returns {} on failure."""
        try:
            import nltk
            nltk.download('punkt_tab', quiet=True)
            nltk.download('wordnet', quiet=True)
            from nltk.translate.meteor_score import meteor_score as _ms
            scores = [_ms([ref.split()], hyp.split()) for hyp, ref in zip(preds, targets)]
            return {'METEOR': float(np.mean(scores)) * 100}
        except Exception:
            return {}

    def print_metrics(self, metrics):
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        print(f"  Decode:  {self.decode_mode} (beam={self.beam_width})")
        print(f"  Samples: {metrics['total_samples']}")
        if metrics.get('WER') is not None:
            print(f"  WER:     {metrics['WER']:.2%}")
            print(f"  BLEU-1:  {metrics.get('BLEU-1', 0):.4f}")
            print(f"  BLEU-4:  {metrics.get('BLEU-4', 0):.4f}")
            if 'ROUGE-L' in metrics:
                print(f"  ROUGE-1: {metrics.get('ROUGE-1', 0):.4f}")
                print(f"  ROUGE-2: {metrics.get('ROUGE-2', 0):.4f}")
                print(f"  ROUGE-L: {metrics.get('ROUGE-L', 0):.4f}")
            if 'METEOR' in metrics:
                print(f"  METEOR:  {metrics.get('METEOR', 0):.4f}")
            print(f"  Exact:   {metrics.get('exact_match', 0):.2%}")
        else:
            print(f"  WER/BLEU: N/A (glossless mode — CTC not used)")
        if 'trans_BLEU-1' in metrics:
            print(f"  --- Translation ---")
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
    n_rows = len(eval_data['names'])
    df = pd.DataFrame({'name': eval_data['names']})

    gloss_targets = list(eval_data['targets'])
    gloss_preds = list(eval_data['preds'])
    if len(gloss_targets) == n_rows and len(gloss_preds) == n_rows:
        df['target_gloss'] = gloss_targets
        df['predicted_gloss'] = gloss_preds
        df['correct'] = [
            p.strip() == t.strip()
            for p, t in zip(gloss_preds, gloss_targets)
        ]
    else:
        df['target_gloss'] = [''] * n_rows
        df['predicted_gloss'] = [''] * n_rows
        df['correct'] = [False] * n_rows

    if eval_data['trans_preds'] or eval_data['trans_targets']:
        tp = list(eval_data['trans_preds'])
        tt = list(eval_data['trans_targets'])
        while len(tp) < n_rows:
            tp.append('')
        while len(tt) < n_rows:
            tt.append('')
        df['target_translation'] = tt[:n_rows]
        df['predicted_translation'] = tp[:n_rows]
    preds_path = results_dir / f"predictions_{exp_name}.csv"
    df.to_csv(preds_path, index=False)
    
    # ── Final Model ──
    model_path = models_dir / f"final_{exp_name}.pt"
    torch.save({
        'model_state_dict': getattr(model, 'module', model).state_dict(),
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
    
    # ── Save BPE tokenizer if used (needed for fine-tuning transfer) ──
    if isinstance(tokenizer, BPETokenizer):
        bpe_path = models_dir / f"bpe_tokenizer_{exp_name}.json"
        tokenizer.save(bpe_path)
        print(f"   🔤 {bpe_path}")

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
    has_trans = any(not math.isnan(r['trans_bleu4']) for r in sweep_rows)

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
    val_vals = [0 if v is None else v for v in val_vals]
    test_vals = [0 if v is None else v for v in test_vals]

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
                   bart_tokenizer, exp_name, use_ctc=True, forced_bos_token_id=None,
                   length_penalty=1.0):
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
                use_ctc=use_ctc, forced_bos_token_id=forced_bos_token_id,
                length_penalty=length_penalty,
            )
            m, _ = ev.evaluate(verbose=False)
            row = dict(
                split=split_name, beam_width=bw,
                WER=round(m['WER'] if m.get('WER') is not None else float('nan'), 4),
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
                        choices=['phoenix', 'how2sign', 'youtube_asl'],
                        help='Dataset to use: phoenix, how2sign, or youtube_asl')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Dataset root (auto-set per dataset if not provided)')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Max frames (default: 250 for phoenix, 300 for how2sign)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--unglossed_ctc_target', type=str, default='pseudogloss',
                        choices=['pseudogloss', 'translation', 'translation_keywords'],
                        help='CTC target for datasets without true glosses. '
                             'pseudogloss uses PSEUDOGLOSS, translation uses normalized '
                             'English text, translation_keywords uses content-word English.')
    parser.add_argument('--unglossed_min_freq', type=int, default=2,
                        help='Minimum token frequency for unglossed CTC vocabularies.')
    parser.add_argument('--ctc_tokenizer', type=str, default='word',
                        choices=['word', 'bpe'],
                        help='CTC tokenizer type. word=word-level (default); '
                             'bpe=BPE subword (recommended for open-domain datasets like How2Sign).')
    parser.add_argument('--bpe_vocab_size', type=int, default=500,
                        help='BPE vocabulary size when --ctc_tokenizer bpe is used.')
    parser.add_argument('--bpe_tokenizer_path', type=str, default=None,
                        help='Path to a saved BPE tokenizer JSON (from a prior run). '
                             'When set, skips training a new BPE model and reuses the saved one. '
                             'Required for fine-tuning on a new dataset with the same vocab.')
    parser.add_argument('--phoenix_ctc_target', type=str, default='gloss',
                        choices=['gloss', 'translation'],
                        help='CTC target for PHOENIX dataset: gloss uses ground-truth gloss '
                             'annotations, translation uses normalized German text tokens '
                             '(Maia et al. 2025 glossless CTC paradigm).')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    
    # Decoding
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_width', type=int, default=10)
    parser.add_argument('--length_penalty', type=float, default=1.0,
                        help='Exponent for length normalization in CTC beam search. '
                             '>1.0 penalizes longer outputs (improves WER when over-generating).')
    
    # BART translation
    parser.add_argument('--use_bart', action='store_true', help='Enable BART Gloss→Text')
    parser.add_argument('--bart_model', type=str, default='auto')
    parser.add_argument('--ctc_weight', type=float, default=0.3,
                        help='CTC loss weight (1.0=CTC only, 0.0=BART only)')
    parser.add_argument('--freeze_bart_epochs', type=int, default=5,
                        help='Epochs to freeze BART before joint training')
    parser.add_argument('--warmup_bart_epochs', type=int, default=None,
                        help='Epochs for detached decoder warmup before full joint training. '
                             'Defaults to freeze_bart_epochs.')
    parser.add_argument('--forced_bos_token_id', type=int, default=None,
                        help='Force decoder to start with this token (e.g. mBART language id for German)')
    parser.add_argument('--ffn_expand', type=int, default=4,
                        help='FFN hidden expansion factor in transformer blocks (default 4)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing for translation cross-entropy (default 0.1)')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps (effective_batch = batch_size * grad_accum)')
    parser.add_argument('--fp16', action='store_true',
                        help='Enable mixed-precision (fp16) training via PyTorch AMP')
    
    # Experiment management
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Custom experiment name (auto-generated if not set)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Base dir for results/ and models/ (default: ..)')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')

    # Transfer learning
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained checkpoint (.pt) to initialise encoder from')
    parser.add_argument('--freeze_strategy', type=str, default='none',
                        choices=['none', 'convblocks', 'full'],
                        help='Encoder freeze strategy for transfer learning: '
                             'none=full fine-tune, convblocks=freeze conv only, '
                             'full=freeze entire encoder (train head only)')
    parser.add_argument('--subset_pct', type=float, default=100.0,
                        help='Percentage of training data to use (1-100). '
                             'Used for low-data adaptation experiments.')

    # ── Enhanced training (Phase 1-3 improvements) ───────────────────
    parser.add_argument('--use_motion', action='store_true',
                        help='Append frame-difference (velocity) features to keypoints '
                             'before input projection. Doubles input_dim. '
                             'Ignored for --multistream (motion always on per-stream).')
    parser.add_argument('--multistream', action='store_true',
                        help='Use MultiStreamSignLanguageTransformer: separate '
                             'pose/hand/face streams with cross-attention fusion. '
                             'Always includes per-stream velocity features.')
    parser.add_argument('--contrastive_weight', type=float, default=0.0,
                        help='Weight for NT-Xent contrastive loss on encoder '
                             'representations (two augmented views per batch). '
                             '0=disabled. Recommended: 0.1')
    parser.add_argument('--rdrop_weight', type=float, default=0.0,
                        help='Weight for R-Drop symmetric KL regularisation on '
                             'CTC logits (two forward passes per batch). '
                             '0=disabled. Recommended: 5.0 for CTC+BART exps '
                             '(Liang et al., NeurIPS 2021).')
    parser.add_argument('--arch', type=str, default='sequential',
                        choices=['sequential', 'interleaved'],
                        help='Encoder block layout. sequential (default): 3 ConvBlocks then '
                             '4 TransformerBlocks. interleaved: 6 paired ConformerBlocks '
                             '(Conv+Transformer alternating kernels 11/5) — replicates Maia '
                             'et al. ASL2Text encoder. For --multistream, each stream gets 3 '
                             'ConformerBlocks then shared 4 TransformerBlocks after fusion.')
    parser.add_argument('--ctc_smoothing', type=float, default=0.1,
                        help='Uniform label smoothing on CTC loss (default 0.1). '
                            'Set 0.0 to disable.')
    parser.add_argument('--joint_encoder_lr_scale', type=float, default=0.2,
                        help='LR multiplier for encoder params during full joint stage '
                             'to reduce catastrophic forgetting.')
    parser.add_argument('--joint_bart_lr_scale', type=float, default=0.5,
                        help='LR multiplier for seq2seq params during full joint stage.')
    parser.add_argument('--early_stop_patience', type=int, default=0,
                        help='Stop training after this many epochs without validation '
                             'score improvement. 0 disables early stopping.')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=0,
                        help='Freeze the encoder for this many epochs at the start of '
                             'training (CTC head warms up alone). After this many epochs '
                             'the encoder is added to the optimizer at a lower LR '
                             '(--unfreeze_encoder_lr_scale). 0 disables (default). '
                             'Only applies to CTC-only mode (no --use_bart).')
    parser.add_argument('--unfreeze_encoder_lr_scale', type=float, default=0.1,
                        help='LR multiplier for the encoder when it is unfrozen relative '
                             'to --lr. Default 0.1 gives 10x lower LR to avoid catastrophic '
                             'forgetting of pretrained encoder representations.')

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # ─── Dataset defaults ───
    if args.root_dir is None:
        if args.dataset == 'phoenix':
            args.root_dir = '/data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T'
        elif args.dataset == 'youtube_asl':
            args.root_dir = '/data/youtube_asl'
        else:
            args.root_dir = '/data/hf_cache/How2Sign_Holistic/how2sign_holistic_features'

    if args.max_frames is None:
        args.max_frames = 250 if args.dataset == 'phoenix' else 300
    
    # ─── Experiment Setup ───
    exp_name = get_experiment_name(args)
    results_dir, models_dir = setup_directories(exp_name, args.output_dir)

    import os
    # Tell W&B to delete local media files after they are uploaded, preventing
    # the wandb/run-*/files/media/ tree from accumulating on the PVC.
    os.environ.setdefault('WANDB_DELETE_LOCAL', '1')

    wandb.init(
        project="slt",
        name=exp_name,
        config={
            **vars(args),
            'is_transfer': args.pretrained_path is not None,
            'freeze_strategy': args.freeze_strategy,
            'subset_pct': args.subset_pct,
        },
        dir=str(results_dir),
        # resume="allow" reuses an existing run dir on restart instead of
        # creating a new one, keeping local wandb/ storage bounded.
        resume="allow",
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
        from transformers import AutoTokenizer
        translation_cfg = resolve_translation_config(args.dataset, args.bart_model)
        args.bart_model = translation_cfg['bart_model']
        bart_tokenizer = AutoTokenizer.from_pretrained(args.bart_model)
        auto_bos = configure_tokenizer_for_target(bart_tokenizer, translation_cfg['target_lang'])
        if args.forced_bos_token_id is None and auto_bos is not None:
            args.forced_bos_token_id = auto_bos
        print(f"  Seq2Seq tokenizer: {args.bart_model}")
        if translation_cfg['target_lang'] is not None:
            print(f"  Target language: {translation_cfg['target_lang']}")
        if args.forced_bos_token_id is not None:
            print(f"  forced_bos_token_id={args.forced_bos_token_id}")
    
    if args.dataset == 'phoenix':
        # Build CTC tokenizer from PHOENIX annotations
        all_gloss = []
        for split in ['train', 'dev', 'test']:
            csv = Path(args.root_dir) / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
            df = pd.read_csv(csv, sep='|')
            if args.phoenix_ctc_target == 'translation':
                all_gloss.extend(df['translation'].dropna().map(normalize_german_text).tolist())
            else:
                all_gloss.extend(df['orth'].dropna().tolist())
        if args.ctc_tokenizer == 'bpe':
            tokenizer = BPETokenizer(all_gloss, vocab_size=args.bpe_vocab_size)
            print(f"  PHOENIX {args.phoenix_ctc_target} BPE tokenizer: {tokenizer.vocab_size} tokens")
        else:
            tokenizer = GlossTokenizer(all_gloss, min_freq=1)
            print(f"  PHOENIX {args.phoenix_ctc_target} CTC tokenizer: {tokenizer.vocab_size} tokens")
    elif args.dataset == 'youtube_asl':
        # YouTube-ASL has no true glosses. Build CTC vocab from the selected
        # weak target, defaulting to legacy PSEUDOGLOSS for compatibility.
        meta_dir = Path(args.root_dir) / 'metadata'
        pseudo_csv = meta_dir / 'youtube_asl_train.csv'
        ctc_texts = []
        if pseudo_csv.exists():
            df_yt = pd.read_csv(pseudo_csv, sep='\t')
            df_yt.columns = [c.strip() for c in df_yt.columns]
            if args.unglossed_ctc_target == 'translation':
                ctc_texts = df_yt['SENTENCE'].fillna('').map(normalize_translation_text).tolist()
            elif args.unglossed_ctc_target == 'translation_keywords':
                ctc_texts = df_yt['SENTENCE'].fillna('').map(translation_keywords).tolist()
            elif 'PSEUDOGLOSS' in df_yt.columns:
                ctc_texts = df_yt['PSEUDOGLOSS'].dropna().tolist()
        if ctc_texts:
            if args.bpe_tokenizer_path:
                tokenizer = BPETokenizer.load(args.bpe_tokenizer_path)
            elif args.ctc_tokenizer == 'bpe':
                tokenizer = BPETokenizer(ctc_texts, vocab_size=args.bpe_vocab_size)
            else:
                tokenizer = GlossTokenizer(ctc_texts, min_freq=args.unglossed_min_freq)
            print(f"  YouTubeASL {args.unglossed_ctc_target} CTC tokenizer: {tokenizer.vocab_size} tokens")
        else:
            print("  YouTubeASL: no CTC targets — using dummy tokenizer (glossless mode)")
            tokenizer = GlossTokenizer(["DUMMY"], min_freq=1)
    else:
        # How2Sign has no true glosses. Build CTC vocab from the selected weak
        # target. translation/translation_keywords evaluate against available
        # English supervision instead of POS pseudo-gloss artifacts.
        # CSVs live in metadata/ inside the HF cache root.
        meta_dir = Path(args.root_dir) / 'metadata'
        pseudo_csv = meta_dir / 'how2sign_realigned_train.csv'
        ctc_texts = []
        if pseudo_csv.exists():
            df_h2s = pd.read_csv(pseudo_csv, sep='\t')
            df_h2s.columns = [c.strip() for c in df_h2s.columns]
            if args.unglossed_ctc_target == 'translation':
                ctc_texts = df_h2s['SENTENCE'].fillna('').map(normalize_translation_text).tolist()
            elif args.unglossed_ctc_target == 'translation_keywords':
                ctc_texts = df_h2s['SENTENCE'].fillna('').map(translation_keywords).tolist()
            elif 'PSEUDOGLOSS' in df_h2s.columns:
                ctc_texts = df_h2s['PSEUDOGLOSS'].dropna().tolist()
        if ctc_texts:
            if args.bpe_tokenizer_path:
                tokenizer = BPETokenizer.load(args.bpe_tokenizer_path)
            elif args.ctc_tokenizer == 'bpe':
                tokenizer = BPETokenizer(ctc_texts, vocab_size=args.bpe_vocab_size)
            else:
                tokenizer = GlossTokenizer(ctc_texts, min_freq=args.unglossed_min_freq)
            print(f"  How2Sign {args.unglossed_ctc_target} CTC tokenizer: {tokenizer.vocab_size} tokens")
        else:
            print("  How2Sign: no CTC targets — using dummy tokenizer (glossless mode)")
            tokenizer = GlossTokenizer(["DUMMY"], min_freq=1)
    
    # ─── Datasets ───
    print(f"Loading {args.dataset} datasets...")
    
    if args.dataset == 'phoenix':
        train_ds = PhoenixSignDataset(args.root_dir, 'train', args.max_frames, True,
                                       tokenizer, bart_tokenizer,
                                       ctc_target=args.phoenix_ctc_target)
        val_ds = PhoenixSignDataset(args.root_dir, 'dev', args.max_frames, False,
                                     tokenizer, bart_tokenizer,
                                     ctc_target=args.phoenix_ctc_target)
        test_ds = PhoenixSignDataset(args.root_dir, 'test', args.max_frames, False,
                                      tokenizer, bart_tokenizer,
                                      ctc_target=args.phoenix_ctc_target)
    elif args.dataset == 'youtube_asl':
        train_ds = YouTubeASLDataset(args.root_dir, 'train', args.max_frames, True,
                                      tokenizer, bart_tokenizer,
                                      ctc_target=args.unglossed_ctc_target)
        val_ds = YouTubeASLDataset(args.root_dir, 'val', args.max_frames, False,
                                    tokenizer, bart_tokenizer,
                                    ctc_target=args.unglossed_ctc_target)
        test_ds = YouTubeASLDataset(args.root_dir, 'test', args.max_frames, False,
                                     tokenizer, bart_tokenizer,
                                     ctc_target=args.unglossed_ctc_target)
    else:
        train_ds = How2SignDataset(args.root_dir, 'train', args.max_frames, True,
                                    tokenizer, bart_tokenizer,
                                    ctc_target=args.unglossed_ctc_target)
        val_ds = How2SignDataset(args.root_dir, 'val', args.max_frames, False,
                                  tokenizer, bart_tokenizer,
                                  ctc_target=args.unglossed_ctc_target)
        test_ds = How2SignDataset(args.root_dir, 'test', args.max_frames, False,
                                   tokenizer, bart_tokenizer,
                                   ctc_target=args.unglossed_ctc_target)
    
    # ── Low-data subset sampling ──
    if args.subset_pct < 100.0:
        n_keep = max(1, int(len(train_ds) * args.subset_pct / 100.0))
        indices = random.sample(range(len(train_ds)), n_keep)
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, indices)
        print(f"  ⚡ Low-data mode: using {n_keep}/{len(train_ds)} training samples "
              f"({args.subset_pct:.0f}%)")

    sample = (train_ds[0] if not hasattr(train_ds, 'dataset')
              else train_ds.dataset[train_ds.indices[0]])
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
    _model_kwargs = dict(
        input_dim=input_dim, dim=args.dim,
        num_classes=tokenizer.vocab_size,
        max_frames=args.max_frames * 2,
        dropout=args.dropout,
        use_bart=args.use_bart,
        bart_model=args.bart_model,
        ctc_weight=args.ctc_weight,
        ffn_expand=args.ffn_expand,
        label_smoothing=args.label_smoothing,
        arch=args.arch,
    )
    if args.multistream:
        # Multi-stream encoder always computes per-stream velocity; use_motion ignored
        model = MultiStreamSignLanguageTransformer(**_model_kwargs)
        print(f"  🔀 Encoder: MultiStream (pose/lhand/rhand/face + per-stream velocity) [{args.arch}]")
    else:
        model = SignLanguageTransformer(use_motion=args.use_motion, **_model_kwargs)
        motion_tag = " + velocity" if args.use_motion else ""
        print(f"  🔀 Encoder: Flat 225-d{motion_tag} [{args.arch}]")
    
    # ── Load pretrained encoder (transfer learning) ──
    if args.pretrained_path and args.pretrained_path.lower() != 'none':
        ckpt_path = Path(args.pretrained_path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu')
            src = ckpt.get('model_state_dict', ckpt)
            # Load only shared encoder weights; ignore head (vocab mismatch across datasets)
            encoder_keys = {k: v for k, v in src.items()
                            if not k.startswith('head.')
                            and not k.startswith('translation_head.')
                            and k != 'pos_encoding.pe'}  # pe is a fixed sinusoidal buffer; drop to allow max_frames mismatch
            missing, unexpected = model.load_state_dict(encoder_keys, strict=False)
            print(f"  📥 Pretrained encoder loaded from {ckpt_path.name}")
            print(f"     Loaded {len(encoder_keys)} keys | "
                  f"missing={len(missing)} | unexpected={len(unexpected)}")
        else:
            print(f"  ⚠️  Pretrained path not found: {args.pretrained_path} — training from scratch")

    # ── Apply freeze strategy ──
    if args.freeze_strategy == 'full':
        model.freeze_encoder()
        print("  🔒 Freeze: FULL — only CTC head is trainable")
    elif args.freeze_strategy == 'convblocks':
        model.freeze_convblocks()
        print("  🔒 Freeze: CONVBLOCKS — transformer blocks + head are trainable")
    else:
        print("  🔓 Freeze: NONE — full fine-tuning")

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen       = total_params - trainable
    print(f"🤖 Model: {total_params:,} total | {trainable:,} trainable | {frozen:,} frozen")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = torch.cuda.device_count() if device == 'cuda' else 0
    print(f"💻 Device: {device} ({n_gpus} GPU(s))")
    model = model.to(device)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"   Using DataParallel across {n_gpus} GPUs")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        getattr(model, 'module', model).load_state_dict(ckpt['model_state_dict'])
        print(f"  Resumed from {args.resume}")
    
    # ─── Train ───
    if not args.eval_only:
        trainer = Trainer(
            model, train_loader, val_loader, tokenizer, device,
            bart_tokenizer=bart_tokenizer,
            use_bart=args.use_bart,
            ctc_weight=args.ctc_weight,
            freeze_bart_epochs=args.freeze_bart_epochs,
            warmup_bart_epochs=args.warmup_bart_epochs,
            models_dir=models_dir,
            grad_accum=args.grad_accum,
            fp16=args.fp16,
            contrastive_weight=args.contrastive_weight,
            rdrop_weight=args.rdrop_weight,
            ctc_smoothing=args.ctc_smoothing,
            total_epochs=args.epochs,
            base_lr=args.lr,
            bart_lr=args.lr * 0.05,
            joint_encoder_lr_scale=args.joint_encoder_lr_scale,
            joint_bart_lr_scale=args.joint_bart_lr_scale,
            early_stop_patience=args.early_stop_patience,
            length_penalty=args.length_penalty,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
            unfreeze_encoder_lr_scale=args.unfreeze_encoder_lr_scale,
        )
        print(f"\n🏋️ Training for {args.epochs} epochs...")
        trainer.train(num_epochs=args.epochs, decode_mode=args.decode,
                      beam_width=args.beam_width)
    
    # ─── Load best checkpoint for final evaluation ───
    best_ckpt = models_dir / 'best_model.pt'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        getattr(model, 'module', model).load_state_dict(ckpt['model_state_dict'])
        print(f"\n  📥 Loaded best checkpoint (epoch {ckpt.get('epoch','?')}, "
              f"val_loss={ckpt.get('val_loss',float('nan')):.4f})")

    use_ctc = args.ctc_weight > 0

    # ─── Val (dev) evaluation ───
    print(f"\n📊 VAL (DEV) SET EVALUATION ({exp_name})")
    val_evaluator = SignLanguageEvaluator(
        model, val_loader, tokenizer, device,
        bart_tokenizer=bart_tokenizer,
        decode_mode=args.decode, beam_width=args.beam_width,
        use_ctc=use_ctc, forced_bos_token_id=args.forced_bos_token_id,
        length_penalty=args.length_penalty,
    )
    val_metrics, val_data = val_evaluator.evaluate(verbose=False)
    val_evaluator.print_metrics(val_metrics)
    wandb.log({f"val_final/{k}": v for k, v in val_metrics.items()})

    val_sample_rows = []
    for i in range(min(50, len(val_data['names']))):
        target_gloss = val_data['targets'][i] if i < len(val_data['targets']) else ''
        pred_gloss = val_data['preds'][i] if i < len(val_data['preds']) else ''
        row = [val_data['names'][i], target_gloss, pred_gloss]
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
        use_ctc=use_ctc, forced_bos_token_id=args.forced_bos_token_id,
        length_penalty=args.length_penalty,
    )
    test_metrics, test_data = test_evaluator.evaluate(verbose=True)
    test_evaluator.print_metrics(test_metrics)
    wandb.log({f"test_final/{k}": v for k, v in test_metrics.items()})

    test_sample_rows = []
    for i in range(min(50, len(test_data['names']))):
        target_gloss = test_data['targets'][i] if i < len(test_data['targets']) else ''
        pred_gloss = test_data['preds'][i] if i < len(test_data['preds']) else ''
        row = [test_data['names'][i], target_gloss, pred_gloss]
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
            round(m['WER'], 4) if m.get('WER') is not None else float('nan'),
            round(m.get('BLEU-1', 0), 4),
            round(m.get('BLEU-4', 0), 4),
            round(m.get('trans_BLEU-1', 0), 4),
            round(m.get('trans_BLEU-4', 0), 4),
            round(m.get('exact_match', 0), 4) if m.get('exact_match') is not None else float('nan'),
        )
    wandb.log({'results/val_vs_test': summary_table})
    plot_val_test_comparison(val_metrics, test_metrics)

    # ─── Beam sweep (val + test) ───
    run_beam_sweep(model, val_loader, test_loader, tokenizer, device,
                   bart_tokenizer, exp_name, use_ctc=use_ctc,
                   forced_bos_token_id=args.forced_bos_token_id,
                   length_penalty=args.length_penalty)

    # ─── Save experiment artefacts ───
    save_experiment(args, exp_name, results_dir, models_dir,
                    test_metrics, test_data, model, tokenizer)

    wandb.finish()
    
    # ─── Show test samples ───
    print(f"\n🔍 Sample predictions (test):")
    for i in range(min(10, len(test_data['names']))):
        target_gloss = test_data['targets'][i] if i < len(test_data['targets']) else ''
        pred_gloss = test_data['preds'][i] if i < len(test_data['preds']) else ''
        correct = bool(pred_gloss.strip()) and pred_gloss.strip() == target_gloss.strip()
        mark = "✅" if correct else "❌"
        if target_gloss or pred_gloss:
            print(f"  {mark} Target: {target_gloss}")
            print(f"     Pred:   {pred_gloss}")
        else:
            print(f"  {mark} Gloss:  N/A (translation-only evaluation)")
        if test_data['trans_preds'] and i < len(test_data['trans_preds']):
            print(f"     Trans:  {test_data['trans_preds'][i]}")
        print()

    print(f"\n✨ Experiment '{exp_name}' complete!")
    return test_metrics


if __name__ == '__main__':
    main()
