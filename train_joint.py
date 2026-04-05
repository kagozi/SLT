"""
Joint PHOENIX + How2Sign training with language-aware encoder.

Trains a LanguageAwareSignTransformer on both datasets simultaneously by
alternating batches.  A learned language embedding (dim-d vector) is added
to the input features so the shared encoder knows which sign language it is
processing.  Separate CTC heads handle the different gloss vocabularies.

This experiment tests whether the encoder learns language-agnostic
gesture representations — confirmed by UMAP visualisation (extract_features.py).

Architecture:
    x (225) → input_proj → LayerNorm → pos_encoding
            → + lang_embedding(0=DGS or 1=ASL)
            → conv_blocks (shared)
            → transformer_blocks (shared)
            → head_phoenix  (DGS batches)  or
              head_how2sign (ASL batches)

Usage (NRP):
    kubectl apply -f nautilius/train-exp11-joint-training.yaml

Usage (local):
    python train_joint.py \\
        --phoenix_dir /data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
        --how2sign_dir /data/how2sign_hf \\
        --output_dir /data/experiments \\
        --exp_name exp11_joint_training \\
        --epochs 150 --dim 256 --batch_size 16
"""

import io
import argparse
import math
import random
import time
import json
from collections import Counter
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import wandb

from dataset import PhoenixSignDataset
from dataset_how2sign import How2SignDataset
from models import LanguageAwareSignTransformer
from utils import GlossTokenizer, CTCLoss, collate_fn, ctc_greedy_decode, ctc_beam_decode


# ── Language IDs ─────────────────────────────────────────────────────
LANG_PHOENIX  = 0   # DGS
LANG_HOW2SIGN = 1   # ASL


# ── Collate with language ID ──────────────────────────────────────────

def collate_fn_with_lang(batch, lang_id: int):
    """Wrap collate_fn and add a lang_id tensor to each batch."""
    result = collate_fn(batch)
    B = result['keypoints'].shape[0]
    result['lang_id'] = torch.full((B,), lang_id, dtype=torch.long)
    return result


# ── Figure helper ────────────────────────────────────────────────────

def _fig_to_wandb(fig):
    from PIL import Image as _PIL
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return wandb.Image(_PIL.open(buf).copy())


# ── Joint Trainer ─────────────────────────────────────────────────────

class JointTrainer:
    """
    Alternates Phoenix and How2Sign batches.  CTC loss is computed on each
    mini-batch using the appropriate head.  The encoder is trained jointly.
    """

    def __init__(self, model, phoenix_train, how2sign_train,
                 phoenix_val, how2sign_val,
                 phoenix_tokenizer, how2sign_tokenizer,
                 device='cuda', batch_size=16, num_workers=4,
                 models_dir='.', lr=1e-3):
        self.model = model.to(device)
        self.device = device
        self.phoenix_tok  = phoenix_tokenizer
        self.how2sign_tok = how2sign_tokenizer
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        _ph_col  = lambda b: collate_fn_with_lang(b, LANG_PHOENIX)
        _h2s_col = lambda b: collate_fn_with_lang(b, LANG_HOW2SIGN)

        self.ph_train_loader = DataLoader(
            phoenix_train, batch_size, shuffle=True,
            collate_fn=_ph_col, num_workers=num_workers, pin_memory=True)
        self.h2s_train_loader = DataLoader(
            how2sign_train, batch_size, shuffle=True,
            collate_fn=_h2s_col, num_workers=num_workers, pin_memory=True)
        self.ph_val_loader = DataLoader(
            phoenix_val, batch_size, shuffle=False,
            collate_fn=_ph_col, num_workers=num_workers, pin_memory=True)
        self.h2s_val_loader = DataLoader(
            how2sign_val, batch_size, shuffle=False,
            collate_fn=_h2s_col, num_workers=num_workers, pin_memory=True)

        self.base_lr = lr
        self.criterion = CTCLoss(blank=0)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-4)

        steps_per_epoch = len(self.ph_train_loader) + len(self.h2s_train_loader)
        self.total_steps = steps_per_epoch * 150
        self.warmup_steps = steps_per_epoch * 5
        self.best_val_loss = float('inf')
        self.global_step = 0

        self.history = {
            'epoch': [], 'train_loss': [], 'train_loss_ph': [], 'train_loss_h2s': [],
            'val_loss_ph': [], 'val_loss_h2s': [],
            'val_wer_ph': [], 'val_wer_h2s': [],
        }

    def _lr(self):
        s = self.global_step
        if s < self.warmup_steps:
            factor = 2 ** -(self.warmup_steps - s)
        else:
            p = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * p))
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.base_lr * factor

    def _ctc_step(self, batch):
        """Forward + CTC loss for one batch (either language)."""
        kps    = batch['keypoints'].to(self.device)
        mask   = batch['mask'].to(self.device)
        tgts   = batch['gloss'].to(self.device)
        lang   = batch['lang_id'].to(self.device)

        logits, mask_out = self.model(kps, mask, lang_id=lang)

        logits_ctc  = logits.permute(1, 0, 2)
        input_lens  = mask_out.sum(1).long().cpu()
        target_lens = (tgts != 0).sum(1).long().cpu()

        if (target_lens == 0).any() or (input_lens < target_lens).any():
            return None
        loss = self.criterion(logits_ctc, tgts, input_lens, target_lens)
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        return loss

    def train_epoch(self, epoch):
        self.model.train()
        ph_iter  = iter(self.ph_train_loader)
        h2s_iter = iter(self.h2s_train_loader)

        total_loss = 0.0
        ph_loss_sum = 0.0
        h2s_loss_sum = 0.0
        n_ph = 0
        n_h2s = 0

        # Interleave Phoenix and How2Sign batches
        while True:
            # Phoenix batch
            try:
                ph_batch = next(ph_iter)
            except StopIteration:
                ph_batch = None
            # How2Sign batch
            try:
                h2s_batch = next(h2s_iter)
            except StopIteration:
                h2s_batch = None

            if ph_batch is None and h2s_batch is None:
                break

            for batch in filter(None, [ph_batch, h2s_batch]):
                self.global_step += 1
                self._lr()
                self.optimizer.zero_grad()
                loss = self._ctc_step(batch)
                if loss is None:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                is_ph = (batch['lang_id'][0].item() == LANG_PHOENIX)
                total_loss += loss.item()
                if is_ph:
                    ph_loss_sum += loss.item();  n_ph += 1
                else:
                    h2s_loss_sum += loss.item(); n_h2s += 1

        n_total = max(1, n_ph + n_h2s)
        return {
            'loss':     total_loss / n_total,
            'loss_ph':  ph_loss_sum  / max(1, n_ph),
            'loss_h2s': h2s_loss_sum / max(1, n_h2s),
        }

    @torch.no_grad()
    def _validate_loader(self, loader, tokenizer):
        self.model.eval()
        total_loss, n = 0.0, 0
        all_preds, all_targets = [], []

        for batch in loader:
            kps  = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            tgts = batch['gloss'].to(self.device)
            lang = batch['lang_id'].to(self.device)

            logits, mask_out = self.model(kps, mask, lang_id=lang)
            logits_ctc  = logits.permute(1, 0, 2)
            input_lens  = mask_out.sum(1).long().cpu()
            target_lens = (tgts != 0).sum(1).long().cpu()

            if (target_lens > 0).all() and (input_lens >= target_lens).all():
                loss = self.criterion(logits_ctc, tgts, input_lens, target_lens)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item(); n += 1

            preds = ctc_greedy_decode(logits.cpu(), input_lens)
            all_preds.extend(preds)
            all_targets.extend(batch['gloss_text'])

        avg_loss = total_loss / max(1, n)
        # WER
        pred_texts = [tokenizer.decode(p) for p in all_preds]
        wer = self._wer(pred_texts, all_targets)
        return avg_loss, wer

    @staticmethod
    def _wer(preds, targets):
        def _w(p, t):
            pw, tw = p.split(), t.split()
            if not tw: return 0.0
            dp = np.zeros((len(tw)+1, len(pw)+1))
            for i in range(len(tw)+1): dp[i,0] = i
            for j in range(len(pw)+1): dp[0,j] = j
            for i in range(1,len(tw)+1):
                for j in range(1,len(pw)+1):
                    cost = 0 if tw[i-1]==pw[j-1] else 1
                    dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1, dp[i-1,j-1]+cost)
            return dp[len(tw),len(pw)] / len(tw)
        if not preds: return 1.0
        return float(np.mean([_w(p,t) for p,t in zip(preds,targets)]))

    def train(self, num_epochs=150, beam_width=10):
        print(f"\n🏋️ Joint training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            t0 = time.time()
            metrics = self.train_epoch(epoch)

            val_loss_ph,  val_wer_ph  = self._validate_loader(
                self.ph_val_loader,  self.phoenix_tok)
            val_loss_h2s, val_wer_h2s = self._validate_loader(
                self.h2s_val_loader, self.how2sign_tok)

            combined_loss = 0.5 * val_loss_ph + 0.5 * val_loss_h2s
            lr = self.optimizer.param_groups[0]['lr']
            elapsed = time.time() - t0

            print(f"Epoch {epoch:3d} [{elapsed:.0f}s]  "
                  f"Train={metrics['loss']:.4f} "
                  f"(PH={metrics['loss_ph']:.4f} H2S={metrics['loss_h2s']:.4f})  "
                  f"Val_PH={val_loss_ph:.4f}(WER={val_wer_ph:.3f})  "
                  f"Val_H2S={val_loss_h2s:.4f}(WER={val_wer_h2s:.3f})  "
                  f"LR={lr:.2e}")

            # History
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(metrics['loss'])
            self.history['train_loss_ph'].append(metrics['loss_ph'])
            self.history['train_loss_h2s'].append(metrics['loss_h2s'])
            self.history['val_loss_ph'].append(val_loss_ph)
            self.history['val_loss_h2s'].append(val_loss_h2s)
            self.history['val_wer_ph'].append(val_wer_ph)
            self.history['val_wer_h2s'].append(val_wer_h2s)

            if wandb.run:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': metrics['loss'],
                    'train/loss_phoenix': metrics['loss_ph'],
                    'train/loss_how2sign': metrics['loss_h2s'],
                    'train/lr': lr,
                    'val/loss_phoenix': val_loss_ph,
                    'val/loss_how2sign': val_loss_h2s,
                    'val/wer_phoenix': val_wer_ph,
                    'val/wer_how2sign': val_wer_h2s,
                }, step=epoch)

            if combined_loss < self.best_val_loss:
                self.best_val_loss = combined_loss
                best_path = self.models_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': combined_loss,
                    'val_wer_phoenix': val_wer_ph,
                    'val_wer_how2sign': val_wer_h2s,
                }, best_path)
                print(f"  ✅ Best model saved (combined_loss={combined_loss:.4f})")
                if wandb.run:
                    wandb.log({'val/best_combined_loss': combined_loss}, step=epoch)

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                }, self.models_dir / f'checkpoint_epoch_{epoch}.pt')

        self._plot_joint_curves()

    def _plot_joint_curves(self):
        h = self.history
        if not h['epoch']:
            return
        ep = h['epoch']
        plots = {}

        # 1. Loss: both languages
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(ep, h['train_loss_ph'],  label='Train PH',  color='royalblue', lw=1.8)
        axes[0].plot(ep, h['val_loss_ph'],    label='Val PH',    color='steelblue', lw=1.8, ls='--')
        axes[0].plot(ep, h['train_loss_h2s'], label='Train H2S', color='tomato',    lw=1.8)
        axes[0].plot(ep, h['val_loss_h2s'],   label='Val H2S',   color='salmon',    lw=1.8, ls='--')
        axes[0].set_title('Loss — PHOENIX vs How2Sign', fontweight='bold')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('CTC Loss')
        axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

        # 2. WER: both languages
        axes[1].plot(ep, h['val_wer_ph'],  label='Val WER (DGS)', color='royalblue', lw=1.8)
        axes[1].plot(ep, h['val_wer_h2s'], label='Val WER (ASL)', color='tomato',    lw=1.8)
        axes[1].set_title('Validation WER — PHOENIX vs How2Sign', fontweight='bold')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('WER')
        axes[1].legend(); axes[1].grid(alpha=0.3)

        for ax in axes:
            ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        plots['plots/joint_loss_wer'] = _fig_to_wandb(fig)

        if wandb.run:
            wandb.log(plots)
            print("  📈 Joint training curves saved to W&B.")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Joint PHOENIX + How2Sign training with language-aware encoder')
    parser.add_argument('--phoenix_dir', type=str,
                        default='/data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T')
    parser.add_argument('--how2sign_dir', type=str, default='/data/how2sign_hf')
    parser.add_argument('--output_dir',  type=str, default='/data/experiments')
    parser.add_argument('--exp_name',    type=str, default='exp11_joint_training')
    parser.add_argument('--epochs',      type=int, default=150)
    parser.add_argument('--dim',         type=int, default=256)
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--batch_size',  type=int, default=16)
    parser.add_argument('--max_frames',  type=int, default=300)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--beam_width',  type=int, default=10)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--decode',      type=str, default='beam',
                        choices=['beam', 'greedy'])
    parser.add_argument('--lr',          type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    phoenix_dir  = Path(args.phoenix_dir)
    how2sign_dir = Path(args.how2sign_dir)
    output_dir   = Path(args.output_dir)
    models_dir   = output_dir / args.exp_name / 'models'
    results_dir  = output_dir / args.exp_name / 'results'
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project='slt',
        name=args.exp_name,
        config={**vars(args), 'mode': 'joint_training'},
    )

    # ── Tokenizers ──
    print("\nBuilding tokenizers...")
    phoenix_glosses = []
    for split in ['train', 'dev', 'test']:
        csv = phoenix_dir / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
        df = pd.read_csv(csv, sep='|')
        phoenix_glosses.extend(df['orth'].dropna().tolist())
    phoenix_tok = GlossTokenizer(phoenix_glosses, min_freq=1)

    h2s_csv = how2sign_dir / 'annotations' / 'how2sign_train.csv'
    h2s_glosses = []
    if h2s_csv.exists():
        df_h2s = pd.read_csv(h2s_csv, sep='\t')
        if 'PSEUDOGLOSS' in df_h2s.columns:
            h2s_glosses = df_h2s['PSEUDOGLOSS'].dropna().tolist()
    how2sign_tok = GlossTokenizer(h2s_glosses or ['DUMMY'], min_freq=2)
    print(f"  PHOENIX vocab: {phoenix_tok.vocab_size}  |  How2Sign vocab: {how2sign_tok.vocab_size}")

    # ── Datasets ──
    print("\nLoading datasets...")
    ph_train = PhoenixSignDataset(phoenix_dir, 'train', args.max_frames, True,  phoenix_tok, None)
    ph_val   = PhoenixSignDataset(phoenix_dir, 'dev',   args.max_frames, False, phoenix_tok, None)
    h2s_train = How2SignDataset(how2sign_dir, 'train', args.max_frames, True,  how2sign_tok, None)
    h2s_val   = How2SignDataset(how2sign_dir, 'val',   args.max_frames, False, how2sign_tok, None)
    print(f"  PHOENIX  train={len(ph_train)}  val={len(ph_val)}")
    print(f"  How2Sign train={len(h2s_train)} val={len(h2s_val)}")

    # ── Model ──
    model = LanguageAwareSignTransformer(
        input_dim=225, dim=args.dim,
        num_classes_phoenix=phoenix_tok.vocab_size,
        num_classes_how2sign=how2sign_tok.vocab_size,
        max_frames=args.max_frames * 2,
        dropout=args.dropout,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"🤖 LanguageAwareSignTransformer: {total:,} parameters")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")

    # ── Train ──
    trainer = JointTrainer(
        model, ph_train, h2s_train, ph_val, h2s_val,
        phoenix_tok, how2sign_tok,
        device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, models_dir=str(models_dir),
        lr=args.lr,
    )
    trainer.train(num_epochs=args.epochs, beam_width=args.beam_width)

    # ── Save config ──
    with open(results_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    wandb.finish()
    print(f"\n✨ Joint training complete — model saved to {models_dir}")


if __name__ == '__main__':
    main()
