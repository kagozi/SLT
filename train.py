"""
Training script for MultiStream SLT.

Three-stage training:
  Stage 1 (epochs 0-freeze_epochs): CTC gloss recognition only (BART frozen)
  Stage 2 (epochs freeze_epochs-end): Joint CTC + translation (BART unfrozen)

Usage:
    python train.py --data_cache data_cache --dataset phoenix2014t --data_root /path/to/phoenix
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import BartTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.config import FullConfig, get_default_config
from models.multistream_slt import MultiStreamSLT
from data.phoenix_dataset import build_dataloaders
from utils.tokenizer import GlossTokenizer, ctc_greedy_decode
from utils.metrics import compute_wer, compute_bleu, evaluate_model


def cosine_lr_schedule(
    epoch: int, warmup_epochs: int, lr_max: float, total_epochs: int
) -> float:
    """Cosine LR with exponential warmup (from reference)."""
    if epoch < warmup_epochs:
        return lr_max * 2 ** -(warmup_epochs - epoch)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return max(1e-7, 0.5 * (1.0 + math.cos(math.pi * progress)) * lr_max)


class Trainer:
    def __init__(
        self,
        model: MultiStreamSLT,
        train_loader,
        val_loader,
        gloss_tokenizer: GlossTokenizer,
        bart_tokenizer: BartTokenizer,
        config: FullConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gloss_tok = gloss_tokenizer
        self.bart_tok = bart_tokenizer
        self.cfg = config.training
        self.device = device

        # Optimizer: AdamW with weight decay (from reference)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.cfg.lr_max,
            weight_decay=self.cfg.weight_decay,
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=self.cfg.mixed_precision)

        # Checkpointing
        self.save_dir = Path(self.cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_wer = float("inf")
        self.best_bleu = 0.0

    def train(self):
        freeze_epochs = 5  # Stage 1: CTC only

        print(f"\n{'='*60}")
        print(f"Starting training: {self.cfg.num_epochs} epochs")
        print(f"Stage 1 (CTC only, BART frozen): epochs 0-{freeze_epochs-1}")
        print(f"Stage 2 (Joint CTC + Translation): epochs {freeze_epochs}+")
        print(f"{'='*60}\n")

        # Stage 1: Freeze BART
        self.model.freeze_translation()

        for epoch in range(self.cfg.num_epochs):
            # Unfreeze BART at stage transition
            if epoch == freeze_epochs:
                print(f"\n>>> Stage 2: Unfreezing BART translation head <<<\n")
                self.model.unfreeze_translation()

            # Update learning rate
            lr = cosine_lr_schedule(epoch, self.cfg.warmup_epochs, self.cfg.lr_max, self.cfg.num_epochs)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
                pg["weight_decay"] = lr * 0.05  # from reference WD_RATIO

            # Train
            train_metrics = self._train_epoch(epoch)

            # Evaluate
            if (epoch + 1) % self.cfg.eval_every_epoch == 0:
                val_metrics = self._evaluate(epoch)
                self._maybe_save(epoch, val_metrics)

            # Log
            print(f"Epoch {epoch+1}/{self.cfg.num_epochs} | "
                  f"lr={lr:.2e} | "
                  f"train_loss={train_metrics['loss']:.4f} | "
                  f"ctc={train_metrics.get('ctc_loss', 0):.4f} | "
                  f"trans={train_metrics.get('trans_loss', 0):.4f}")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_ctc = 0.0
        total_trans = 0.0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train E{epoch+1}")
        for batch in pbar:
            rgb = batch["rgb"].to(self.device)
            hands = batch["hands"].to(self.device)
            kpts = batch["kpts"].to(self.device)
            mask = batch["mask"].to(self.device)

            gloss_targets = batch.get("gloss_targets")
            gloss_lengths = batch.get("gloss_lengths")
            trans_targets = batch.get("translation_targets")

            if gloss_targets is not None:
                gloss_targets = gloss_targets.to(self.device)
                gloss_lengths = gloss_lengths.to(self.device)
            if trans_targets is not None:
                trans_targets = trans_targets.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self.cfg.mixed_precision):
                output = self.model(
                    rgb=rgb, hands=hands, kpts=kpts, mask=mask,
                    gloss_targets=gloss_targets,
                    gloss_lengths=gloss_lengths,
                    translation_targets=trans_targets,
                )
                loss = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            total_ctc += output.get("ctc_loss", torch.tensor(0)).item()
            total_trans += output.get("translation_loss", torch.tensor(0)).item()
            n_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {
            "loss": total_loss / max(1, n_batches),
            "ctc_loss": total_ctc / max(1, n_batches),
            "trans_loss": total_trans / max(1, n_batches),
        }

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        gloss_refs, gloss_hyps = [], []
        trans_refs, trans_hyps = [], []

        for batch in tqdm(self.val_loader, desc=f"Eval E{epoch+1}"):
            rgb = batch["rgb"].to(self.device)
            hands = batch["hands"].to(self.device)
            kpts = batch["kpts"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Gloss recognition via CTC
            fused = self.model.encode(rgb, hands, kpts, mask)
            gloss_logits = self.model.ctc_head(fused)
            decoded_glosses = ctc_greedy_decode(gloss_logits)

            if "gloss_targets" in batch:
                for i, target in enumerate(batch["gloss_targets"]):
                    ref = self.gloss_tok.decode(target.tolist())
                    hyp = self.gloss_tok.decode(decoded_glosses[i])
                    gloss_refs.append(ref)
                    gloss_hyps.append(hyp)

            # Translation via BART beam search
            try:
                token_ids = self.model.translate(rgb, hands, kpts, mask, beam_width=5)
                for i in range(token_ids.shape[0]):
                    hyp = self.bart_tok.decode(token_ids[i], skip_special_tokens=True)
                    trans_hyps.append(hyp)

                if "translation_targets" in batch:
                    for target in batch["translation_targets"]:
                        ref = self.bart_tok.decode(target, skip_special_tokens=True)
                        trans_refs.append(ref)
            except Exception:
                pass  # Translation may fail during stage 1

        metrics = evaluate_model(gloss_refs, gloss_hyps, trans_refs, trans_hyps)

        print(f"\n--- Eval Epoch {epoch+1} ---")
        if "wer" in metrics:
            print(f"  WER:    {metrics['wer']:.4f}")
        if "bleu-4" in metrics:
            print(f"  BLEU-4: {metrics['bleu-4']:.4f}")
        if "bleu-1" in metrics:
            print(f"  BLEU-1: {metrics['bleu-1']:.4f}")

        # Show a few examples
        for i in range(min(3, len(gloss_refs))):
            print(f"  Ref gloss:  {gloss_refs[i]}")
            print(f"  Pred gloss: {gloss_hyps[i]}")
            if i < len(trans_refs):
                print(f"  Ref trans:  {trans_refs[i]}")
                print(f"  Pred trans: {trans_hyps[i]}")
            print("  ---")

        return metrics

    def _maybe_save(self, epoch: int, metrics: Dict[str, float]):
        wer = metrics.get("wer", float("inf"))
        bleu = metrics.get("bleu-4", 0.0)

        save_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "gloss_tokenizer": {"word2idx": self.gloss_tok.word2idx},
        }

        # Save latest
        torch.save(save_data, self.save_dir / "latest.pt")

        # Save best WER
        if wer < self.best_wer:
            self.best_wer = wer
            torch.save(save_data, self.save_dir / "best_wer.pt")
            print(f"  ✅ New best WER: {wer:.4f}")

        # Save best BLEU
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            torch.save(save_data, self.save_dir / "best_bleu.pt")
            print(f"  ✅ New best BLEU-4: {bleu:.4f}")


def build_gloss_tokenizer(data_cache: str, dataset: str) -> GlossTokenizer:
    """Build gloss vocabulary from RGB manifest annotations."""
    tok = GlossTokenizer()
    gloss_strings = []

    for split in ["train", "dev", "test"]:
        manifest_path = Path(data_cache) / dataset / "manifests" / f"{split}_rgb_manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            entries = json.load(f)
        for e in entries:
            if e.get("success") and e.get("orth"):
                gloss_strings.append(e["orth"])

    tok.build_vocab(gloss_strings, min_freq=1)
    return tok


def main():
    parser = argparse.ArgumentParser(description="Train MultiStream SLT")
    parser.add_argument("--data_cache", type=str, default="data_cache")
    parser.add_argument("--dataset", type=str, default="phoenix2014t")
    parser.add_argument("--data_root", type=str, default="../data_cache")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--fusion", type=str, default="cross_attention",
                        choices=["cross_attention", "concat", "gated"])
    parser.add_argument("--bart_model", type=str, default="facebook/bart-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Config
    config = get_default_config(args.data_root)
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.lr_max = args.lr

    # Tokenizers
    print("Building gloss tokenizer...")
    gloss_tok = build_gloss_tokenizer(args.data_cache, args.dataset)

    print("Loading BART tokenizer...")
    bart_tok = BartTokenizer.from_pretrained(args.bart_model)

    # DataLoaders
    print("Building dataloaders...")
    loaders = build_dataloaders(
        data_cache_dir=args.data_cache,
        dataset=args.dataset,
        gloss_tokenizer=gloss_tok,
        translation_tokenizer=bart_tok,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_frames=args.num_frames,
    )

    # Model
    print("Building MultiStreamSLT model...")
    model = MultiStreamSLT(
        hidden_dim=args.hidden_dim,
        num_frames=args.num_frames,
        kpts_dim=498,
        gloss_vocab_size=gloss_tok.vocab_size,
        fusion_strategy=args.fusion,
        bart_model=args.bart_model,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {total_params:,} total, {trainable:,} trainable")

    # Resume
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Train
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["dev"],
        gloss_tokenizer=gloss_tok,
        bart_tokenizer=bart_tok,
        config=config,
        device=device,
    )
    trainer.train()

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()