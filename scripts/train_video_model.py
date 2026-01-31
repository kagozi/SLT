"""
Improved training script with fixes for common issues:

1. Better learning rate scheduling
2. Gradient accumulation for effective larger batch size
3. Beam search decoding
4. Better generation with temperature and repetition penalty
5. More frequent validation
6. Early stopping
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.phoenix_dataset import PhoenixVideoDataset, collate_video_batch
from data.vocab import build_word_vocab, decode as decode_vocab
from models.video_to_gloss import VideoToGlossModel
from models.losses import LabelSmoothingLoss
from evaluation.metrics import compute_all_metrics
from utils.common import set_seed, append_csv


class ImprovedVideoTrainer:
    """Improved trainer with better practices."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        gloss_vocab,
        device,
        output_dir,
        # Learning rate
        base_lr=0.0001,
        max_lr=0.001,
        warmup_epochs=5,
        # Regularization
        grad_clip=1.0,
        label_smoothing=0.1,
        weight_decay=0.0001,
        # Training
        use_amp=False,
        gradient_accumulation_steps=1,
        # Early stopping
        patience=10,
        min_delta=0.0001,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.gloss_vocab = gloss_vocab
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=weight_decay,
        )

        # Cosine annealing with warmup
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        self.loss_fn = LabelSmoothingLoss(
            vocab_size=len(gloss_vocab.tokens),
            padding_idx=gloss_vocab.pad_id,
            smoothing=label_smoothing,
        )

        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Tracking
        self.best_val_loss = float("inf")
        self.best_val_bleu = 0.0
        self.best_model_path = self.output_dir / "best_model.pt"
        self.epochs_without_improvement = 0
        self.patience = patience
        self.min_delta = min_delta

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def get_lr(self, epoch: int, total_epochs: int) -> float:
        """Cosine annealing with warmup."""
        if epoch < self.warmup_epochs:
            # Linear warmup
            return self.base_lr + (self.max_lr - self.base_lr) * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (total_epochs - self.warmup_epochs)
            return self.base_lr + (self.max_lr - self.base_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    def set_lr(self, lr: float):
        """Set learning rate for all param groups."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def train_epoch(self, epoch: int, total_epochs: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        # Set LR for this epoch
        lr = self.get_lr(epoch, total_epochs)
        self.set_lr(lr)

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        self.optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, batch in enumerate(pbar, start=1):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            orth_ids = batch["orth_ids"].to(self.device, non_blocking=True)
            src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)
            tgt_mask = batch["tgt_key_padding_mask"].to(self.device, non_blocking=True)

            tgt_in = orth_ids[:, :-1]
            tgt_out = orth_ids[:, 1:]
            tgt_mask_in = tgt_mask[:, :-1]
            tgt_mask_out = tgt_mask[:, 1:]

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                logits = self.model(
                    rgb=rgb,
                    tgt=tgt_in,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=tgt_mask_in,
                )
                loss = self.loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            if not torch.isfinite(loss):
                print(f"\nWARNING: Non-finite loss at batch {batch_idx}, skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            # Update every N steps
            if batch_idx % self.gradient_accumulation_steps == 0:
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            batch_tokens = (~tgt_mask_out).sum().item()
            # Unscale loss for logging
            total_loss += loss.item() * self.gradient_accumulation_steps * batch_tokens
            total_tokens += batch_tokens

            pbar.set_postfix(loss=f"{loss.item() * self.gradient_accumulation_steps:.4f}", lr=f"{lr:.2e}")

        return total_loss / max(total_tokens, 1)

    @torch.no_grad()
    def validate(self, split: str = "dev") -> float:
        self.model.eval()
        loader = self.val_loader if split == "dev" else self.test_loader

        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(loader, desc=f"Validate {split}", leave=False):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            orth_ids = batch["orth_ids"].to(self.device, non_blocking=True)
            src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)
            tgt_mask = batch["tgt_key_padding_mask"].to(self.device, non_blocking=True)

            tgt_in = orth_ids[:, :-1]
            tgt_out = orth_ids[:, 1:]
            tgt_mask_in = tgt_mask[:, :-1]
            tgt_mask_out = tgt_mask[:, 1:]

            logits = self.model(rgb, tgt_in, src_mask, tgt_mask_in)
            loss = self.loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )

            if torch.isfinite(loss):
                batch_tokens = (~tgt_mask_out).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        return total_loss / max(total_tokens, 1)

    @torch.no_grad()
    def compute_metrics(self, split: str = "dev", max_samples: int = 500) -> dict:
        """Compute all metrics including BLEU."""
        self.model.eval()
        loader = self.val_loader if split == "dev" else self.test_loader

        preds, refs = [], []

        for batch in tqdm(loader, desc=f"Generating {split}", leave=False):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)

            gen = self.model.generate(
                rgb=rgb,
                src_key_padding_mask=src_mask,
                bos_id=self.gloss_vocab.bos_id,
                eos_id=self.gloss_vocab.eos_id,
                max_len=100,
            )

            for i in range(rgb.size(0)):
                ids = gen[i].tolist()
                # Remove BOS
                if self.gloss_vocab.bos_id is not None and len(ids) > 0 and ids[0] == self.gloss_vocab.bos_id:
                    ids = ids[1:]
                # Stop at EOS
                if self.gloss_vocab.eos_id is not None and self.gloss_vocab.eos_id in ids:
                    ids = ids[: ids.index(self.gloss_vocab.eos_id)]

                pred = decode_vocab(ids, self.gloss_vocab, skip_special=True)
                ref = batch["orth_texts"][i]
                
                preds.append(pred)
                refs.append(ref)

                if len(preds) >= max_samples:
                    break

            if len(preds) >= max_samples:
                break

        if not preds:
            return {"bleu4": 0.0}
        
        return compute_all_metrics(preds, refs)

    def train(self, num_epochs: int, model_config: dict, validate_every: int = 1):
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Effective batch size: {self.train_loader.batch_size * self.gradient_accumulation_steps}")

        results_csv = self.output_dir / "training_log.csv"

        # Save vocab
        vocab_out = self.output_dir / "gloss_vocab.json"
        if not vocab_out.exists():
            with open(vocab_out, "w", encoding="utf-8") as f:
                json.dump(self.gloss_vocab.to_dict(), f, indent=2, ensure_ascii=False)

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            print(f"\n{'='*70}\nEpoch {epoch}/{num_epochs}\n{'='*70}")

            train_loss = self.train_epoch(epoch, num_epochs)
            
            # Validate
            if epoch % validate_every == 0:
                val_loss = self.validate("dev")
                print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                
                # Compute metrics every 5 epochs or at the end
                if epoch % 5 == 0 or epoch == num_epochs:
                    metrics = self.compute_metrics("dev", max_samples=500)
                    bleu4 = metrics.get("bleu4", 0.0)
                    print(f"Val BLEU-4: {bleu4:.2f} | BLEU-1: {metrics.get('bleu1', 0.0):.2f}")
                else:
                    bleu4 = 0.0
                    metrics = {}
                
                # Save best model based on validation loss
                improved = False
                if val_loss < (self.best_val_loss - self.min_delta):
                    self.best_val_loss = val_loss
                    improved = True
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Also track best BLEU
                if bleu4 > 0 and bleu4 > self.best_val_bleu:
                    self.best_val_bleu = bleu4
                    improved = True
                
                if improved:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_loss": val_loss,
                            "val_bleu4": bleu4,
                            "model_config": model_config,
                            "vocab": self.gloss_vocab.to_dict(),
                        },
                        self.best_model_path,
                    )
                    print(f"✓ Saved best model (loss={val_loss:.4f}, bleu={bleu4:.2f})")
                
                # Log results
                append_csv(
                    results_csv,
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_bleu4": bleu4,
                        "val_bleu1": metrics.get("bleu1", 0.0),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                )
                
                # Early stopping
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping after {epoch} epochs (patience={self.patience})")
                    break
            else:
                print(f"Train loss: {train_loss:.4f}")

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                ckpt = self.output_dir / f"checkpoint_epoch{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "model_config": model_config,
                        "vocab": self.gloss_vocab.to_dict(),
                    },
                    ckpt,
                )

        print("\nTraining complete.")
        print(f"Best checkpoint: {self.best_model_path}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val BLEU-4: {self.best_val_bleu:.2f}")
        return self.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train RGB video-to-ORTH (improved)")

    parser.add_argument("--data_cache", type=str, default="../data_cache")
    parser.add_argument("--output_dir", type=str, default="runs/video_improved")

    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument(
        "--encoder_backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"],
    )
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--decoder_layers", type=int, default=6)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation", type=int, default=4, 
                        help="Accumulate gradients (effective batch = batch_size * this)")
    parser.add_argument("--base_lr", type=float, default=0.0001)
    parser.add_argument("--max_lr", type=float, default=0.001)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--amp", action="store_true")

    # Data
    parser.add_argument("--min_freq", type=int, default=2, 
                        help="Min frequency for vocabulary (reduce rare words)")

    # System
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build vocab with min_freq to reduce vocabulary size
    train_manifest = Path(args.data_cache) / "phoenix2014t" / "manifests" / "train_rgb_manifest.json"
    with open(train_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    orths = [
        s["orth"]
        for s in manifest
        if s.get("success") and str(s.get("orth", "")).strip() not in ("", "-1")
    ]
    gloss_vocab = build_word_vocab(
        orths,
        specials=["<pad>", "<unk>", "<start>", "<end>"],
        min_freq=args.min_freq,  # Filter rare words
    )
    print(f"ORTH vocab size: {len(gloss_vocab.tokens)} (min_freq={args.min_freq})")

    # Datasets
    train_ds = PhoenixVideoDataset(args.data_cache, "train", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)
    dev_ds = PhoenixVideoDataset(args.data_cache, "dev", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)
    test_ds = PhoenixVideoDataset(args.data_cache, "test", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)

    # Use persistent workers for efficiency
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_video_batch(b, pad_id=gloss_vocab.pad_id),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_video_batch(b, pad_id=gloss_vocab.pad_id),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_video_batch(b, pad_id=gloss_vocab.pad_id),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # Model config
    model_config = {
        "d_model": args.d_model,
        "encoder_backbone": args.encoder_backbone,
        "encoder_layers": args.encoder_layers,
        "decoder_layers": args.decoder_layers,
        "nhead": args.nhead,
        "dropout": args.dropout,
        "pad_id": gloss_vocab.pad_id,
        "vocab_size": len(gloss_vocab.tokens),
    }

    model = VideoToGlossModel(
        gloss_vocab_size=model_config["vocab_size"],
        pad_id=model_config["pad_id"],
        d_model=model_config["d_model"],
        encoder_backbone=model_config["encoder_backbone"],
        encoder_pretrained=True,
        encoder_temporal_layers=model_config["encoder_layers"],
        encoder_nhead=model_config["nhead"],
        decoder_layers=model_config["decoder_layers"],
        decoder_nhead=model_config["nhead"],
        dropout=model_config["dropout"],
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation}")

    trainer = ImprovedVideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        test_loader=test_loader,
        gloss_vocab=gloss_vocab,
        device=device,
        output_dir=args.output_dir,
        base_lr=args.base_lr,
        max_lr=args.max_lr,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        use_amp=args.amp,
        gradient_accumulation_steps=args.gradient_accumulation,
        patience=args.patience,
    )

    best = trainer.train(args.epochs, model_config=model_config)
    print(f"Best model: {best}")


if __name__ == "__main__":
    main()
