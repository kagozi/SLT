# """
# Training script for video-to-gloss (ORTH) model (Stage 1: RGB-only).

# Usage:
#     python scripts/train_video_model.py \
#         --data_cache ../data_cache \
#         --output_dir runs/video_stage1 \
#         --epochs 50 \
#         --batch_size 8
# """

# import argparse
# import json
# import sys
# from pathlib import Path

# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from data.phoenix_dataset import PhoenixVideoDataset, collate_video_batch
# from data.vocab import build_word_vocab, decode as decode_vocab
# from models.video_to_gloss import VideoToGlossModel
# from models.losses import LabelSmoothingLoss
# from evaluation.metrics import corpus_bleu
# from utils.common import set_seed, append_csv


# class VideoTrainer:
#     """Trainer for RGB-only video-to-ORTH models."""

#     def __init__(
#         self,
#         model,
#         train_loader,
#         val_loader,
#         test_loader,
#         gloss_vocab,
#         device,
#         output_dir,
#         lr_factor=1.0,
#         warmup_steps=4000,
#         grad_clip=1.0,
#         label_smoothing=0.1,
#         use_amp=False,
#     ):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.test_loader = test_loader
#         self.gloss_vocab = gloss_vocab
#         self.device = device
#         self.output_dir = Path(output_dir)
#         self.output_dir.mkdir(parents=True, exist_ok=True)

#         self.optimizer = torch.optim.Adam(
#             self.model.parameters(),
#             lr=1.0,  # Noam will override
#             betas=(0.9, 0.98),
#             eps=1e-9,
#         )

#         self.lr_factor = lr_factor
#         self.warmup_steps = warmup_steps
#         self.d_model = model.d_model
#         self.step_num = 0

#         self.loss_fn = LabelSmoothingLoss(
#             vocab_size=len(gloss_vocab.tokens),
#             padding_idx=gloss_vocab.pad_id,
#             smoothing=label_smoothing,
#         )

#         self.grad_clip = grad_clip
#         self.best_val_loss = float("inf")
#         self.best_model_path = self.output_dir / "best_model.pt"

#         self.use_amp = use_amp
#         self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

#     def _update_lr(self) -> float:
#         self.step_num += 1
#         lr = self.lr_factor * (self.d_model ** -0.5) * min(
#             self.step_num ** -0.5,
#             self.step_num * (self.warmup_steps ** -1.5),
#         )
#         for pg in self.optimizer.param_groups:
#             pg["lr"] = lr
#         return lr

#     def train_epoch(self, epoch: int) -> float:
#         self.model.train()
#         total_loss = 0.0
#         total_tokens = 0

#         pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
#         for batch in pbar:
#             rgb = batch["rgb"].to(self.device, non_blocking=True)
#             orth_ids = batch["orth_ids"].to(self.device, non_blocking=True)
#             src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)
#             tgt_mask = batch["tgt_key_padding_mask"].to(self.device, non_blocking=True)

#             # teacher forcing: predict orth_ids[:,1:] from orth_ids[:,:-1]
#             tgt_in = orth_ids[:, :-1]
#             tgt_out = orth_ids[:, 1:]
#             tgt_mask_in = tgt_mask[:, :-1]
#             tgt_mask_out = tgt_mask[:, 1:]

#             self.optimizer.zero_grad(set_to_none=True)

#             with torch.cuda.amp.autocast(enabled=self.use_amp):
#                 logits = self.model(
#                     rgb=rgb,
#                     tgt=tgt_in,
#                     src_key_padding_mask=src_mask,
#                     tgt_key_padding_mask=tgt_mask_in,
#                 )
#                 loss = self.loss_fn(
#                     logits.reshape(-1, logits.size(-1)),
#                     tgt_out.reshape(-1),
#                 )

#             self.scaler.scale(loss).backward()

#             if self.grad_clip and self.grad_clip > 0:
#                 self.scaler.unscale_(self.optimizer)
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#             lr = self._update_lr()

#             # token counting excludes padding in tgt_out
#             batch_tokens = (~tgt_mask_out).sum().item()
#             total_loss += loss.item() * batch_tokens
#             total_tokens += batch_tokens

#             pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

#         return total_loss / max(total_tokens, 1)

#     @torch.no_grad()
#     def validate(self, split: str = "dev") -> float:
#         self.model.eval()
#         loader = self.val_loader if split == "dev" else self.test_loader

#         total_loss = 0.0
#         total_tokens = 0

#         for batch in tqdm(loader, desc=f"Validate {split}"):
#             rgb = batch["rgb"].to(self.device, non_blocking=True)
#             orth_ids = batch["orth_ids"].to(self.device, non_blocking=True)
#             src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)
#             tgt_mask = batch["tgt_key_padding_mask"].to(self.device, non_blocking=True)

#             tgt_in = orth_ids[:, :-1]
#             tgt_out = orth_ids[:, 1:]
#             tgt_mask_in = tgt_mask[:, :-1]
#             tgt_mask_out = tgt_mask[:, 1:]

#             logits = self.model(rgb, tgt_in, src_mask, tgt_mask_in)
#             loss = self.loss_fn(
#                 logits.reshape(-1, logits.size(-1)),
#                 tgt_out.reshape(-1),
#             )

#             batch_tokens = (~tgt_mask_out).sum().item()
#             total_loss += loss.item() * batch_tokens
#             total_tokens += batch_tokens

#         return total_loss / max(total_tokens, 1)

#     @torch.no_grad()
#     def compute_bleu(self, split: str = "dev", max_samples: int = 200) -> float:
#         self.model.eval()
#         loader = self.val_loader if split == "dev" else self.test_loader

#         preds, refs = [], []

#         for batch in tqdm(loader, desc=f"BLEU {split}"):
#             rgb = batch["rgb"].to(self.device, non_blocking=True)
#             src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)

#             gen = self.model.generate(
#                 rgb=rgb,
#                 src_key_padding_mask=src_mask,
#                 bos_id=self.gloss_vocab.bos_id,
#                 eos_id=self.gloss_vocab.eos_id,
#                 max_len=100,
#             )

#             for i in range(rgb.size(0)):
#                 ids = gen[i].tolist()
#                 if self.gloss_vocab.eos_id in ids:
#                     ids = ids[: ids.index(self.gloss_vocab.eos_id)]
#                 pred = decode_vocab(ids, self.gloss_vocab, skip_special=True)

#                 ref = batch["orth_texts"][i]
#                 preds.append(pred)
#                 refs.append(ref)

#                 if len(preds) >= max_samples:
#                     break

#             if len(preds) >= max_samples:
#                 break

#         if not preds:
#             return 0.0
#         return corpus_bleu(preds, refs)

#     def train(self, num_epochs: int, model_config: dict):
#         print(f"Device: {self.device}")
#         print(f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")

#         results_csv = self.output_dir / "training_log.csv"

#         # Save vocab once
#         vocab_out = self.output_dir / "gloss_vocab.json"
#         if not vocab_out.exists():
#             with open(vocab_out, "w", encoding="utf-8") as f:
#                 json.dump(self.gloss_vocab.to_dict(), f, indent=2, ensure_ascii=False)

#         for epoch in range(1, num_epochs + 1):
#             print(f"\n{'='*70}\nEpoch {epoch}/{num_epochs}\n{'='*70}")

#             train_loss = self.train_epoch(epoch)
#             val_loss = self.validate("dev")

#             print(f"Train loss: {train_loss:.4f}")
#             print(f"Val loss:   {val_loss:.4f}")

#             # Save best
#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 torch.save(
#                     {
#                         "epoch": epoch,
#                         "model_state_dict": self.model.state_dict(),
#                         "optimizer_state_dict": self.optimizer.state_dict(),
#                         "val_loss": val_loss,
#                         "step_num": self.step_num,
#                         "model_config": model_config,
#                         "vocab": self.gloss_vocab.to_dict(),
#                     },
#                     self.best_model_path,
#                 )
#                 print(f"✓ Saved best model: {self.best_model_path}")

#             if epoch % 5 == 0 or epoch == num_epochs:
#                 bleu = self.compute_bleu("dev", max_samples=200)
#                 print(f"Val BLEU-4: {bleu:.2f}")
#             else:
#                 bleu = 0.0

#             append_csv(
#                 results_csv,
#                 {
#                     "epoch": epoch,
#                     "train_loss": train_loss,
#                     "val_loss": val_loss,
#                     "val_bleu4": bleu,
#                     "lr": self.optimizer.param_groups[0]["lr"],
#                     "step": self.step_num,
#                 },
#             )

#             if epoch % 10 == 0:
#                 ckpt = self.output_dir / f"checkpoint_epoch{epoch}.pt"
#                 torch.save(
#                     {
#                         "epoch": epoch,
#                         "model_state_dict": self.model.state_dict(),
#                         "optimizer_state_dict": self.optimizer.state_dict(),
#                         "step_num": self.step_num,
#                         "model_config": model_config,
#                         "vocab": self.gloss_vocab.to_dict(),
#                     },
#                     ckpt,
#                 )

#         print("\nTraining complete.")
#         print(f"Best checkpoint: {self.best_model_path}")
#         return self.best_model_path


# def main():
#     parser = argparse.ArgumentParser(description="Train RGB-only video-to-ORTH model")

#     parser.add_argument("--data_cache", type=str, default="../data_cache")
#     parser.add_argument("--output_dir", type=str, default="runs/video_stage1")

#     # model
#     parser.add_argument("--d_model", type=int, default=512)
#     parser.add_argument("--encoder_backbone", type=str, default="resnet18",
#                         choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
#     parser.add_argument("--encoder_layers", type=int, default=4)
#     parser.add_argument("--decoder_layers", type=int, default=6)
#     parser.add_argument("--nhead", type=int, default=8)
#     parser.add_argument("--dropout", type=float, default=0.1)

#     # training
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--lr_factor", type=float, default=1.0,
#                         help="Multiplier for Noam LR schedule (common: 1.0)")
#     parser.add_argument("--warmup_steps", type=int, default=4000)
#     parser.add_argument("--grad_clip", type=float, default=1.0)
#     parser.add_argument("--label_smoothing", type=float, default=0.1)
#     parser.add_argument("--amp", action="store_true", help="Use mixed precision AMP")

#     # system
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--num_workers", type=int, default=4)
#     parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

#     args = parser.parse_args()
#     set_seed(args.seed)

#     device = torch.device(args.device)
#     print(f"Using device: {device}")

#     # Build vocab from TRAIN orth
#     train_manifest = Path(args.data_cache) / "phoenix2014t" / "manifests" / "train_rgb_manifest.json"
#     with open(train_manifest, "r", encoding="utf-8") as f:
#         manifest = json.load(f)

#     orths = [s["orth"] for s in manifest if s.get("success") and str(s.get("orth", "")).strip() not in ("", "-1")]
#     gloss_vocab = build_word_vocab(
#         orths,
#         specials=["<pad>", "<unk>", "<start>", "<end>"],
#         min_freq=1,
#     )
#     print(f"ORTH vocab size: {len(gloss_vocab.tokens)}")

#     # datasets
#     train_ds = PhoenixVideoDataset(args.data_cache, "train", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)
#     dev_ds = PhoenixVideoDataset(args.data_cache, "dev", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)
#     test_ds = PhoenixVideoDataset(args.data_cache, "test", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         collate_fn=lambda b: collate_video_batch(b, pad_id=gloss_vocab.pad_id),
#         pin_memory=(device.type == "cuda"),
#     )
#     dev_loader = DataLoader(
#         dev_ds,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         collate_fn=lambda b: collate_video_batch(b, pad_id=gloss_vocab.pad_id),
#         pin_memory=(device.type == "cuda"),
#     )
#     test_loader = DataLoader(
#         test_ds,
#         batch_size=args.batch_size,
#         shuffle=False,
#         num_workers=args.num_workers,
#         collate_fn=lambda b: collate_video_batch(b, pad_id=gloss_vocab.pad_id),
#         pin_memory=(device.type == "cuda"),
#     )

#     # model
#     model_config = {
#         "d_model": args.d_model,
#         "encoder_backbone": args.encoder_backbone,
#         "encoder_layers": args.encoder_layers,
#         "decoder_layers": args.decoder_layers,
#         "nhead": args.nhead,
#         "dropout": args.dropout,
#         "pad_id": gloss_vocab.pad_id,
#         "vocab_size": len(gloss_vocab.tokens),
#     }

#     model = VideoToGlossModel(
#         gloss_vocab_size=model_config["vocab_size"],
#         pad_id=model_config["pad_id"],
#         d_model=model_config["d_model"],
#         encoder_backbone=model_config["encoder_backbone"],
#         encoder_pretrained=True,
#         encoder_temporal_layers=model_config["encoder_layers"],
#         encoder_nhead=model_config["nhead"],
#         decoder_layers=model_config["decoder_layers"],
#         decoder_nhead=model_config["nhead"],
#         dropout=model_config["dropout"],
#     )

#     print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

#     trainer = VideoTrainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=dev_loader,
#         test_loader=test_loader,
#         gloss_vocab=gloss_vocab,
#         device=device,
#         output_dir=args.output_dir,
#         lr_factor=args.lr_factor,
#         warmup_steps=args.warmup_steps,
#         grad_clip=args.grad_clip,
#         label_smoothing=args.label_smoothing,
#         use_amp=args.amp,
#     )

#     best = trainer.train(args.epochs, model_config=model_config)
#     print(f"Best model: {best}")


# if __name__ == "__main__":
#     main()
"""
Training script for video-to-gloss (ORTH) model (Stage 1: RGB-only).

Usage:
    python scripts/train_video_model.py \
        --data_cache ../data_cache \
        --output_dir runs/video_stage1 \
        --epochs 50 \
        --batch_size 8
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
from evaluation.metrics import corpus_bleu
from utils.common import set_seed, append_csv


class VideoTrainer:
    """Trainer for RGB-only video-to-ORTH models."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        gloss_vocab,
        device,
        output_dir,
        lr_factor=1.0,
        warmup_steps=4000,
        grad_clip=1.0,
        label_smoothing=0.1,
        use_amp=False,
        skip_nonfinite=True,
        log_nonfinite_every=25,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.gloss_vocab = gloss_vocab
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT:
        # Start Adam at lr=0.0 (or small), and let Noam schedule set it BEFORE each step.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        self.lr_factor = lr_factor
        self.warmup_steps = warmup_steps
        self.d_model = model.d_model
        self.step_num = 0

        self.loss_fn = LabelSmoothingLoss(
            vocab_size=len(gloss_vocab.tokens),
            padding_idx=gloss_vocab.pad_id,
            smoothing=label_smoothing,
        )

        self.grad_clip = grad_clip
        self.best_val_loss = float("inf")
        self.best_model_path = self.output_dir / "best_model.pt"

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.skip_nonfinite = skip_nonfinite
        self.log_nonfinite_every = log_nonfinite_every
        self._nonfinite_skips = 0

    def _update_lr(self) -> float:
        """Noam schedule: set lr for the CURRENT step."""
        self.step_num += 1
        lr = self.lr_factor * (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5),
        )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        for it, batch in enumerate(pbar, start=1):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            orth_ids = batch["orth_ids"].to(self.device, non_blocking=True)
            src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)
            tgt_mask = batch["tgt_key_padding_mask"].to(self.device, non_blocking=True)

            # teacher forcing: predict orth_ids[:,1:] from orth_ids[:,:-1]
            tgt_in = orth_ids[:, :-1]
            tgt_out = orth_ids[:, 1:]
            tgt_mask_in = tgt_mask[:, :-1]
            tgt_mask_out = tgt_mask[:, 1:]

            # ✅ set LR BEFORE computing/stepping (so this update uses correct lr)
            lr = self._update_lr()

            self.optimizer.zero_grad(set_to_none=True)

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

            # ✅ guard against NaN/Inf loss (common early if something spikes)
            if not torch.isfinite(loss):
                self._nonfinite_skips += 1
                if (self._nonfinite_skips % self.log_nonfinite_every) == 1:
                    print(
                        f"\n[WARN] Non-finite loss detected (loss={loss.item()}). "
                        f"Skipping update. skips={self._nonfinite_skips} | epoch={epoch} | iter={it} | lr={lr:.2e}"
                    )
                if self.skip_nonfinite:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                else:
                    raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

            self.scaler.scale(loss).backward()

            if self.grad_clip and self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # token counting excludes padding in tgt_out
            batch_tokens = (~tgt_mask_out).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}", skip=self._nonfinite_skips)

        return total_loss / max(total_tokens, 1)

    @torch.no_grad()
    def validate(self, split: str = "dev") -> float:
        self.model.eval()
        loader = self.val_loader if split == "dev" else self.test_loader

        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(loader, desc=f"Validate {split}"):
            rgb = batch["rgb"].to(self.device, non_blocking=True)
            orth_ids = batch["orth_ids"].to(self.device, non_blocking=True)
            src_mask = batch["src_key_padding_mask"].to(self.device, non_blocking=True)
            tgt_mask = batch["tgt_key_padding_mask"].to(self.device, non_blocking=True)

            tgt_in = orth_ids[:, :-1]
            tgt_out = orth_ids[:, 1:]
            tgt_mask_in = tgt_mask[:, :-1]
            tgt_mask_out = tgt_mask[:, 1:]

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

            # if eval ever hits non-finite, just skip that batch
            if not torch.isfinite(loss):
                continue

            batch_tokens = (~tgt_mask_out).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        return total_loss / max(total_tokens, 1)

    @torch.no_grad()
    def compute_bleu(self, split: str = "dev", max_samples: int = 200) -> float:
        self.model.eval()
        loader = self.val_loader if split == "dev" else self.test_loader

        preds, refs = [], []

        for batch in tqdm(loader, desc=f"BLEU {split}"):
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
                if self.gloss_vocab.eos_id in ids:
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
            return 0.0
        return corpus_bleu(preds, refs)

    def train(self, num_epochs: int, model_config: dict):
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)} | Val batches: {len(self.val_loader)}")

        results_csv = self.output_dir / "training_log.csv"

        # Save vocab once (only if your Vocab implements to_dict)
        vocab_out = self.output_dir / "gloss_vocab.json"
        if not vocab_out.exists():
            if hasattr(self.gloss_vocab, "to_dict"):
                with open(vocab_out, "w", encoding="utf-8") as f:
                    json.dump(self.gloss_vocab.to_dict(), f, indent=2, ensure_ascii=False)

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}\nEpoch {epoch}/{num_epochs}\n{'='*70}")

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate("dev")

            print(f"Train loss: {train_loss:.4f}")
            print(f"Val loss:   {val_loss:.4f}")

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "step_num": self.step_num,
                        "model_config": model_config,
                        "vocab": self.gloss_vocab.to_dict() if hasattr(self.gloss_vocab, "to_dict") else None,
                    },
                    self.best_model_path,
                )
                print(f"✓ Saved best model: {self.best_model_path}")

            if epoch % 5 == 0 or epoch == num_epochs:
                bleu = self.compute_bleu("dev", max_samples=200)
                print(f"Val BLEU-4: {bleu:.2f}")
            else:
                bleu = 0.0

            append_csv(
                results_csv,
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_bleu4": bleu,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.step_num,
                    "nonfinite_skips": self._nonfinite_skips,
                },
            )

            if epoch % 10 == 0:
                ckpt = self.output_dir / f"checkpoint_epoch{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "step_num": self.step_num,
                        "model_config": model_config,
                        "vocab": self.gloss_vocab.to_dict() if hasattr(self.gloss_vocab, "to_dict") else None,
                    },
                    ckpt,
                )

        print("\nTraining complete.")
        print(f"Best checkpoint: {self.best_model_path}")
        return self.best_model_path


def main():
    parser = argparse.ArgumentParser(description="Train RGB-only video-to-ORTH model")

    parser.add_argument("--data_cache", type=str, default="../data_cache")
    parser.add_argument("--output_dir", type=str, default="runs/video_stage1")

    # model
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

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_factor", type=float, default=1.0, help="Multiplier for Noam LR schedule")
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision AMP")

    # system
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build vocab from TRAIN orth
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
        min_freq=1,
    )
    print(f"ORTH vocab size: {len(gloss_vocab.tokens)}")

    # datasets
    train_ds = PhoenixVideoDataset(args.data_cache, "train", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)
    dev_ds = PhoenixVideoDataset(args.data_cache, "dev", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)
    test_ds = PhoenixVideoDataset(args.data_cache, "test", gloss_vocab=gloss_vocab, use_rgb=True, strict=True)

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

    # model config for checkpointing
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

    trainer = VideoTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=dev_loader,
        test_loader=test_loader,
        gloss_vocab=gloss_vocab,
        device=device,
        output_dir=args.output_dir,
        lr_factor=args.lr_factor,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        label_smoothing=args.label_smoothing,
        use_amp=args.amp,
    )

    best = trainer.train(args.epochs, model_config=model_config)
    print(f"Best model: {best}")


if __name__ == "__main__":
    main()
