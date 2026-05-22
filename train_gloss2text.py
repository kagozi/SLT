"""
train_gloss2text.py
Fine-tunes mBART-50 on PHOENIX-2014-T gloss → German translation pairs.

Replicates the Gloss2Text component of Maia et al. (Sign2German, 2025):
  input:  space-separated PHOENIX gloss sequence   (e.g. "HEUTE WETTER GUT")
  output: German translation sentence              (e.g. "Heute ist das Wetter gut.")

This is a pure text-to-text model — no keypoints, no visual encoder.
It represents the upper bound on what is achievable from perfect gloss
recognition: given gold gloss sequences, how well can we recover the German?

Compare against:
  Exp 23 (flat CTC-on-text, no glosses):       BLEU-4 13.96
  Exp 25 (multistream CTC-on-text, no glosses): BLEU-4 14.30
  Exp 32 (multistream Sign2Gloss2Text):         Trans. BLEU-4 ~16+
  Exp 33 (this, gold-gloss Gloss2Text oracle):  expected higher — glosses are gold
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import wandb
from sacrebleu.metrics import BLEU
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

BART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"
SRC_LANG   = "de_DE"   # treat glosses as German-derived tokens
TGT_LANG   = "de_DE"
BEAM_WIDTHS = [1, 2, 4, 8, 16, 32]


# ── Dataset ──────────────────────────────────────────────────────────────────

class GlossTranslationDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs  # list of (gloss_str, translation_str)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def load_phoenix_pairs(root_dir: str, split: str):
    csv = (
        Path(root_dir) / "annotations" / "manual"
        / f"PHOENIX-2014-T.{split}.corpus.csv"
    )
    df = pd.read_csv(csv, sep="|")
    pairs = []
    for _, row in df.iterrows():
        gloss = str(row["orth"]).strip()
        trans  = str(row["translation"]).strip()
        if gloss and trans and gloss != "nan" and trans != "nan":
            pairs.append((gloss, trans))
    return pairs


def make_collate(tokenizer, max_src_len, max_tgt_len):
    def collate(batch):
        glosses = [b[0] for b in batch]
        translations = [b[1] for b in batch]

        tokenizer.src_lang = SRC_LANG
        enc = tokenizer(
            glosses,
            max_length=max_src_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with tokenizer.as_target_tokenizer():
            tgt = tokenizer(
                translations,
                max_length=max_tgt_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
        labels = tgt["input_ids"].masked_fill(
            tgt["input_ids"] == tokenizer.pad_token_id, -100
        )
        enc["labels"] = labels
        return enc
    return collate


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, tokenizer, device, beam_width=4, max_tgt_len=128):
    model.eval()
    forced_bos = tokenizer.lang_code_to_id[TGT_LANG]
    all_preds, all_refs = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_width,
            max_length=max_tgt_len,
            forced_bos_token_id=forced_bos,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
        for ids in generated:
            all_preds.append(tokenizer.decode(ids, skip_special_tokens=True))

    all_refs = [pair[1] for pair in loader.dataset.pairs]

    bleu = BLEU(tokenize="13a", lowercase=True)
    result = bleu.corpus_score(all_preds, [all_refs])
    return {
        "BLEU-1": result.precisions[0],
        "BLEU-4": result.precisions[3],
        "BLEU":   result.score,
        "preds":  all_preds,
        "refs":   all_refs,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gloss→German mBART fine-tuning")
    parser.add_argument("--root_dir", type=str,
        default="/data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T")
    parser.add_argument("--output_dir",  type=str, default="/data/experiments")
    parser.add_argument("--exp_name",    type=str,
        default="exp33_phoenix_gloss2text_maia")
    parser.add_argument("--epochs",      type=int,   default=60)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--grad_accum",  type=int,   default=4)
    parser.add_argument("--beam_width",  type=int,   default=10)
    parser.add_argument("--max_src_len", type=int,   default=64)
    parser.add_argument("--max_tgt_len", type=int,   default=128)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--fp16",        action="store_true")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results_dir = Path(args.output_dir) / "results" / args.exp_name
    models_dir  = Path(args.output_dir) / "models"  / args.exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("WANDB_DELETE_LOCAL", "1")
    wandb.init(project="slt", name=args.exp_name, config=vars(args))

    print("Loading mBART tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(BART_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BART_MODEL)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device)
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    print("Loading PHOENIX gloss→German pairs...")
    train_pairs = load_phoenix_pairs(args.root_dir, "train")
    val_pairs   = load_phoenix_pairs(args.root_dir, "dev")
    test_pairs  = load_phoenix_pairs(args.root_dir, "test")
    print(f"  train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    collate = make_collate(tokenizer, args.max_src_len, args.max_tgt_len)
    train_loader = DataLoader(GlossTranslationDataset(train_pairs),
                              args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=args.num_workers)
    val_loader   = DataLoader(GlossTranslationDataset(val_pairs),
                              args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=args.num_workers)
    test_loader  = DataLoader(GlossTranslationDataset(test_pairs),
                              args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps   = len(train_loader) * args.epochs // max(1, args.grad_accum)
    warmup_steps  = total_steps // 10
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler() if args.fp16 else None

    best_val_bleu4    = 0.0
    epochs_no_improve = 0
    best_ckpt         = models_dir / f"best_{args.exp_name}.pt"

    print(f"\n🏋️  Training for {args.epochs} epochs  "
          f"(effective batch={args.batch_size * args.grad_accum})...")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            if scaler:
                with torch.cuda.amp.autocast():
                    out = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
                    loss = out.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out  = model(input_ids=input_ids,
                             attention_mask=attention_mask, labels=labels)
                loss = out.loss / args.grad_accum
                loss.backward()

            total_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = total_loss / len(train_loader)
        val_m    = evaluate(model, val_loader, tokenizer, device,
                            beam_width=4, max_tgt_len=args.max_tgt_len)

        print(f"  Epoch {epoch+1:3d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  val_BLEU-4={val_m['BLEU-4']:.2f}")
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss,
                   "val/BLEU-4": val_m["BLEU-4"], "val/BLEU-1": val_m["BLEU-1"]})

        if val_m["BLEU-4"] > best_val_bleu4:
            best_val_bleu4    = val_m["BLEU-4"]
            epochs_no_improve = 0
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch + 1,
                        "val_BLEU-4": val_m["BLEU-4"],
                        "args": vars(args)}, best_ckpt)
            print(f"    ✅ New best val BLEU-4 = {best_val_bleu4:.2f}")
        else:
            epochs_no_improve += 1
            if (args.early_stop_patience > 0
                    and epochs_no_improve >= args.early_stop_patience):
                print(f"  Early stopping at epoch {epoch + 1}.")
                break

    # ── Load best checkpoint ──────────────────────────────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"\n  📥 Best checkpoint: epoch {ckpt['epoch']}, "
          f"val BLEU-4={ckpt['val_BLEU-4']:.2f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    test_m = evaluate(model, test_loader, tokenizer, device,
                      beam_width=args.beam_width, max_tgt_len=args.max_tgt_len)
    print(f"\n{'='*60}")
    print("📊 TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  BLEU-1:  {test_m['BLEU-1']:.4f}")
    print(f"  BLEU-4:  {test_m['BLEU-4']:.4f}")
    print(f"  Corpus BLEU: {test_m['BLEU']:.4f}")
    print(f"{'='*60}")
    wandb.log({"test/BLEU-1": test_m["BLEU-1"], "test/BLEU-4": test_m["BLEU-4"]})

    # ── Beam sweep ────────────────────────────────────────────────────────────
    print(f"\n🔍 Beam sweep — {args.exp_name}")
    sweep_table = wandb.Table(columns=["split", "beam_width", "BLEU-1", "BLEU-4"])
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        for bw in BEAM_WIDTHS:
            m = evaluate(model, loader, tokenizer, device,
                         beam_width=bw, max_tgt_len=args.max_tgt_len)
            sweep_table.add_data(split_name, bw,
                                 round(m["BLEU-1"], 4), round(m["BLEU-4"], 4))
            print(f"  [{split_name}] beam={bw:2d}  BLEU-4={m['BLEU-4']:.4f}")
    wandb.log({"beam_sweep/table": sweep_table})
    print("  ✅ Beam sweep logged to W&B.")

    # ── Save artifacts ────────────────────────────────────────────────────────
    pd.DataFrame({
        "gloss":      [p[0] for p in test_pairs],
        "reference":  test_m["refs"],
        "prediction": test_m["preds"],
    }).to_csv(results_dir / f"predictions_{args.exp_name}.csv", index=False)

    with open(results_dir / f"metrics_{args.exp_name}.json", "w") as f:
        json.dump({
            "experiment": args.exp_name,
            "test_BLEU-1": test_m["BLEU-1"],
            "test_BLEU-4": test_m["BLEU-4"],
            "args": vars(args),
        }, f, indent=2)

    wandb.finish()
    print(f"\n✨ {args.exp_name} complete!  Test BLEU-4 = {test_m['BLEU-4']:.2f}")


if __name__ == "__main__":
    main()
