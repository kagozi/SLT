"""
Test trained RGB-only video-to-ORTH model.

Usage:
    python scripts/test_video_model.py \
        --checkpoint runs/video_stage1/best_model.pt \
        --data_cache ../data_cache \
        --split test
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
from data.vocab import Vocab, decode as decode_vocab
from models.video_to_gloss import VideoToGlossModel
from evaluation.metrics import compute_all_metrics


def load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_config" not in ckpt or "vocab" not in ckpt:
        raise ValueError(
            "Checkpoint missing 'model_config' or 'vocab'. "
            "Re-train using the corrected train script that saves them."
        )
    return ckpt


def build_model_from_ckpt(ckpt: dict, device: torch.device):
    cfg = ckpt["model_config"]
    vocab = Vocab.from_dict(ckpt["vocab"])

    model = VideoToGlossModel(
        gloss_vocab_size=cfg["vocab_size"],
        pad_id=cfg["pad_id"],
        d_model=cfg["d_model"],
        encoder_backbone=cfg["encoder_backbone"],
        encoder_pretrained=False,  # weights loaded
        encoder_temporal_layers=cfg["encoder_layers"],
        encoder_nhead=cfg["nhead"],
        decoder_layers=cfg["decoder_layers"],
        decoder_nhead=cfg["nhead"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, vocab, cfg


@torch.no_grad()
def generate_predictions(model, dataloader, vocab, device, max_samples=None):
    preds, refs, vids = [], [], []

    for batch in tqdm(dataloader, desc="Generating"):
        rgb = batch["rgb"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)

        gen = model.generate(
            rgb=rgb,
            src_key_padding_mask=src_mask,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            max_len=100,
        )

        for i in range(rgb.size(0)):
            ids = gen[i].tolist()
            if vocab.eos_id in ids:
                ids = ids[: ids.index(vocab.eos_id)]
            pred = decode_vocab(ids, vocab, skip_special=True)
            ref = batch["orth_texts"][i]

            preds.append(pred)
            refs.append(ref)
            vids.append(batch["video_ids"][i])

            if max_samples and len(preds) >= max_samples:
                return preds, refs, vids

    return preds, refs, vids


def main():
    parser = argparse.ArgumentParser(description="Test RGB-only video-to-ORTH model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_cache", type=str, default="../data_cache")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", type=str, default=None)

    args = parser.parse_args()
    device = torch.device(args.device)

    ckpt = load_checkpoint(args.checkpoint, device)
    model, vocab, cfg = build_model_from_ckpt(ckpt, device)

    ds = PhoenixVideoDataset(
        data_cache_dir=args.data_cache,
        split=args.split,
        gloss_vocab=vocab,
        use_rgb=True,
        use_keypoints=False,
        use_hands=False,
        strict=True,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_video_batch(b, pad_id=vocab.pad_id),
        pin_memory=(device.type == "cuda"),
    )

    preds, refs, vids = generate_predictions(model, dl, vocab, device, max_samples=args.max_samples)

    metrics = compute_all_metrics(preds, refs)

    print("\n" + "=" * 80)
    print(f"Results on {args.split} ({len(preds)} samples)")
    print("=" * 80)
    for k, v in metrics.items():
        if isinstance(v, float):
            if "exact_match" in k:
                print(f"{k:12s}: {v*100:.2f}%")
            else:
                print(f"{k:12s}: {v:.4f}")
        else:
            print(f"{k:12s}: {v}")
    print("=" * 80)

    # Save outputs if requested
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "num_samples": len(preds),
                    "model_config": cfg,
                    "metrics": metrics,
                    "samples": [
                        {"video_id": vid, "prediction": p, "reference": r}
                        for vid, p, r in zip(vids, preds, refs)
                    ],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved detailed results to: {out}")


if __name__ == "__main__":
    main()
