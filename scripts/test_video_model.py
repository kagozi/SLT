"""
Test trained RGB-only video-to-ORTH model with BEAM SEARCH.

Usage:
    python scripts/test_video_model.py \
        --checkpoint runs/video_stage1/best_model.pt \
        --data_cache ../data_cache \
        --split test \
        --output ../runs/video_stage1/test_results.json \
        --beam_size 5
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

from beam_search import add_beam_search_to_model


def load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_config" not in ckpt or "vocab" not in ckpt:
        raise ValueError(
            "Checkpoint missing 'model_config' or 'vocab'. "
            "Re-train using the corrected train script that saves them."
        )
    if "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint missing 'model_state_dict'.")
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
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, vocab, cfg


@torch.no_grad()
def generate_predictions(
    model, 
    dataloader, 
    vocab, 
    device, 
    max_samples=None,
    use_beam_search=False,
    beam_size=5,
    length_penalty=0.6,
    repetition_penalty=1.2,
):
    """
    Generate predictions using either greedy or beam search decoding.
    
    Args:
        use_beam_search: If True, use beam search. If False, use greedy.
        beam_size: Beam width for beam search
        length_penalty: Length normalization alpha
        repetition_penalty: Penalty for repeated tokens
    """
    preds, refs, vids = [], [], []
    
    # ✅ ADDED: Setup beam search if requested
    if use_beam_search:
        print(f"Using beam search (beam_size={beam_size}, "
              f"length_penalty={length_penalty}, repetition_penalty={repetition_penalty})")
        add_beam_search_to_model(model, vocab)
    else:
        print("Using greedy decoding")

    for batch in tqdm(dataloader, desc="Generating"):
        rgb = batch["rgb"].to(device, non_blocking=True)
        src_mask = batch["src_key_padding_mask"].to(device, non_blocking=True)

        # ✅ MODIFIED: Choose decoding strategy
        if use_beam_search:
            gen = model.beam_search(
                rgb=rgb,
                src_key_padding_mask=src_mask,
                beam_size=beam_size,
                max_len=100,
                length_penalty=length_penalty,
                repetition_penalty=repetition_penalty,
            )
        else:
            # Original greedy decoding
            gen = model.generate(
                rgb=rgb,
                src_key_padding_mask=src_mask,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                max_len=100,
            )

        for i in range(rgb.size(0)):
            ids = gen[i].tolist()
            
            # Remove BOS if present
            if vocab.bos_id is not None and len(ids) > 0 and ids[0] == vocab.bos_id:
                ids = ids[1:]
            
            # Stop at EOS
            if vocab.eos_id is not None and vocab.eos_id in ids:
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

    # ✅ ADDED: Beam search arguments
    parser.add_argument("--beam_size", type=int, default=0, 
                        help="Beam size (0 = greedy, >0 = beam search)")
    parser.add_argument("--length_penalty", type=float, default=0.6,
                        help="Length penalty for beam search (0.6-0.8 typical)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="Repetition penalty (1.0 = no penalty, 1.2-1.5 typical)")

    # Support BOTH flags (alias)
    parser.add_argument("--save_json", type=str, default=None, help="Path to save detailed JSON")
    parser.add_argument("--output", type=str, default=None, help="Alias for --save_json")

    args = parser.parse_args()
    device = torch.device(args.device)

    # Unify save flag
    save_path = args.save_json or args.output

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
        persistent_workers=(args.num_workers > 0),
    )

    # ✅ MODIFIED: Pass beam search parameters
    use_beam = args.beam_size > 0
    preds, refs, vids = generate_predictions(
        model, 
        dl, 
        vocab, 
        device, 
        max_samples=args.max_samples,
        use_beam_search=use_beam,
        beam_size=args.beam_size if use_beam else 5,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    metrics = compute_all_metrics(preds, refs)

    print("\n" + "=" * 80)
    print(f"Results on {args.split} ({len(preds)} samples)")
    if use_beam:
        print(f"Decoding: Beam Search (beam_size={args.beam_size})")
    else:
        print("Decoding: Greedy")
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
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "num_samples": len(preds),
                    "model_config": cfg,
                    "decoding": {
                        "method": "beam_search" if use_beam else "greedy",
                        "beam_size": args.beam_size if use_beam else None,
                        "length_penalty": args.length_penalty if use_beam else None,
                        "repetition_penalty": args.repetition_penalty if use_beam else None,
                    },
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