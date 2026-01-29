# debug_data.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.phoenix_dataset import PhoenixVideoDataset, collate_video_batch
from data.vocab import build_word_vocab
import json
import torch

# Load vocab
manifest_path = Path("../data_cache/phoenix2014t/manifests/train_rgb_manifest.json")
with open(manifest_path) as f:
    manifest = json.load(f)

orths = [s["orth"] for s in manifest if s.get("success") and str(s.get("orth", "")).strip() not in ("", "-1")]
vocab = build_word_vocab(orths, specials=["<pad>", "<unk>", "<start>", "<end>"], min_freq=1)

print(f"Vocab size: {len(vocab.tokens)}")
print(f"pad_id: {vocab.pad_id}, bos_id: {vocab.bos_id}, eos_id: {vocab.eos_id}")

# Check dataset
dataset = PhoenixVideoDataset("../data_cache", "train", gloss_vocab=vocab, use_rgb=True, strict=True)
print(f"Dataset size: {len(dataset)}")

# Check a batch
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=8, shuffle=False, 
                    collate_fn=lambda b: collate_video_batch(b, pad_id=vocab.pad_id))

batch = next(iter(loader))
print(f"\nBatch RGB: {batch['rgb'].shape}")
print(f"Batch orth_ids: {batch['orth_ids'].shape}")
print(f"orth_ids min: {batch['orth_ids'].min()}, max: {batch['orth_ids'].max()}")
print(f"Unique tokens in batch: {torch.unique(batch['orth_ids'])}")

# Check for invalid token IDs
if batch['orth_ids'].max() >= len(vocab.tokens):
    print(f"⚠️  ERROR: Token ID {batch['orth_ids'].max()} >= vocab size {len(vocab.tokens)}")
else:
    print(f"✓ All token IDs valid")

# Check RGB values
print(f"\nRGB stats:")
print(f"  min: {batch['rgb'].min():.4f}, max: {batch['rgb'].max():.4f}")
print(f"  mean: {batch['rgb'].mean():.4f}, std: {batch['rgb'].std():.4f}")

if batch['rgb'].min() < -5 or batch['rgb'].max() > 5:
    print("⚠️  WARNING: RGB values might not be normalized correctly")
else:
    print("✓ RGB values look reasonable")