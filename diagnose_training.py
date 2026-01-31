"""
Diagnostic script to identify training issues.

This will check:
1. Data statistics and distribution
2. Vocabulary analysis
3. Model output diversity
4. Gradient flow
5. Loss curve analysis
"""

import json
import sys
from pathlib import Path
from collections import Counter
import torch
import numpy as np

def analyze_manifest(manifest_path):
    """Analyze the data manifest for potential issues."""
    print(f"\n{'='*80}")
    print(f"Analyzing manifest: {manifest_path}")
    print(f"{'='*80}")
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    successful = [s for s in data if s.get('success', False)]
    print(f"Total samples: {len(data)}")
    print(f"Successful: {len(successful)}")
    
    # Analyze orth sequences
    orths = [s['orth'] for s in successful if s.get('orth')]
    orth_lengths = [len(o.split()) for o in orths]
    
    print(f"\nORTH Statistics:")
    print(f"  Min length: {min(orth_lengths)}")
    print(f"  Max length: {max(orth_lengths)}")
    print(f"  Mean length: {np.mean(orth_lengths):.2f}")
    print(f"  Median length: {np.median(orth_lengths):.2f}")
    
    # Most common words
    all_words = []
    for orth in orths:
        all_words.extend(orth.split())
    
    word_counts = Counter(all_words)
    print(f"\nVocabulary size: {len(word_counts)}")
    print(f"Total tokens: {len(all_words)}")
    print(f"\nTop 20 most frequent words:")
    for word, count in word_counts.most_common(20):
        print(f"  {word}: {count}")
    
    # Check for extremely rare words
    rare_words = [w for w, c in word_counts.items() if c == 1]
    print(f"\nWords appearing only once: {len(rare_words)}")
    if len(rare_words) > 0:
        print(f"  Examples: {rare_words[:10]}")
    
    # Analyze frame counts
    frame_counts = [s.get('original_frames', 0) for s in successful]
    print(f"\nFrame Statistics:")
    print(f"  Min frames: {min(frame_counts)}")
    print(f"  Max frames: {max(frame_counts)}")
    print(f"  Mean frames: {np.mean(frame_counts):.2f}")
    
    return successful, word_counts


def analyze_vocab(vocab_path):
    """Analyze vocabulary file."""
    print(f"\n{'='*80}")
    print(f"Analyzing vocabulary: {vocab_path}")
    print(f"{'='*80}")
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    tokens = vocab_data['tokens']
    print(f"Vocabulary size: {len(tokens)}")
    print(f"Special tokens:")
    print(f"  PAD: {vocab_data.get('pad_id')} -> {tokens[vocab_data.get('pad_id', 0)]}")
    print(f"  UNK: {vocab_data.get('unk_id')} -> {tokens[vocab_data.get('unk_id', 1)]}")
    print(f"  BOS: {vocab_data.get('bos_id')} -> {tokens[vocab_data.get('bos_id', 2)] if vocab_data.get('bos_id') is not None else 'None'}")
    print(f"  EOS: {vocab_data.get('eos_id')} -> {tokens[vocab_data.get('eos_id', 3)] if vocab_data.get('eos_id') is not None else 'None'}")
    
    print(f"\nFirst 30 tokens (after specials):")
    for i, tok in enumerate(tokens[4:34]):
        print(f"  {i+4}: {tok}")
    
    return vocab_data


def analyze_predictions(results_path):
    """Analyze test results for patterns."""
    print(f"\n{'='*80}")
    print(f"Analyzing predictions: {results_path}")
    print(f"{'='*80}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    samples = results['samples']
    predictions = [s['prediction'] for s in samples]
    references = [s['reference'] for s in samples]
    
    # Check for repetition
    def has_repetition(text, min_repeat=3):
        words = text.split()
        if len(words) < min_repeat:
            return False
        for i in range(len(words) - min_repeat + 1):
            if len(set(words[i:i+min_repeat])) == 1:
                return True
        return False
    
    repetitive = sum(has_repetition(p) for p in predictions)
    print(f"Predictions with 3+ word repetition: {repetitive}/{len(predictions)} ({100*repetitive/len(predictions):.1f}%)")
    
    # Prediction diversity
    unique_preds = len(set(predictions))
    print(f"Unique predictions: {unique_preds}/{len(predictions)} ({100*unique_preds/len(predictions):.1f}%)")
    
    # Most common predictions
    pred_counts = Counter(predictions)
    print(f"\nTop 10 most common predictions:")
    for pred, count in pred_counts.most_common(10):
        print(f"  {count}x: {pred}")
    
    # Prediction length distribution
    pred_lengths = [len(p.split()) for p in predictions]
    print(f"\nPrediction lengths:")
    print(f"  Min: {min(pred_lengths)}")
    print(f"  Max: {max(pred_lengths)}")
    print(f"  Mean: {np.mean(pred_lengths):.2f}")
    print(f"  Median: {np.median(pred_lengths):.2f}")
    
    ref_lengths = [len(r.split()) for r in references]
    print(f"\nReference lengths:")
    print(f"  Min: {min(ref_lengths)}")
    print(f"  Max: {max(ref_lengths)}")
    print(f"  Mean: {np.mean(ref_lengths):.2f}")
    print(f"  Median: {np.median(ref_lengths):.2f}")
    
    # Show some examples
    print(f"\nExample predictions (first 10):")
    for i in range(min(10, len(samples))):
        print(f"\n{i+1}. {samples[i]['video_id']}")
        print(f"   PRED: {samples[i]['prediction']}")
        print(f"   REF:  {samples[i]['reference']}")


def check_training_log(log_path):
    """Analyze training log."""
    print(f"\n{'='*80}")
    print(f"Analyzing training log: {log_path}")
    print(f"{'='*80}")
    
    if not Path(log_path).exists():
        print("Training log not found!")
        return
    
    import csv
    
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No training data!")
        return
    
    epochs = [int(r['epoch']) for r in rows]
    train_losses = [float(r['train_loss']) for r in rows]
    val_losses = [float(r['val_loss']) for r in rows]
    
    print(f"Training epochs: {len(rows)}")
    print(f"\nTrain loss:")
    print(f"  Initial: {train_losses[0]:.4f}")
    print(f"  Final: {train_losses[-1]:.4f}")
    print(f"  Best: {min(train_losses):.4f}")
    print(f"  Change: {train_losses[-1] - train_losses[0]:.4f}")
    
    print(f"\nVal loss:")
    print(f"  Initial: {val_losses[0]:.4f}")
    print(f"  Final: {val_losses[-1]:.4f}")
    print(f"  Best: {min(val_losses):.4f}")
    print(f"  Change: {val_losses[-1] - val_losses[0]:.4f}")
    
    # Check for overfitting
    train_val_gap = [train_losses[i] - val_losses[i] for i in range(len(rows))]
    print(f"\nTrain-Val gap (final): {train_val_gap[-1]:.4f}")
    
    # Check if loss is stuck
    if len(train_losses) > 10:
        recent_std = np.std(train_losses[-10:])
        print(f"Recent loss std (last 10 epochs): {recent_std:.6f}")
        if recent_std < 0.001:
            print("  WARNING: Loss appears stuck!")


def main():
    data_cache = Path("../data_cache")
    run_dir = Path("../runs/video_stage1")
    
    # Analyze training data
    train_manifest = data_cache / "phoenix2014t" / "manifests" / "train_rgb_manifest.json"
    dev_manifest = data_cache / "phoenix2014t" / "manifests" / "dev_rgb_manifest.json"
    
    if train_manifest.exists():
        analyze_manifest(train_manifest)
    
    if dev_manifest.exists():
        analyze_manifest(dev_manifest)
    
    # Analyze vocabulary
    vocab_path = run_dir / "gloss_vocab.json"
    if vocab_path.exists():
        analyze_vocab(vocab_path)
    
    # Analyze predictions
    results_path = run_dir / "test_results.json"
    if results_path.exists():
        analyze_predictions(results_path)
    
    # Analyze training log
    log_path = run_dir / "training_log.csv"
    if log_path.exists():
        check_training_log(log_path)
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    print("""
1. If predictions are highly repetitive:
   - Increase temperature during generation
   - Add nucleus sampling (top-p)
   - Check for mode collapse
   
2. If loss is stuck or not decreasing:
   - Reduce learning rate
   - Increase warmup steps
   - Check gradient flow
   - Verify data preprocessing
   
3. If high train-val gap:
   - Add more dropout
   - Reduce model capacity
   - Add data augmentation
   
4. If vocabulary has many rare words:
   - Increase min_freq threshold
   - Consider subword tokenization
   
5. General improvements:
   - Use beam search instead of greedy
   - Add attention visualization
   - Implement length penalty
   - Try different backbones (ResNet50, EfficientNet)
    """)


if __name__ == "__main__":
    main()