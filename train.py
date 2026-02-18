import time
import pandas as pd
from collections import Counter
import math
from tabulate import tabulate
import json
from dataset import PhoenixSignDataset
from models import SignLanguageTransformer
import numpy as np
import torch
from utils import GlossTokenizer, Trainer, collate_fn
from torch.utils.data import DataLoader
import random
from pathlib import Path


class SignLanguageEvaluator:
    """Comprehensive evaluation metrics for sign language recognition"""
    
    def __init__(self, model, test_loader, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.device = device
        self.wer_costs = {'del': 3, 'ins': 3, 'sub': 4}
        
    @torch.no_grad()
    def evaluate(self, beam_width=5, verbose=True):
        self.model.eval()
        all_predictions, all_targets, all_translations, all_names = [], [], [], []
        inference_times = []
        
        for batch_idx, batch in enumerate(self.test_loader):
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            start_time = time.time()
            logits, mask_out = self.model(keypoints, mask)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            logits = logits.cpu()
            mask_out = mask_out.cpu()
            input_lengths = mask_out.sum(dim=1).long()
            
            predictions = self._greedy_decode(logits, input_lengths)
            
            all_predictions.extend(predictions)
            all_targets.extend(batch['gloss_text'])
            all_translations.extend(batch['translation'])
            all_names.extend(batch['name'])
            
            if verbose and batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(self.test_loader)}")
        
        metrics = self.calculate_all_metrics(
            all_predictions, all_targets, all_translations, 
            inference_times, all_names
        )
        return metrics
    
    def _greedy_decode(self, logits, input_lengths):
        predictions = []
        for i in range(logits.shape[0]):
            valid_len = min(input_lengths[i].item(), logits.shape[1])
            pred = logits[i, :valid_len].argmax(dim=-1).numpy()
            unique_pred = []
            prev = -1
            for p in pred:
                if p != prev:
                    if p != 0:
                        unique_pred.append(p)
                    prev = p
            pred_text = self.tokenizer.decode(unique_pred)
            predictions.append(pred_text)
        return predictions
    
    def calculate_bleu(self, predictions, targets, max_n=4):
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        
        bleu_scores = {}
        for n in range(1, max_n+1):
            total_precision = 0
            valid_sentences = 0
            for pred, target in zip(predictions, targets):
                pred_tokens = pred.split()
                target_tokens = target.split()
                if len(pred_tokens) < n or len(target_tokens) < n:
                    continue
                pred_ngrams = Counter(get_ngrams(pred_tokens, n))
                target_ngrams = Counter(get_ngrams(target_tokens, n))
                matches = sum((pred_ngrams & target_ngrams).values())
                total_pred = sum(pred_ngrams.values())
                if total_pred > 0:
                    total_precision += matches / total_pred
                    valid_sentences += 1
            bleu_scores[f'BLEU-{n}'] = total_precision / valid_sentences if valid_sentences > 0 else 0.0
        
        if all(v > 0 for v in bleu_scores.values()):
            log_sum = sum(math.log(v) for v in bleu_scores.values())
            bleu_scores['BLEU'] = math.exp(log_sum / len(bleu_scores))
        else:
            bleu_scores['BLEU'] = 0.0
        return bleu_scores
    
    def calculate_wer(self, predictions, targets):
        def wer_single(pred, target):
            pred_words = pred.split()
            target_words = target.split()
            if not target_words:
                return 0.0
            d = np.zeros((len(target_words) + 1, len(pred_words) + 1))
            for i in range(len(target_words) + 1):
                d[i, 0] = i * self.wer_costs['del']
            for j in range(len(pred_words) + 1):
                d[0, j] = j * self.wer_costs['ins']
            for i in range(1, len(target_words) + 1):
                for j in range(1, len(pred_words) + 1):
                    cost = 0 if target_words[i-1] == pred_words[j-1] else self.wer_costs['sub']
                    d[i, j] = min(
                        d[i-1, j] + self.wer_costs['del'],
                        d[i, j-1] + self.wer_costs['ins'],
                        d[i-1, j-1] + cost
                    )
            max_cost = max(self.wer_costs.values()) * max(len(target_words), len(pred_words))
            return d[len(target_words), len(pred_words)] / max_cost if max_cost > 0 else 0.0
        
        return np.mean([wer_single(p, t) for p, t in zip(predictions, targets)])
    
    def calculate_all_metrics(self, predictions, targets, translations, inference_times, names):
        metrics = {
            'total_samples': len(predictions),
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'fps': 1.0 / np.mean(inference_times) if inference_times else 0,
        }
        bleu_scores = self.calculate_bleu(predictions, targets, max_n=4)
        metrics.update(bleu_scores)
        metrics['WER'] = self.calculate_wer(predictions, targets)
        exact_matches = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
        metrics['exact_match_accuracy'] = exact_matches / len(predictions) if predictions else 0
        
        self.save_detailed_results(predictions, targets, translations, names, metrics)
        return metrics
    
    def save_detailed_results(self, predictions, targets, translations, names, metrics):
        results_df = pd.DataFrame({
            'sequence_name': names,
            'target_gloss': targets,
            'predicted_gloss': predictions,
            'german_translation': translations,
            'correct': [p.strip() == t.strip() for p, t in zip(predictions, targets)]
        })
        results_df.to_csv('test_predictions.csv', index=False)
        
        metrics_ser = {}
        for k, v in metrics.items():
            if isinstance(v, (np.floating, np.integer)):
                metrics_ser[k] = float(v)
            elif isinstance(v, dict):
                metrics_ser[k] = {sk: float(sv) if isinstance(sv, (np.floating, np.integer)) else sv 
                                  for sk, sv in v.items()}
            else:
                metrics_ser[k] = v
        with open('test_metrics.json', 'w') as f:
            json.dump(metrics_ser, f, indent=2)
        print(f"\n✅ Saved test_predictions.csv and test_metrics.json")
    
    def print_metrics_table(self, metrics):
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        print(f"  Samples: {metrics['total_samples']}")
        print(f"  WER:     {metrics['WER']:.2%}")
        print(f"  BLEU-1:  {metrics.get('BLEU-1', 0):.4f}")
        print(f"  BLEU-4:  {metrics.get('BLEU-4', 0):.4f}")
        print(f"  Exact:   {metrics.get('exact_match_accuracy', 0):.2%}")
        print("="*60)


AdvancedEvaluator = SignLanguageEvaluator  # alias for compatibility


def main():
    # Configuration
    root_dir = '../phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/'
    batch_size = 20
    max_frames = 250
    num_epochs = 100
    
    # ─── Step 1: Build WORD-LEVEL tokenizer from ALL splits ───
    print("📂 Building word-level gloss tokenizer...")
    all_gloss_sentences = []
    for split in ['train', 'dev', 'test']:
        csv_path = Path(root_dir) / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
        df = pd.read_csv(csv_path, sep='|')
        all_gloss_sentences.extend(df['orth'].dropna().tolist())
    
    tokenizer = GlossTokenizer(all_gloss_sentences, min_freq=1)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    
    # ─── Step 2: Create datasets WITH the tokenizer ───
    print("📂 Loading datasets...")
    train_dataset = PhoenixSignDataset(root_dir, split='train', max_frames=max_frames, 
                                        augment=True, tokenizer=tokenizer)
    val_dataset = PhoenixSignDataset(root_dir, split='dev', max_frames=max_frames, 
                                      augment=False, tokenizer=tokenizer)
    test_dataset = PhoenixSignDataset(root_dir, split='test', max_frames=max_frames, 
                                       augment=False, tokenizer=tokenizer)
    
    # ─── Step 3: Verify tokenization works ───
    sample = train_dataset[0]
    print(f"\n  Sample check:")
    print(f"    Gloss text:  '{sample['gloss_text']}'")
    print(f"    Token ids:   {sample['gloss'].tolist()}")
    print(f"    Decoded:     '{tokenizer.decode(sample['gloss'])}'")
    print(f"    Keypoints:   {sample['keypoints'].shape}")
    print(f"    Num frames:  {sample['num_frames']}")
    
    input_dim = sample['keypoints'].shape[-1]
    print(f"    Input dim:   {input_dim}")
    
    # ─── Step 4: DataLoaders ───
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # ─── Step 5: Model ───
    model = SignLanguageTransformer(
        input_dim=input_dim,
        dim=192,
        num_classes=tokenizer.vocab_size,
        max_frames=max_frames * 2,  # safety for PE buffer
        dropout=0.2
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🤖 Model: {total_params:,} parameters")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    # ─── Step 6: Train ───
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    trainer = Trainer(model, train_loader, val_loader, tokenizer, device=device)
    
    print("\n🏋️ Starting training...")
    trainer.train(num_epochs=num_epochs)
    
    # ─── Step 7: Save ───
    model_path = 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.gloss_to_idx,
        'config': {
            'input_dim': input_dim,
            'dim': 192,
            'num_classes': tokenizer.vocab_size,
            'max_frames': max_frames
        }
    }, model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # ─── Step 8: Evaluate ───
    print("\n🎯 FINAL TEST SET EVALUATION")
    test_evaluator = SignLanguageEvaluator(model, test_loader, tokenizer, device)
    test_metrics = test_evaluator.evaluate(beam_width=1, verbose=True)
    test_evaluator.print_metrics_table(test_metrics)
    
    # Show sample predictions
    print("\n🔍 Sample predictions:")
    sample_df = pd.read_csv('test_predictions.csv').head(10)
    for _, row in sample_df.iterrows():
        match = "✅" if row['correct'] else "❌"
        print(f"  {match} Target: {row['target_gloss']}")
        print(f"     Pred:   {row['predicted_gloss']}")
        print()
    
    return test_metrics


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    metrics = main()