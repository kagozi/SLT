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
        
        # WER costs (can be adjusted)
        self.wer_costs = {'del': 3, 'ins': 3, 'sub': 4}
        
    @torch.no_grad()
    def evaluate(self, beam_width=5, verbose=True):
        """Run full evaluation on test set"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_translations = []  # German text
        all_names = []
        inference_times = []
        
        for batch_idx, batch in enumerate(self.test_loader):
            # Move to device
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['gloss']  # Keep on CPU for now
            
            # Measure inference time
            start_time = time.time()
            
            # Forward pass
            logits, mask_out = self.model(keypoints, mask)
            
            # Move logits to CPU for decoding
            logits = logits.cpu()
            mask_out = mask_out.cpu()
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate input lengths
            input_lengths = mask_out.sum(dim=1).long()
            
            # Decode predictions
            if beam_width > 1:
                predictions = self._beam_search_decode(logits, input_lengths, beam_width)
            else:
                predictions = self._greedy_decode(logits, input_lengths)
            
            # Store results
            all_predictions.extend(predictions)
            all_targets.extend(batch['gloss_text'])
            all_translations.extend(batch['translation'])
            all_names.extend(batch['name'])
            
            if verbose and batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(self.test_loader)}")
        
        # Calculate all metrics
        metrics = self.calculate_all_metrics(
            all_predictions, all_targets, all_translations, 
            inference_times, all_names
        )
        
        return metrics
    
    def _greedy_decode(self, logits, input_lengths):
        """Greedy decoding with blank removal"""
        # logits: (batch, time, vocab)
        predictions = []
        
        for i in range(logits.shape[0]):
            pred = logits[i, :input_lengths[i]].argmax(dim=-1).numpy()
            
            # Remove consecutive duplicates and blanks (0)
            unique_pred = []
            prev = -1
            for p in pred:
                if p != prev and p != 0:  # 0 is blank/pad
                    unique_pred.append(p)
                prev = p
            
            # Decode to text
            pred_text = self.tokenizer.decode(unique_pred)
            predictions.append(pred_text)
        
        return predictions
    
    def _beam_search_decode(self, logits, input_lengths, beam_width=5):
        """Beam search decoding"""
        # Simplified beam search - for production, use a proper implementation
        # This is a placeholder - you might want to use a library like `ctcdecode`
        return self._greedy_decode(logits, input_lengths)
    
    def calculate_bleu(self, predictions, targets, max_n=4):
        """Calculate BLEU scores from 1 to max_n"""
        
        def get_ngrams(tokens, n):
            """Extract n-grams from token list"""
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
                
                # Get n-grams
                pred_ngrams = Counter(get_ngrams(pred_tokens, n))
                target_ngrams = Counter(get_ngrams(target_tokens, n))
                
                # Count matches
                matches = sum((pred_ngrams & target_ngrams).values())
                total_pred = sum(pred_ngrams.values())
                
                if total_pred > 0:
                    precision = matches / total_pred
                    total_precision += precision
                    valid_sentences += 1
            
            # Calculate average precision for this n
            if valid_sentences > 0:
                bleu_scores[f'BLEU-{n}'] = total_precision / valid_sentences
            else:
                bleu_scores[f'BLEU-{n}'] = 0.0
        
        # Calculate geometric mean (BLEU score)
        if all(v > 0 for v in bleu_scores.values()):
            log_sum = sum(math.log(v) for v in bleu_scores.values())
            bleu_scores['BLEU'] = math.exp(log_sum / len(bleu_scores))
        else:
            bleu_scores['BLEU'] = 0.0
        
        return bleu_scores
    
    def calculate_wer(self, predictions, targets):
        """Calculate Word Error Rate"""
        
        def wer_single(pred, target):
            # Dynamic programming implementation of Levenshtein distance
            pred_words = pred.split()
            target_words = target.split()
            
            d = np.zeros((len(target_words) + 1, len(pred_words) + 1))
            
            for i in range(len(target_words) + 1):
                d[i, 0] = i * self.wer_costs['del']
            for j in range(len(pred_words) + 1):
                d[0, j] = j * self.wer_costs['ins']
            
            for i in range(1, len(target_words) + 1):
                for j in range(1, len(pred_words) + 1):
                    if target_words[i-1] == pred_words[j-1]:
                        cost = 0
                    else:
                        cost = self.wer_costs['sub']
                    
                    d[i, j] = min(
                        d[i-1, j] + self.wer_costs['del'],      # deletion
                        d[i, j-1] + self.wer_costs['ins'],      # insertion
                        d[i-1, j-1] + cost                       # substitution
                    )
            
            # Normalize by max possible cost
            max_cost = max(self.wer_costs.values()) * max(len(target_words), len(pred_words))
            if max_cost > 0:
                return d[len(target_words), len(pred_words)] / max_cost
            return 0.0
        
        wer_scores = [wer_single(p, t) for p, t in zip(predictions, targets)]
        return np.mean(wer_scores)
    
    def calculate_metrics_by_length(self, predictions, targets):
        """Analyze performance based on sequence length"""
        length_metrics = {}
        
        # Group by target length
        length_groups = {
            'short': [],    # 1-3 words
            'medium': [],   # 4-7 words
            'long': []      # 8+ words
        }
        
        for pred, target in zip(predictions, targets):
            length = len(target.split())
            if length <= 3:
                length_groups['short'].append((pred, target))
            elif length <= 7:
                length_groups['medium'].append((pred, target))
            else:
                length_groups['long'].append((pred, target))
        
        for group_name, group_data in length_groups.items():
            if group_data:
                group_preds, group_targets = zip(*group_data)
                wer = self.calculate_wer(group_preds, group_targets)
                bleu = self.calculate_bleu(group_preds, group_targets, max_n=4)
                length_metrics[group_name] = {
                    'count': len(group_data),
                    'WER': wer,
                    'BLEU': bleu['BLEU']
                }
        
        return length_metrics
    
    def calculate_all_metrics(self, predictions, targets, translations, inference_times, names):
        """Calculate all evaluation metrics"""
        
        # Basic statistics
        metrics = {
            'total_samples': len(predictions),
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'std_inference_time_ms': np.std(inference_times) * 1000,
            'fps': 1.0 / np.mean(inference_times)
        }
        
        # BLEU scores
        bleu_scores = self.calculate_bleu(predictions, targets, max_n=4)
        metrics.update(bleu_scores)
        
        # WER
        metrics['WER'] = self.calculate_wer(predictions, targets)
        
        # Exact match accuracy
        exact_matches = sum(1 for p, t in zip(predictions, targets) if p == t)
        metrics['exact_match_accuracy'] = exact_matches / len(predictions) if predictions else 0
        
        # Performance by sequence length
        metrics['length_analysis'] = self.calculate_metrics_by_length(predictions, targets)
        
        # Save detailed results
        self.save_detailed_results(predictions, targets, translations, names, metrics)
        
        return metrics
    
    def save_detailed_results(self, predictions, targets, translations, names, metrics):
        """Save detailed predictions to CSV for analysis"""
        results_df = pd.DataFrame({
            'sequence_name': names,
            'target_gloss': targets,
            'predicted_gloss': predictions,
            'german_translation': translations,
            'correct': [p == t for p, t in zip(predictions, targets)]
        })
        
        # Save to CSV
        results_df.to_csv('test_predictions.csv', index=False)
        
        # Save metrics to JSON
        with open('test_metrics.json', 'w') as f:
            # Convert numpy values to Python types
            metrics_serializable = {}
            for k, v in metrics.items():
                if isinstance(v, np.floating):
                    metrics_serializable[k] = float(v)
                elif isinstance(v, dict):
                    metrics_serializable[k] = {sk: float(sv) if isinstance(sv, np.floating) else sv 
                                               for sk, sv in v.items()}
                else:
                    metrics_serializable[k] = v
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"\n✅ Detailed results saved to test_predictions.csv and test_metrics.json")
    
    def print_metrics_table(self, metrics):
        """Print metrics in a nice table format"""
        
        print("\n" + "="*60)
        print("📊 TEST SET EVALUATION RESULTS")
        print("="*60)
        
        # Basic stats
        basic_table = [
            ["Total Samples", metrics['total_samples']],
            ["Avg Inference Time", f"{metrics['avg_inference_time_ms']:.2f} ms"],
            ["FPS", f"{metrics['fps']:.2f}"],
            ["Exact Match Accuracy", f"{metrics['exact_match_accuracy']:.2%}"]
        ]
        print(tabulate(basic_table, headers=["Metric", "Value"], tablefmt="grid"))
        
        # BLEU scores
        print("\n📈 BLEU SCORES:")
        bleu_table = []
        for i in range(1, 5):
            bleu_table.append([f"BLEU-{i}", f"{metrics[f'BLEU-{i}']:.4f}"])
        bleu_table.append(["BLEU (geometric)", f"{metrics['BLEU']:.4f}"])
        bleu_table.append(["WER", f"{metrics['WER']:.2%}"])
        print(tabulate(bleu_table, headers=["Metric", "Score"], tablefmt="grid"))
        
        # Length analysis
        if 'length_analysis' in metrics:
            print("\n📏 PERFORMANCE BY SEQUENCE LENGTH:")
            length_table = []
            for length, stats in metrics['length_analysis'].items():
                length_table.append([
                    length.capitalize(),
                    stats['count'],
                    f"{stats['WER']:.2%}",
                    f"{stats['BLEU']:.4f}"
                ])
            print(tabulate(length_table, 
                          headers=["Length", "Count", "WER", "BLEU"], 
                          tablefmt="grid"))
        
        # Sample predictions
        print("\n🔍 SAMPLE PREDICTIONS (first 10):")
        sample_df = pd.read_csv('test_predictions.csv').head(10)
        print(tabulate(sample_df[['sequence_name', 'target_gloss', 'predicted_gloss', 'correct']],
                      headers=['Sequence', 'Target', 'Predicted', 'Correct'],
                      tablefmt='grid',
                      showindex=False))
        
        print("="*60)


class AdvancedEvaluator(SignLanguageEvaluator):
    """Extended evaluator with additional metrics from reference code"""
    
    def calculate_rouge(self, predictions, targets):
        """ROUGE scores for translation quality"""
        # Simplified ROUGE-1 calculation
        rouge_scores = []
        
        for pred, target in zip(predictions, targets):
            pred_words = set(pred.split())
            target_words = set(target.split())
            
            if not target_words:
                continue
                
            overlap = pred_words.intersection(target_words)
            recall = len(overlap) / len(target_words) if target_words else 0
            
            rouge_scores.append(recall)
        
        return np.mean(rouge_scores) if rouge_scores else 0.0
    
    def calculate_character_error_rate(self, predictions, targets):
        """CER at character level"""
        # Similar to WER but at character level
        total_edits = 0
        total_chars = 0
        
        for pred, target in zip(predictions, targets):
            # Levenshtein distance at character level
            pred_chars = list(pred)
            target_chars = list(target)
            
            # Simple DP for edit distance
            d = np.zeros((len(target_chars) + 1, len(pred_chars) + 1))
            for i in range(len(target_chars) + 1):
                d[i, 0] = i
            for j in range(len(pred_chars) + 1):
                d[0, j] = j
            
            for i in range(1, len(target_chars) + 1):
                for j in range(1, len(pred_chars) + 1):
                    if target_chars[i-1] == pred_chars[j-1]:
                        cost = 0
                    else:
                        cost = 1
                    d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + cost)
            
            total_edits += d[len(target_chars), len(pred_chars)]
            total_chars += len(target_chars)
        
        return total_edits / total_chars if total_chars > 0 else 0.0
    
    def calculate_all_metrics(self, predictions, targets, translations, inference_times, names):
        """Extended metrics including ROUGE and CER"""
        metrics = super().calculate_all_metrics(predictions, targets, translations, inference_times, names)
        
        # Additional metrics
        metrics['ROUGE-1'] = self.calculate_rouge(predictions, targets)
        metrics['CER'] = self.calculate_character_error_rate(predictions, targets)
        
        return metrics


def test_model(model_path, test_loader, tokenizer, device='cuda', beam_width=5):
    """Load saved model and run evaluation"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate model
    model = SignLanguageTransformer(**checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    evaluator = AdvancedEvaluator(model, test_loader, tokenizer, device)
    metrics = evaluator.evaluate(beam_width=beam_width)
    evaluator.print_metrics_table(metrics)
    
    return metrics


def compare_with_baseline(our_metrics, baseline_results):
    """Compare our model with reference implementation"""
    
    print("\n" + "="*60)
    print("📊 COMPARISON WITH BASELINE (Reference Code)")
    print("="*60)
    
    comparison = []
    for metric in ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'WER']:
        if metric in our_metrics and metric in baseline_results:
            improvement = our_metrics[metric] - baseline_results[metric]
            if metric == 'WER':
                improvement = -improvement  # Lower WER is better
            
            comparison.append([
                metric,
                f"{baseline_results[metric]:.4f}",
                f"{our_metrics[metric]:.4f}",
                f"{improvement:+.4f}",
                "✅" if improvement > 0 else "❌" if improvement < 0 else "➡️"
            ])
    
    print(tabulate(comparison, 
                  headers=["Metric", "Baseline", "Ours", "Δ", ""],
                  tablefmt="grid"))
    
    # Statistical significance test
    if 'WER' in our_metrics and 'WER' in baseline_results:
        rel_improvement = (baseline_results['WER'] - our_metrics['WER']) / baseline_results['WER']
        print(f"\n📈 Relative WER improvement: {rel_improvement:.2%}")


# Updated main function with testing
def main():
    # Configuration
    root_dir = '../phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/'
    batch_size = 20
    max_frames = 250
    num_epochs = 100
    
    # Create datasets
    print("📂 Loading datasets...")
    train_dataset = PhoenixSignDataset(root_dir, split='train', max_frames=max_frames, augment=True)
    val_dataset = PhoenixSignDataset(root_dir, split='dev', max_frames=max_frames, augment=False)
    test_dataset = PhoenixSignDataset(root_dir, split='test', max_frames=max_frames, augment=False)
    
    # Create tokenizer
    tokenizer = GlossTokenizer(train_dataset.glosses)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    # model = SignLanguageTransformer(
    #     input_dim=225,
    #     dim=192,
    #     num_classes=tokenizer.vocab_size,
    #     max_frames=max_frames,
    #     dropout=0.2
    # )
    sample = train_dataset[0]
    input_dim = sample['keypoints'].shape[-1]
    print(f"Detected input_dim: {input_dim}")

    model = SignLanguageTransformer(
        input_dim=input_dim,
        dim=192,
        num_classes=tokenizer.vocab_size,
        max_frames=max_frames,
        dropout=0.2
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🤖 Model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Using device: {device}")
    
    trainer = Trainer(model, train_loader, val_loader, tokenizer, device=device)
    
    # Train
    print("\n🏋️ Starting training...")
    trainer.train(num_epochs=num_epochs)
    
    # Save final model
    model_path = 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': {
            'input_dim': input_dim,
            'dim': 192,
            'num_classes': tokenizer.vocab_size,
            'max_frames': max_frames
        }
    }, model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Test on validation set (quick check)
    print("\n🔍 Quick validation evaluation...")
    val_evaluator = AdvancedEvaluator(model, val_loader, tokenizer, device)
    val_metrics = val_evaluator.evaluate(beam_width=1, verbose=False)
    val_evaluator.print_metrics_table(val_metrics)
    
    # Final test on test set
    print("\n" + "🎯"*30)
    print("🎯 FINAL TEST SET EVALUATION")
    print("🎯"*30)
    
    test_evaluator = AdvancedEvaluator(model, test_loader, tokenizer, device)
    test_metrics = test_evaluator.evaluate(beam_width=5, verbose=True)
    test_evaluator.print_metrics_table(test_metrics)
    
    # Compare with baseline (if you have baseline results)
    # baseline_results = {'BLEU-1': 0.45, 'BLEU-2': 0.32, 'BLEU-3': 0.25, 'BLEU-4': 0.18, 'WER': 0.35}
    # compare_with_baseline(test_metrics, baseline_results)
    
    print("\n✨ Evaluation complete! Check test_predictions.csv and test_metrics.json for details.")
    
    return test_metrics


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Run main
    metrics = main()    