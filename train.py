"""
Train Sign Language Transformer with configurable modes.

Example comparison runs:
  # Run 1: CTC-only, greedy, 100 epochs (baseline)
  python train.py --epochs 100 --decode greedy

  # Run 2: CTC-only, beam search, 150 epochs
  python train.py --epochs 150 --decode beam --beam_width 10

  # Run 3: CTC + BART translation, 150 epochs
  python train.py --epochs 150 --decode beam --beam_width 10 --use_bart --bart_model facebook/bart-base

  # Run 4: CTC + BART, more aggressive joint training
  python train.py --epochs 150 --use_bart --ctc_weight 0.5 --freeze_bart_epochs 10
"""

import time
import argparse
import pandas as pd
from collections import Counter
import math
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import random

from dataset import PhoenixSignDataset
from models import SignLanguageTransformer
from utils import GlossTokenizer, Trainer, collate_fn, ctc_greedy_decode, ctc_beam_decode


class SignLanguageEvaluator:
    def __init__(self, model, test_loader, tokenizer, device='cuda',
                 bart_tokenizer=None, decode_mode='greedy', beam_width=5):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.device = device
        self.decode_mode = decode_mode
        self.beam_width = beam_width
        self.wer_costs = {'del': 3, 'ins': 3, 'sub': 4}

    @torch.no_grad()
    def evaluate(self, verbose=True):
        self.model.eval()
        all_preds, all_targets, all_trans_preds, all_trans_targets = [], [], [], []
        all_names = []
        inference_times = []

        for batch_idx, batch in enumerate(self.test_loader):
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)

            start = time.time()
            
            if self.model.use_bart:
                output = self.model(keypoints, mask)
                logits = output['logits']
                mask_out = output['mask']
            else:
                logits, mask_out = self.model(keypoints, mask)

            inference_times.append(time.time() - start)

            logits_cpu = logits.cpu()
            input_lengths = mask_out.sum(dim=1).long().cpu()

            # CTC decode glosses
            if self.decode_mode == 'beam':
                preds = ctc_beam_decode(logits_cpu, input_lengths, self.beam_width)
            else:
                preds = ctc_greedy_decode(logits_cpu, input_lengths)

            for p in preds:
                all_preds.append(self.tokenizer.decode(p))
            all_targets.extend(batch['gloss_text'])
            all_names.extend(batch['name'])
            
            # BART translation
            if self.model.use_bart and self.bart_tokenizer is not None:
                try:
                    token_ids = self.model.translate(
                        keypoints, mask, beam_width=self.beam_width)
                    for i in range(token_ids.shape[0]):
                        text = self.bart_tokenizer.decode(
                            token_ids[i], skip_special_tokens=True)
                        all_trans_preds.append(text)
                    all_trans_targets.extend(batch['translation'])
                except Exception as e:
                    if verbose:
                        print(f"  Translation failed: {e}")

            if verbose and batch_idx % 10 == 0:
                print(f"  Eval batch {batch_idx}/{len(self.test_loader)}")

        metrics = self._compute_metrics(all_preds, all_targets, inference_times)
        
        # Translation metrics
        if all_trans_preds:
            trans_bleu = self._compute_bleu(all_trans_preds, all_trans_targets)
            metrics['trans_BLEU-1'] = trans_bleu.get('BLEU-1', 0)
            metrics['trans_BLEU-4'] = trans_bleu.get('BLEU-4', 0)

        # Save results
        self._save_results(all_preds, all_targets, all_names,
                           all_trans_preds, all_trans_targets, metrics)
        return metrics

    def _compute_metrics(self, preds, targets, times):
        metrics = {
            'total_samples': len(preds),
            'avg_ms': np.mean(times) * 1000,
            'WER': self._compute_wer(preds, targets),
        }
        bleu = self._compute_bleu(preds, targets)
        metrics.update(bleu)
        exact = sum(1 for p, t in zip(preds, targets) if p.strip() == t.strip())
        metrics['exact_match'] = exact / len(preds) if preds else 0
        return metrics

    def _compute_wer(self, preds, targets):
        def wer_single(p, t):
            pw, tw = p.split(), t.split()
            if not tw: return 0.0
            d = np.zeros((len(tw)+1, len(pw)+1))
            for i in range(len(tw)+1): d[i,0] = i * 3
            for j in range(len(pw)+1): d[0,j] = j * 3
            for i in range(1, len(tw)+1):
                for j in range(1, len(pw)+1):
                    cost = 0 if tw[i-1] == pw[j-1] else 4
                    d[i,j] = min(d[i-1,j]+3, d[i,j-1]+3, d[i-1,j-1]+cost)
            mc = 4 * max(len(tw), len(pw))
            return d[len(tw), len(pw)] / mc if mc > 0 else 0.0
        return np.mean([wer_single(p, t) for p, t in zip(preds, targets)])

    def _compute_bleu(self, preds, targets, max_n=4):
        def ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        scores = {}
        for n in range(1, max_n+1):
            prec_sum, count = 0, 0
            for p, t in zip(preds, targets):
                pt, tt = p.split(), t.split()
                if len(pt) < n or len(tt) < n: continue
                pn, tn = Counter(ngrams(pt, n)), Counter(ngrams(tt, n))
                matches = sum((pn & tn).values())
                total = sum(pn.values())
                if total > 0:
                    prec_sum += matches / total
                    count += 1
            scores[f'BLEU-{n}'] = prec_sum / count if count > 0 else 0.0
        if all(v > 0 for v in scores.values()):
            scores['BLEU'] = math.exp(sum(math.log(v) for v in scores.values()) / len(scores))
        else:
            scores['BLEU'] = 0.0
        return scores

    def _save_results(self, preds, targets, names, trans_preds, trans_targets, metrics):
        df = pd.DataFrame({
            'name': names, 'target': targets, 'predicted': preds,
            'correct': [p.strip() == t.strip() for p, t in zip(preds, targets)]
        })
        if trans_preds:
            # Align lengths
            while len(trans_preds) < len(names):
                trans_preds.append('')
                trans_targets.append('')
            df['trans_target'] = trans_targets[:len(names)]
            df['trans_pred'] = trans_preds[:len(names)]
        df.to_csv('test_predictions.csv', index=False)
        
        ser = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
               for k, v in metrics.items()}
        with open('test_metrics.json', 'w') as f:
            json.dump(ser, f, indent=2)
        print(f"  ✅ Saved test_predictions.csv and test_metrics.json")

    def print_metrics(self, metrics):
        print("\n" + "="*60)
        print("📊 EVALUATION RESULTS")
        print("="*60)
        print(f"  Decode:  {self.decode_mode} (beam={self.beam_width})")
        print(f"  Samples: {metrics['total_samples']}")
        print(f"  WER:     {metrics['WER']:.2%}")
        print(f"  BLEU-1:  {metrics.get('BLEU-1', 0):.4f}")
        print(f"  BLEU-4:  {metrics.get('BLEU-4', 0):.4f}")
        print(f"  Exact:   {metrics.get('exact_match', 0):.2%}")
        if 'trans_BLEU-1' in metrics:
            print(f"  --- Translation ---")
            print(f"  Trans BLEU-1: {metrics['trans_BLEU-1']:.4f}")
            print(f"  Trans BLEU-4: {metrics['trans_BLEU-4']:.4f}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train Sign Language Transformer")
    
    # Data
    parser.add_argument('--root_dir', type=str, 
                        default='../phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--max_frames', type=int, default=250)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    
    # Decoding
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_width', type=int, default=10)
    
    # BART translation
    parser.add_argument('--use_bart', action='store_true', help='Enable BART Gloss→Text')
    parser.add_argument('--bart_model', type=str, default='facebook/bart-base')
    parser.add_argument('--ctc_weight', type=float, default=0.3,
                        help='CTC loss weight in joint training (1-ctc_weight for BART)')
    parser.add_argument('--freeze_bart_epochs', type=int, default=5,
                        help='Epochs to freeze BART before joint training')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval_only', action='store_true')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    print(f"{'='*60}")
    print(f"Config: epochs={args.epochs}, decode={args.decode}, "
          f"beam={args.beam_width}, bart={args.use_bart}")
    print(f"{'='*60}")
    
    # ─── Tokenizers ───
    print("Building tokenizers...")
    all_gloss = []
    for split in ['train', 'dev', 'test']:
        csv = Path(args.root_dir) / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
        df = pd.read_csv(csv, sep='|')
        all_gloss.extend(df['orth'].dropna().tolist())
    tokenizer = GlossTokenizer(all_gloss, min_freq=1)
    
    bart_tokenizer = None
    if args.use_bart:
        from transformers import BartTokenizer
        bart_tokenizer = BartTokenizer.from_pretrained(args.bart_model)
        print(f"  BART tokenizer loaded: {args.bart_model}")
    
    # ─── Datasets ───
    print("Loading datasets...")
    train_ds = PhoenixSignDataset(args.root_dir, 'train', args.max_frames, True,
                                   tokenizer, bart_tokenizer)
    val_ds = PhoenixSignDataset(args.root_dir, 'dev', args.max_frames, False,
                                 tokenizer, bart_tokenizer)
    test_ds = PhoenixSignDataset(args.root_dir, 'test', args.max_frames, False,
                                  tokenizer, bart_tokenizer)
    
    sample = train_ds[0]
    input_dim = sample['keypoints'].shape[-1]
    print(f"  input_dim={input_dim}, vocab={tokenizer.vocab_size}, "
          f"train={len(train_ds)}, dev={len(val_ds)}, test={len(test_ds)}")
    
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=args.num_workers,
                               pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=args.num_workers,
                             pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.num_workers,
                              pin_memory=True)
    
    # ─── Model ───
    model = SignLanguageTransformer(
        input_dim=input_dim, dim=args.dim,
        num_classes=tokenizer.vocab_size,
        max_frames=args.max_frames * 2,
        dropout=args.dropout,
        use_bart=args.use_bart,
        bart_model=args.bart_model,
        ctc_weight=args.ctc_weight,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🤖 Model: {total_params:,} total, {trainable:,} trainable")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"💻 Device: {device}")
    
    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Resumed from {args.resume}")
    
    # ─── Train ───
    if not args.eval_only:
        trainer = Trainer(
            model, train_loader, val_loader, tokenizer, device,
            bart_tokenizer=bart_tokenizer,
            use_bart=args.use_bart,
            ctc_weight=args.ctc_weight,
            freeze_bart_epochs=args.freeze_bart_epochs,
        )
        print(f"\n🏋️ Training for {args.epochs} epochs...")
        trainer.train(num_epochs=args.epochs, decode_mode=args.decode,
                      beam_width=args.beam_width)
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'tokenizer_vocab': tokenizer.gloss_to_idx,
            'config': {
                'input_dim': input_dim, 'dim': args.dim,
                'num_classes': tokenizer.vocab_size,
                'max_frames': args.max_frames,
                'use_bart': args.use_bart,
                'bart_model': args.bart_model,
            },
            'args': vars(args),
        }, 'final_model.pt')
        print("💾 Saved final_model.pt")
    
    # ─── Evaluate ───
    print(f"\n🎯 FINAL EVALUATION (decode={args.decode}, beam={args.beam_width})")
    evaluator = SignLanguageEvaluator(
        model, test_loader, tokenizer, device,
        bart_tokenizer=bart_tokenizer,
        decode_mode=args.decode, beam_width=args.beam_width,
    )
    metrics = evaluator.evaluate(verbose=True)
    evaluator.print_metrics(metrics)
    
    # Show samples
    print("\n🔍 Samples:")
    df = pd.read_csv('test_predictions.csv').head(10)
    for _, row in df.iterrows():
        m = "✅" if row['correct'] else "❌"
        print(f"  {m} Target: {row['target']}")
        print(f"     Pred:   {row['predicted']}")
        if 'trans_pred' in row and pd.notna(row.get('trans_pred', None)):
            print(f"     Trans:  {row['trans_pred']}")
        print()
    
    return metrics


if __name__ == '__main__':
    main()