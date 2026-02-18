import json
import math
from typing import List, Dict, Any, Iterable
from pathlib import Path
import os
from collections import Counter
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CTCLoss(nn.Module):
    """CTC Loss wrapper for sequence prediction"""
    
    def __init__(self, blank=0, reduction='mean'):
        super().__init__()
        self.blank = blank
        self.reduction = reduction
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
        
    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        Args:
            logits: (T, B, C) - raw logits (time-major)
            targets: (B, S) - target sequences
            input_lengths: (B,) - lengths of input sequences
            target_lengths: (B,) - lengths of target sequences
        """
        log_probs = F.log_softmax(logits, dim=-1)
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    keypoints = [item['keypoints'] for item in batch]
    glosses = [item['gloss'] for item in batch]
    num_frames = [item['num_frames'] for item in batch]
    
    # Pad keypoints
    keypoints_padded = nn.utils.rnn.pad_sequence(keypoints, batch_first=True)
    
    # Create mask from actual frame counts (NOT from zero-checking, which
    # breaks after chicken-neck normalization centers data around origin)
    B = len(batch)
    T = keypoints_padded.shape[1]
    mask = torch.zeros(B, T, dtype=torch.float32)
    for i, nf in enumerate(num_frames):
        mask[i, :nf] = 1.0
    
    # Pad gloss token sequences
    glosses_padded = nn.utils.rnn.pad_sequence(glosses, batch_first=True, padding_value=0)
    
    return {
        'keypoints': keypoints_padded,
        'mask': mask,
        'gloss': glosses_padded,
        'gloss_text': [item['gloss_text'] for item in batch],
        'translation': [item['translation'] for item in batch],
        'name': [item['name'] for item in batch]
    }


class Trainer:
    """Training loop with learning rate scheduling and evaluation"""
    
    def __init__(self, model, train_loader, val_loader, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        
        self.criterion = CTCLoss(blank=0)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Learning rate scheduler with warmup
        self.total_steps = len(train_loader) * 100  # 100 epochs
        self.warmup_steps = len(train_loader) * 5   # 5 epochs warmup
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['gloss'].to(self.device)
            
            # Update learning rate
            step = epoch * len(self.train_loader) + batch_idx
            self._adjust_learning_rate(step)
            
            # Forward pass
            logits, mask_out = self.model(keypoints, mask)
            
            # Prepare for CTC loss: (B, T, C) -> (T, B, C)
            logits_ctc = logits.permute(1, 0, 2)
            
            # Input lengths from mask (number of valid frames)
            input_lengths = mask_out.sum(dim=1).long().cpu()
            
            # Target lengths (number of non-zero gloss tokens)
            target_lengths = (targets != 0).sum(dim=1).long().cpu()
            
            # Skip batch if any target is empty or input shorter than target
            if (target_lengths == 0).any():
                continue
            if (input_lengths < target_lengths).any():
                continue
            
            # Compute loss
            loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f'  Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, '
                      f'Loss: {loss.item():.4f}, LR: {lr:.2e}')
                
        return total_loss / max(1, num_batches)
    
    def _adjust_learning_rate(self, step):
        """Adjust learning rate with warmup and cosine decay"""
        if step < self.warmup_steps:
            # Exponential warmup (from reference code — better than linear)
            lr = 1e-3 * 2 ** -(self.warmup_steps - step)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        for batch in self.val_loader:
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['gloss'].to(self.device)
            
            logits, mask_out = self.model(keypoints, mask)
            logits_ctc = logits.permute(1, 0, 2)
            
            input_lengths = mask_out.sum(dim=1).long().cpu()
            target_lengths = (targets != 0).sum(dim=1).long().cpu()
            
            if (target_lengths > 0).all() and (input_lengths >= target_lengths).all():
                loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1
            
            # Decode predictions
            predictions = self._decode_predictions(logits, input_lengths)
            all_predictions.extend(predictions)
            all_targets.extend(batch['gloss_text'])
            
        avg_loss = total_loss / max(1, num_batches)
        return avg_loss, all_predictions, all_targets
    
    def _decode_predictions(self, logits, input_lengths):
        """Greedy CTC decoding"""
        # logits: (B, T, C)
        predictions = []
        
        for i in range(logits.shape[0]):
            valid_len = min(input_lengths[i].item(), logits.shape[1])
            pred = logits[i, :valid_len].argmax(dim=-1).cpu().numpy()
            
            # CTC decode: remove consecutive duplicates, then remove blanks (0)
            unique_pred = []
            prev = -1
            for p in pred:
                if p != prev:
                    if p != 0:  # 0 is blank
                        unique_pred.append(int(p))
                    prev = p
                    
            predictions.append(unique_pred)
            
        return predictions
    
    def train(self, num_epochs=100):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss, predictions, targets = self.validate(epoch)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Print some examples
            if epoch % 5 == 0:
                for i in range(min(5, len(predictions))):
                    pred_text = self.tokenizer.decode(predictions[i])
                    print(f'  Target: {targets[i]}')
                    print(f'  Pred  : {pred_text}')
                    print('  ---')
                    
            # Save best checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_model.pt')
                print(f'  ✅ New best model saved (val_loss={val_loss:.4f})')
                    
            # Save periodic checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'checkpoint_epoch_{epoch}.pt')
                

class GlossTokenizer:
    """
    WORD-LEVEL tokenizer for gloss sequences.
    
    Splits gloss strings like "ICH OSTERN WETTER ZUFRIEDEN" into individual
    word tokens: [ICH, OSTERN, WETTER, ZUFRIEDEN].
    
    Token 0 = <blank> (CTC blank)
    Token 1 = <unk> (unknown words)
    Token 2+ = gloss words
    """
    
    def __init__(self, gloss_sentences=None, min_freq=1):
        self.gloss_to_idx = {'<blank>': 0, '<unk>': 1}
        self.idx_to_gloss = {0: '<blank>', 1: '<unk>'}
        
        if gloss_sentences is not None:
            self._build_vocab(gloss_sentences, min_freq)
    
    def _build_vocab(self, gloss_sentences, min_freq=1):
        """Build word-level vocabulary from gloss sentences."""
        word_counts = Counter()
        for sentence in gloss_sentences:
            if isinstance(sentence, str):
                words = sentence.strip().upper().split()
                word_counts.update(words)
        
        for word, count in word_counts.most_common():
            if count >= min_freq and word not in self.gloss_to_idx:
                idx = len(self.gloss_to_idx)
                self.gloss_to_idx[word] = idx
                self.idx_to_gloss[idx] = word
        
        print(f"GlossTokenizer: {len(self.gloss_to_idx)} word tokens "
              f"(from {len(gloss_sentences)} sentences, {len(word_counts)} unique words)")
                
    def encode(self, gloss_text):
        """
        Encode a gloss sentence string to a sequence of word token indices.
        
        "ICH OSTERN WETTER" → [42, 156, 203]
        """
        if isinstance(gloss_text, str):
            words = gloss_text.strip().upper().split()
            return torch.tensor(
                [self.gloss_to_idx.get(w, 1) for w in words],
                dtype=torch.long
            )
        else:
            # Already a list of words
            return torch.tensor(
                [self.gloss_to_idx.get(w, 1) for w in gloss_text],
                dtype=torch.long
            )
    
    def decode(self, indices):
        """Decode token indices back to a gloss string."""
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
            
        if isinstance(indices, (int, np.integer)):
            return self.idx_to_gloss.get(int(indices), '<unk>')
        else:
            words = []
            for i in indices:
                i = int(i)
                if i > 1:  # Skip <blank>=0 and <unk>=1
                    words.append(self.idx_to_gloss.get(i, '<unk>'))
            return ' '.join(words)
    
    @property
    def vocab_size(self):
        return len(self.gloss_to_idx)