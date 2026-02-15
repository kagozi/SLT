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
import math
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
            logits: (T, B, C) - log probabilities (time-major)
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
    
    # Pad keypoints
    keypoints_padded = nn.utils.rnn.pad_sequence(keypoints, batch_first=True)
    
    # Create mask
    mask = (keypoints_padded.abs().sum(dim=-1) != 0).float()
    
    # Stack glosses (will need to handle variable-length gloss sequences)
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
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
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
            
            # Prepare for CTC loss
            logits = logits.transpose(0, 1)  # (T, B, C)
            
            # Calculate input lengths (based on mask)
            input_lengths = mask_out.sum(dim=1).long().cpu()
            
            # Calculate target lengths
            target_lengths = (targets != 0).sum(dim=1).long().cpu()
            
            # Compute loss
            loss = self.criterion(logits, targets, input_lengths, target_lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        return total_loss / len(self.train_loader)
    
    def _adjust_learning_rate(self, step):
        """Adjust learning rate with warmup and cosine decay"""
        if step < self.warmup_steps:
            # Linear warmup
            lr = 1e-3 * (step + 1) / self.warmup_steps
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        for batch in self.val_loader:
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['gloss'].to(self.device)
            
            logits, mask_out = self.model(keypoints, mask)
            logits = logits.transpose(0, 1)
            
            input_lengths = mask_out.sum(dim=1).long().cpu()
            target_lengths = (targets != 0).sum(dim=1).long().cpu()
            
            loss = self.criterion(logits, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Decode predictions
            predictions = self._decode_predictions(logits, input_lengths)
            all_predictions.extend(predictions)
            all_targets.extend([self.tokenizer.decode(t) for t in targets])
            
        return total_loss / len(self.val_loader), all_predictions, all_targets
    
    def _decode_predictions(self, logits, input_lengths, beam_width=5):
        """Greedy decoding (can be extended to beam search)"""
        logits = logits.transpose(0, 1)  # (B, T, C)
        predictions = []
        
        for i in range(logits.shape[0]):
            pred = logits[i, :input_lengths[i]].argmax(dim=-1).cpu().numpy()
            
            # Remove consecutive duplicates and blanks (0)
            unique_pred = []
            prev = -1
            for p in pred:
                if p != prev and p != 0:
                    unique_pred.append(p)
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
                for i in range(min(3, len(predictions))):
                    print(f'Target: {targets[i]}')
                    print(f'Pred  : {self.tokenizer.decode(predictions[i])}')
                    print('---')
                    
            # Save checkpoint
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, f'checkpoint_epoch_{epoch}.pt')
                
class GlossTokenizer:
    """Tokenizer for gloss sequences"""
    
    def __init__(self, glosses):
        self.gloss_to_idx = {'<pad>': 0, '<blank>': 1}
        self.idx_to_gloss = {0: '<pad>', 1: '<blank>'}
        
        for gloss in glosses:
            if gloss not in self.gloss_to_idx:
                idx = len(self.gloss_to_idx)
                self.gloss_to_idx[gloss] = idx
                self.idx_to_gloss[idx] = gloss
                
    def encode(self, gloss):
        """Encode a gloss string to indices"""
        if isinstance(gloss, str):
            return torch.tensor([self.gloss_to_idx.get(gloss, 1)])  # 1 is blank
        else:
            return torch.tensor([self.gloss_to_idx.get(g, 1) for g in gloss])
    
    def decode(self, indices):
        """Decode indices to gloss string"""
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
            
        if isinstance(indices, (int, np.integer)):
            return self.idx_to_gloss.get(indices, '<unk>')
        else:
            return ' '.join([self.idx_to_gloss.get(i, '<unk>') for i in indices if i > 1])  # Skip pad/blank
    
    @property
    def vocab_size(self):
        return len(self.gloss_to_idx)