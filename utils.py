# import json
# import math
# from typing import List, Dict
# from collections import Counter
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class CTCLoss(nn.Module):
#     def __init__(self, blank=0, reduction='mean'):
#         super().__init__()
#         self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
#     def forward(self, logits, targets, input_lengths, target_lengths):
#         log_probs = F.log_softmax(logits, dim=-1)
#         return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


# def collate_fn(batch):
#     """Custom collate for variable-length sequences."""
#     keypoints = [item['keypoints'] for item in batch]
#     glosses = [item['gloss'] for item in batch]
#     num_frames = [item['num_frames'] for item in batch]

#     keypoints_padded = nn.utils.rnn.pad_sequence(keypoints, batch_first=True)

#     B, T = len(batch), keypoints_padded.shape[1]
#     mask = torch.zeros(B, T, dtype=torch.float32)
#     for i, nf in enumerate(num_frames):
#         mask[i, :nf] = 1.0

#     glosses_padded = nn.utils.rnn.pad_sequence(glosses, batch_first=True, padding_value=0)

#     result = {
#         'keypoints': keypoints_padded,
#         'mask': mask,
#         'gloss': glosses_padded,
#         'gloss_text': [item['gloss_text'] for item in batch],
#         'translation': [item['translation'] for item in batch],
#         'name': [item['name'] for item in batch],
#     }
    
#     # Include translation token ids if present
#     if 'translation_ids' in batch[0] and batch[0]['translation_ids'] is not None:
#         trans_ids = [item['translation_ids'] for item in batch]
#         result['translation_ids'] = nn.utils.rnn.pad_sequence(
#             trans_ids, batch_first=True, padding_value=1  # pad_token_id=1 for BART
#         )
    
#     return result


# # ─── CTC Decoding ────────────────────────────────────────────────────

# def ctc_greedy_decode(logits, input_lengths=None):
#     """
#     Greedy CTC decoding.
#     logits: (B, T, V)
#     Returns: list of decoded token id lists
#     """
#     B, T, V = logits.shape
#     predictions = []
#     for i in range(B):
#         valid_len = input_lengths[i].item() if input_lengths is not None else T
#         pred = logits[i, :valid_len].argmax(dim=-1).cpu().numpy()
#         unique_pred = []
#         prev = -1
#         for p in pred:
#             if p != prev:
#                 if p != 0:
#                     unique_pred.append(int(p))
#                 prev = p
#         predictions.append(unique_pred)
#     return predictions


# def ctc_beam_decode(logits, input_lengths=None, beam_width=10):
#     """
#     Prefix beam search CTC decoding with length normalization.
#     logits: (B, T, V) — raw logits (will be log_softmax'd)
#     Returns: list of decoded token id lists
#     """
#     if beam_width <= 1:
#         return ctc_greedy_decode(logits, input_lengths)

#     B, T, V = logits.shape
#     log_probs = F.log_softmax(logits, dim=-1)
#     decoded = []

#     for b in range(B):
#         valid_len = input_lengths[b].item() if input_lengths is not None else T
        
#         # beam: dict of {prefix_tuple: log_prob}
#         beams = {(): 0.0}

#         for t in range(valid_len):
#             lp = log_probs[b, t]  # (V,)
#             # Only consider top-k tokens per timestep for efficiency
#             topk_vals, topk_idx = lp.topk(min(beam_width * 2, V))
            
#             new_beams = {}
#             for prefix, score in beams.items():
#                 for k in range(topk_vals.shape[0]):
#                     c = topk_idx[k].item()
#                     new_score = score + topk_vals[k].item()

#                     if c == 0:  # blank
#                         key = prefix
#                     elif len(prefix) > 0 and prefix[-1] == c:
#                         key = prefix  # collapse duplicate
#                     else:
#                         key = prefix + (c,)

#                     if key not in new_beams or new_beams[key] < new_score:
#                         new_beams[key] = new_score

#             # Prune: keep top beam_width, normalized by length
#             scored = [(k, v / max(1, len(k))) for k, v in new_beams.items()]
#             scored.sort(key=lambda x: -x[1])
#             # Store un-normalized scores for next iteration
#             beams = {}
#             for k, _ in scored[:beam_width]:
#                 beams[k] = new_beams[k]

#         # Pick best beam (length-normalized)
#         if beams:
#             best = max(beams.items(), key=lambda x: x[1] / max(1, len(x[0])))
#             decoded.append(list(best[0]))
#         else:
#             decoded.append([])

#     return decoded


# # ─── Trainer ─────────────────────────────────────────────────────────

# class Trainer:
#     """Training loop supporting CTC-only and joint CTC+BART modes."""
    
#     def __init__(self, model, train_loader, val_loader, tokenizer, device='cuda',
#                  bart_tokenizer=None, use_bart=False, ctc_weight=0.3,
#                  freeze_bart_epochs=5):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.tokenizer = tokenizer
#         self.bart_tokenizer = bart_tokenizer
#         self.device = device
#         self.use_bart = use_bart
#         self.ctc_weight = ctc_weight
#         self.freeze_bart_epochs = freeze_bart_epochs

#         self.criterion = CTCLoss(blank=0)
#         self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

#         self.total_steps = len(train_loader) * 150
#         self.warmup_steps = len(train_loader) * 5
#         self.best_val_loss = float('inf')

#     def train_epoch(self, epoch):
#         self.model.train()
#         total_loss = 0
#         total_ctc = 0
#         total_trans = 0
#         num_batches = 0

#         for batch_idx, batch in enumerate(self.train_loader):
#             keypoints = batch['keypoints'].to(self.device)
#             mask = batch['mask'].to(self.device)
#             targets = batch['gloss'].to(self.device)

#             step = epoch * len(self.train_loader) + batch_idx
#             self._adjust_learning_rate(step)

#             self.optimizer.zero_grad()

#             if self.use_bart:
#                 trans_targets = batch.get('translation_ids')
#                 if trans_targets is not None:
#                     trans_targets = trans_targets.to(self.device)
                
#                 output = self.model(keypoints, mask, translation_targets=trans_targets)
#                 logits = output['logits']
#                 mask_out = output['mask']
#             else:
#                 logits, mask_out = self.model(keypoints, mask)

#             # CTC loss
#             logits_ctc = logits.permute(1, 0, 2)
#             input_lengths = mask_out.sum(dim=1).long().cpu()
#             target_lengths = (targets != 0).sum(dim=1).long().cpu()

#             if (target_lengths == 0).any() or (input_lengths < target_lengths).any():
#                 continue

#             ctc_loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)
            
#             if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
#                 continue

#             # Total loss
#             if self.use_bart and 'translation_loss' in output:
#                 trans_loss = output['translation_loss']
#                 loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * trans_loss
#                 total_trans += trans_loss.item()
#             else:
#                 loss = ctc_loss

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
#             self.optimizer.step()

#             total_loss += loss.item()
#             total_ctc += ctc_loss.item()
#             num_batches += 1

#             if batch_idx % 50 == 0:
#                 lr = self.optimizer.param_groups[0]['lr']
#                 msg = f'  Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}'
#                 if self.use_bart and 'translation_loss' in output:
#                     msg += f', CTC: {ctc_loss.item():.4f}, Trans: {trans_loss.item():.4f}'
#                 msg += f', LR: {lr:.2e}'
#                 print(msg)

#         n = max(1, num_batches)
#         return {'loss': total_loss/n, 'ctc': total_ctc/n, 'trans': total_trans/n}

#     def _adjust_learning_rate(self, step):
#         if step < self.warmup_steps:
#             lr = 1e-3 * 2 ** -(self.warmup_steps - step)
#         else:
#             progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
#             lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * progress))
#         for pg in self.optimizer.param_groups:
#             pg['lr'] = lr

#     @torch.no_grad()
#     def validate(self, epoch, decode_mode='greedy', beam_width=5):
#         self.model.eval()
#         total_loss = 0
#         num_batches = 0
#         all_predictions = []
#         all_targets = []

#         for batch in self.val_loader:
#             keypoints = batch['keypoints'].to(self.device)
#             mask = batch['mask'].to(self.device)
#             targets = batch['gloss'].to(self.device)

#             if self.use_bart:
#                 output = self.model(keypoints, mask)
#                 logits = output['logits']
#                 mask_out = output['mask']
#             else:
#                 logits, mask_out = self.model(keypoints, mask)

#             logits_ctc = logits.permute(1, 0, 2)
#             input_lengths = mask_out.sum(dim=1).long().cpu()
#             target_lengths = (targets != 0).sum(dim=1).long().cpu()

#             if (target_lengths > 0).all() and (input_lengths >= target_lengths).all():
#                 loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)
#                 if not (torch.isnan(loss) or torch.isinf(loss)):
#                     total_loss += loss.item()
#                     num_batches += 1

#             # Decode
#             if decode_mode == 'beam':
#                 preds = ctc_beam_decode(logits.cpu(), input_lengths, beam_width)
#             else:
#                 preds = ctc_greedy_decode(logits.cpu(), input_lengths)
            
#             all_predictions.extend(preds)
#             all_targets.extend(batch['gloss_text'])

#         return total_loss / max(1, num_batches), all_predictions, all_targets

#     def train(self, num_epochs=100, decode_mode='greedy', beam_width=5):
#         # Freeze BART initially if using it
#         if self.use_bart:
#             print(f"  Stage 1: Freezing BART for first {self.freeze_bart_epochs} epochs")
#             self.model.freeze_translation()

#         for epoch in range(num_epochs):
#             # Unfreeze BART at transition
#             if self.use_bart and epoch == self.freeze_bart_epochs:
#                 print(f"\n  >>> Stage 2: Unfreezing BART <<<\n")
#                 self.model.unfreeze_translation()

#             metrics = self.train_epoch(epoch)
#             val_loss, predictions, targets = self.validate(
#                 epoch, decode_mode=decode_mode, beam_width=beam_width)

#             print(f'Epoch {epoch}: Train={metrics["loss"]:.4f} '
#                   f'(CTC={metrics["ctc"]:.4f} Trans={metrics["trans"]:.4f}) '
#                   f'Val={val_loss:.4f}')

#             if epoch % 5 == 0:
#                 for i in range(min(3, len(predictions))):
#                     pred_text = self.tokenizer.decode(predictions[i])
#                     print(f'  Target: {targets[i]}')
#                     print(f'  Pred  : {pred_text}')
#                     print('  ---')

#             if val_loss < self.best_val_loss:
#                 self.best_val_loss = val_loss
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': self.optimizer.state_dict(),
#                     'val_loss': val_loss,
#                 }, 'best_model.pt')
#                 print(f'  ✅ Best model saved (val_loss={val_loss:.4f})')

#             if epoch % 10 == 0:
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': self.model.state_dict(),
#                     'optimizer_state_dict': self.optimizer.state_dict(),
#                 }, f'checkpoint_epoch_{epoch}.pt')


# # ─── Tokenizer ───────────────────────────────────────────────────────

# class GlossTokenizer:
#     """Word-level tokenizer for gloss sequences. Token 0=<blank>, 1=<unk>."""
#     def __init__(self, gloss_sentences=None, min_freq=1):
#         self.gloss_to_idx = {'<blank>': 0, '<unk>': 1}
#         self.idx_to_gloss = {0: '<blank>', 1: '<unk>'}
#         if gloss_sentences is not None:
#             self._build_vocab(gloss_sentences, min_freq)

#     def _build_vocab(self, gloss_sentences, min_freq=1):
#         word_counts = Counter()
#         for s in gloss_sentences:
#             if isinstance(s, str):
#                 word_counts.update(s.strip().upper().split())
#         for word, count in word_counts.most_common():
#             if count >= min_freq and word not in self.gloss_to_idx:
#                 idx = len(self.gloss_to_idx)
#                 self.gloss_to_idx[word] = idx
#                 self.idx_to_gloss[idx] = word
#         print(f"GlossTokenizer: {len(self.gloss_to_idx)} word tokens")

#     def encode(self, gloss_text):
#         if isinstance(gloss_text, str):
#             words = gloss_text.strip().upper().split()
#             return torch.tensor([self.gloss_to_idx.get(w, 1) for w in words], dtype=torch.long)
#         return torch.tensor([self.gloss_to_idx.get(w, 1) for w in gloss_text], dtype=torch.long)

#     def decode(self, indices):
#         if torch.is_tensor(indices):
#             indices = indices.cpu().numpy()
#         if isinstance(indices, (int, np.integer)):
#             return self.idx_to_gloss.get(int(indices), '<unk>')
#         return ' '.join(self.idx_to_gloss.get(int(i), '<unk>') for i in indices if int(i) > 1)

#     @property
#     def vocab_size(self):
#         return len(self.gloss_to_idx)

import json
import math
from typing import List, Dict
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CTCLoss(nn.Module):
    def __init__(self, blank=0, reduction='mean'):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)


def collate_fn(batch):
    """Custom collate for variable-length sequences."""
    keypoints = [item['keypoints'] for item in batch]
    glosses = [item['gloss'] for item in batch]
    num_frames = [item['num_frames'] for item in batch]

    keypoints_padded = nn.utils.rnn.pad_sequence(keypoints, batch_first=True)

    B, T = len(batch), keypoints_padded.shape[1]
    mask = torch.zeros(B, T, dtype=torch.float32)
    for i, nf in enumerate(num_frames):
        mask[i, :nf] = 1.0

    glosses_padded = nn.utils.rnn.pad_sequence(glosses, batch_first=True, padding_value=0)

    result = {
        'keypoints': keypoints_padded,
        'mask': mask,
        'gloss': glosses_padded,
        'gloss_text': [item['gloss_text'] for item in batch],
        'translation': [item['translation'] for item in batch],
        'name': [item['name'] for item in batch],
    }
    
    # Include translation token ids if present (all items must have them)
    if all('translation_ids' in item and item['translation_ids'] is not None for item in batch):
        trans_ids = [item['translation_ids'] for item in batch]
        result['translation_ids'] = nn.utils.rnn.pad_sequence(
            trans_ids, batch_first=True, padding_value=1  # BART pad_token_id=1
        )
    
    return result


# ─── CTC Decoding ────────────────────────────────────────────────────

def ctc_greedy_decode(logits, input_lengths=None):
    """
    Greedy CTC decoding.
    logits: (B, T, V)
    Returns: list of decoded token id lists
    """
    B, T, V = logits.shape
    predictions = []
    for i in range(B):
        valid_len = input_lengths[i].item() if input_lengths is not None else T
        pred = logits[i, :valid_len].argmax(dim=-1).cpu().numpy()
        unique_pred = []
        prev = -1
        for p in pred:
            if p != prev:
                if p != 0:
                    unique_pred.append(int(p))
                prev = p
        predictions.append(unique_pred)
    return predictions


def ctc_beam_decode(logits, input_lengths=None, beam_width=10):
    """
    Prefix beam search CTC decoding with length normalization.
    logits: (B, T, V) — raw logits (will be log_softmax'd)
    Returns: list of decoded token id lists
    """
    if beam_width <= 1:
        return ctc_greedy_decode(logits, input_lengths)

    B, T, V = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    decoded = []

    for b in range(B):
        valid_len = input_lengths[b].item() if input_lengths is not None else T
        
        # beam: dict of {prefix_tuple: log_prob}
        beams = {(): 0.0}

        for t in range(valid_len):
            lp = log_probs[b, t]  # (V,)
            # Only consider top-k tokens per timestep for efficiency
            topk_vals, topk_idx = lp.topk(min(beam_width * 2, V))
            
            new_beams = {}
            for prefix, score in beams.items():
                for k in range(topk_vals.shape[0]):
                    c = topk_idx[k].item()
                    new_score = score + topk_vals[k].item()

                    if c == 0:  # blank
                        key = prefix
                    elif len(prefix) > 0 and prefix[-1] == c:
                        key = prefix  # collapse duplicate
                    else:
                        key = prefix + (c,)

                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            # Prune: keep top beam_width, normalized by length
            scored = [(k, v / max(1, len(k))) for k, v in new_beams.items()]
            scored.sort(key=lambda x: -x[1])
            # Store un-normalized scores for next iteration
            beams = {}
            for k, _ in scored[:beam_width]:
                beams[k] = new_beams[k]

        # Pick best beam (length-normalized)
        if beams:
            best = max(beams.items(), key=lambda x: x[1] / max(1, len(x[0])))
            decoded.append(list(best[0]))
        else:
            decoded.append([])

    return decoded


# ─── Trainer ─────────────────────────────────────────────────────────

class Trainer:
    """Training loop supporting CTC-only and joint CTC+BART modes.
    
    Three-stage training for BART:
      Stage 1 (epochs 0..freeze_bart_epochs): CTC only, BART+proj frozen, 
              encoder learns gloss recognition
      Stage 2 (epochs freeze_bart_epochs..2*freeze_bart_epochs): Unfreeze BART+proj,
              train with detached encoder hidden (BART learns to read encoder features
              without corrupting them)
      Stage 3 (remaining epochs): Full joint training, gradients flow through everything
    """
    
    def __init__(self, model, train_loader, val_loader, tokenizer, device='cuda',
                 bart_tokenizer=None, use_bart=False, ctc_weight=0.3,
                 freeze_bart_epochs=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.device = device
        self.use_bart = use_bart
        self.ctc_weight = ctc_weight
        self.freeze_bart_epochs = freeze_bart_epochs
        self.stage = 1

        self.criterion = CTCLoss(blank=0)
        
        # Separate param groups so we can use different LRs
        encoder_params = []
        bart_params = []
        for name, p in model.named_parameters():
            if 'translation_head' in name:
                bart_params.append(p)
            else:
                encoder_params.append(p)
        
        self.optimizer = torch.optim.AdamW([
            {'params': encoder_params, 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': bart_params, 'lr': 5e-5, 'weight_decay': 0.01},  # lower LR for BART
        ])

        self.total_steps = len(train_loader) * 150
        self.warmup_steps = len(train_loader) * 5
        self.best_val_loss = float('inf')

    def _set_stage(self, epoch):
        """Manage training stages for BART."""
        if not self.use_bart:
            return
        
        if epoch < self.freeze_bart_epochs:
            if self.stage != 1:
                return
            self.stage = 1
            print(f"\n  >>> Stage 1: CTC-only (BART frozen) <<<")
            self.model.freeze_translation()
        elif epoch < self.freeze_bart_epochs * 2:
            if self.stage == 2:
                return
            self.stage = 2
            print(f"\n  >>> Stage 2: BART warming up (encoder detached) <<<")
            # Unfreeze BART but keep _frozen=True so forward() detaches encoder
            if self.model.translation_head:
                for p in self.model.translation_head.bart.parameters():
                    p.requires_grad = True
                for p in self.model.translation_head.encoder_proj.parameters():
                    p.requires_grad = True
                self.model.translation_head._frozen = True  # still detach encoder
        else:
            if self.stage == 3:
                return
            self.stage = 3
            print(f"\n  >>> Stage 3: Full joint training <<<")
            self.model.unfreeze_translation()
            # _frozen = False now, so gradients flow through

    def train_epoch(self, epoch):
        self.model.train()
        self._set_stage(epoch)
        
        total_loss = 0
        total_ctc = 0
        total_trans = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['gloss'].to(self.device)

            step = epoch * len(self.train_loader) + batch_idx
            self._adjust_learning_rate(step)

            self.optimizer.zero_grad()

            if self.use_bart and self.stage >= 2:
                trans_targets = batch.get('translation_ids')
                if trans_targets is not None:
                    trans_targets = trans_targets.to(self.device)
                
                output = self.model(keypoints, mask, translation_targets=trans_targets)
                logits = output['logits']
                mask_out = output['mask']
            else:
                # Stage 1 or non-BART: CTC only, skip BART forward entirely
                if self.use_bart:
                    hidden, mask_out = self.model.encode(keypoints, mask)
                    logits = self.model.head(hidden)
                    output = {'logits': logits, 'mask': mask_out}
                else:
                    logits, mask_out = self.model(keypoints, mask)

            # CTC loss
            logits_ctc = logits.permute(1, 0, 2)
            input_lengths = mask_out.sum(dim=1).long().cpu()
            target_lengths = (targets != 0).sum(dim=1).long().cpu()

            if (target_lengths == 0).any() or (input_lengths < target_lengths).any():
                continue

            ctc_loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)
            
            if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                continue

            # Total loss — scale carefully
            if self.use_bart and self.stage >= 2 and isinstance(output, dict) and 'translation_loss' in output:
                trans_loss = output['translation_loss']
                if not (torch.isnan(trans_loss) or torch.isinf(trans_loss)):
                    loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * trans_loss
                    total_trans += trans_loss.item()
                else:
                    loss = ctc_loss
            else:
                loss = ctc_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_ctc += ctc_loss.item()
            num_batches += 1

            if batch_idx % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f'  Epoch {epoch} [S{self.stage}], Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}'
                if self.use_bart and self.stage >= 2 and isinstance(output, dict) and 'translation_loss' in output:
                    msg += f', CTC: {ctc_loss.item():.4f}, Trans: {output["translation_loss"].item():.4f}'
                msg += f', LR: {lr:.2e}'
                print(msg)

        n = max(1, num_batches)
        return {'loss': total_loss/n, 'ctc': total_ctc/n, 'trans': total_trans/n}

    def _adjust_learning_rate(self, step):
        if step < self.warmup_steps:
            factor = 2 ** -(self.warmup_steps - step)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        # Encoder group: base LR 1e-3
        self.optimizer.param_groups[0]['lr'] = 1e-3 * factor
        # BART group: base LR 5e-5 (20x lower)
        if len(self.optimizer.param_groups) > 1:
            self.optimizer.param_groups[1]['lr'] = 5e-5 * factor

    @torch.no_grad()
    def validate(self, epoch, decode_mode='greedy', beam_width=5):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []

        for batch in self.val_loader:
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['gloss'].to(self.device)

            if self.use_bart:
                output = self.model(keypoints, mask)
                logits = output['logits']
                mask_out = output['mask']
            else:
                logits, mask_out = self.model(keypoints, mask)

            logits_ctc = logits.permute(1, 0, 2)
            input_lengths = mask_out.sum(dim=1).long().cpu()
            target_lengths = (targets != 0).sum(dim=1).long().cpu()

            if (target_lengths > 0).all() and (input_lengths >= target_lengths).all():
                loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    num_batches += 1

            # Decode
            if decode_mode == 'beam':
                preds = ctc_beam_decode(logits.cpu(), input_lengths, beam_width)
            else:
                preds = ctc_greedy_decode(logits.cpu(), input_lengths)
            
            all_predictions.extend(preds)
            all_targets.extend(batch['gloss_text'])

        return total_loss / max(1, num_batches), all_predictions, all_targets

    def train(self, num_epochs=100, decode_mode='greedy', beam_width=5):
        for epoch in range(num_epochs):

            metrics = self.train_epoch(epoch)
            val_loss, predictions, targets = self.validate(
                epoch, decode_mode=decode_mode, beam_width=beam_width)

            print(f'Epoch {epoch}: Train={metrics["loss"]:.4f} '
                  f'(CTC={metrics["ctc"]:.4f} Trans={metrics["trans"]:.4f}) '
                  f'Val={val_loss:.4f}')

            if epoch % 5 == 0:
                for i in range(min(3, len(predictions))):
                    pred_text = self.tokenizer.decode(predictions[i])
                    print(f'  Target: {targets[i]}')
                    print(f'  Pred  : {pred_text}')
                    print('  ---')

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, 'best_model.pt')
                print(f'  ✅ Best model saved (val_loss={val_loss:.4f})')

            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f'checkpoint_epoch_{epoch}.pt')


# ─── Tokenizer ───────────────────────────────────────────────────────

class GlossTokenizer:
    """Word-level tokenizer for gloss sequences. Token 0=<blank>, 1=<unk>."""
    def __init__(self, gloss_sentences=None, min_freq=1):
        self.gloss_to_idx = {'<blank>': 0, '<unk>': 1}
        self.idx_to_gloss = {0: '<blank>', 1: '<unk>'}
        if gloss_sentences is not None:
            self._build_vocab(gloss_sentences, min_freq)

    def _build_vocab(self, gloss_sentences, min_freq=1):
        word_counts = Counter()
        for s in gloss_sentences:
            if isinstance(s, str):
                word_counts.update(s.strip().upper().split())
        for word, count in word_counts.most_common():
            if count >= min_freq and word not in self.gloss_to_idx:
                idx = len(self.gloss_to_idx)
                self.gloss_to_idx[word] = idx
                self.idx_to_gloss[idx] = word
        print(f"GlossTokenizer: {len(self.gloss_to_idx)} word tokens")

    def encode(self, gloss_text):
        if isinstance(gloss_text, str):
            words = gloss_text.strip().upper().split()
            return torch.tensor([self.gloss_to_idx.get(w, 1) for w in words], dtype=torch.long)
        return torch.tensor([self.gloss_to_idx.get(w, 1) for w in gloss_text], dtype=torch.long)

    def decode(self, indices):
        if torch.is_tensor(indices):
            indices = indices.cpu().numpy()
        if isinstance(indices, (int, np.integer)):
            return self.idx_to_gloss.get(int(indices), '<unk>')
        return ' '.join(self.idx_to_gloss.get(int(i), '<unk>') for i in indices if int(i) > 1)

    @property
    def vocab_size(self):
        return len(self.gloss_to_idx)