import io
import math
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb


def _fig_to_wandb(fig) -> wandb.Image:
    """Convert a matplotlib Figure to a wandb.Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    from PIL import Image as _PILImage
    return wandb.Image(_PILImage.open(buf).copy())


class CTCLoss(nn.Module):
    """CTC loss with optional uniform label smoothing.

    Mixes the hard CTC target with a uniform prior (entropy regularisation).
    Following Pereyra et al. (2017) "Regularizing Neural Networks by
    Penalizing Confident Output Distributions":

        loss = (1 - α) * CTC_hard + α * (−mean log_prob)

    α=0.1 consistently reduces over-confident CTC alignments and yields
    1–2 WER point improvements without any architectural changes.
    """
    def __init__(self, blank=0, reduction='mean', smoothing=0.1):
        super().__init__()
        self.ctc_loss  = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
        self.smoothing = smoothing

    def forward(self, logits, targets, input_lengths, target_lengths):
        log_probs = F.log_softmax(logits, dim=-1)
        hard_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        if self.smoothing > 0:
            smooth_loss = -log_probs.mean()   # uniform prior over all timesteps × vocab
            return (1.0 - self.smoothing) * hard_loss + self.smoothing * smooth_loss
        return hard_loss


def augment_keypoints(
    keypoints: torch.Tensor,
    mask: torch.Tensor,
    noise_std: float = 0.01,
    scale_range: tuple = (0.9, 1.1),
) -> torch.Tensor:
    """Spatial augmentation for contrastive learning (second-view generation).

    Applies per-sample random scale + additive Gaussian noise on valid frames.
    Noise is masked so padded frames stay exactly zero (preserving mask validity).

    Used in SimCLR-style NT-Xent: call twice on the same batch to get two
    independent augmented views without any label information.

    Args:
        keypoints: (B, T, D) float tensor
        mask:      (B, T) float tensor, 1=valid 0=padding
        noise_std: std of additive Gaussian noise
        scale_range: (lo, hi) for uniform per-sample scale factor
    """
    B = keypoints.size(0)
    scale = keypoints.new_empty(B, 1, 1).uniform_(*scale_range)
    x = keypoints * scale
    noise = torch.randn_like(x) * noise_std
    x = x + noise * mask.unsqueeze(-1)   # noise only on valid frames
    return x


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
                 freeze_bart_epochs=5, models_dir=None, grad_accum=1, fp16=False,
                 contrastive_weight=0.0, rdrop_weight=0.0, ctc_smoothing=0.1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer
        self.device = device
        self.use_bart = use_bart
        self.ctc_weight = ctc_weight
        self.freeze_bart_epochs = freeze_bart_epochs
        self.grad_accum = max(1, grad_accum)
        self.fp16 = fp16 and (device == 'cuda' or str(device).startswith('cuda'))
        self.scaler = torch.cuda.amp.GradScaler() if self.fp16 else None
        self.models_dir = Path(models_dir) if models_dir else Path('.')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.stage = 1
        self.contrastive_weight = contrastive_weight
        self.rdrop_weight       = rdrop_weight

        # Optional contrastive projection head — registered on the model so it
        # is saved/loaded with the checkpoint automatically.
        if contrastive_weight > 0:
            from models import SignContrastiveLoss
            model.contrastive_head = SignContrastiveLoss(
                encoder_dim=model.dim, proj_dim=128, temperature=0.07
            ).to(device)
            print(f"  ✅ Contrastive head added (weight={contrastive_weight}, τ=0.07)")
        if rdrop_weight > 0:
            print(f"  ✅ R-Drop enabled (weight={rdrop_weight})")

        # Per-epoch history for plotting
        self.history: dict = {
            'epoch':            [],
            'train_loss':       [], 'train_ctc':         [], 'train_trans':   [],
            'train_contrastive':[], 'train_rdrop':        [],
            'val_loss':         [], 'val_wer':            [],
            'val_bleu1':        [], 'val_bleu4':          [],
            'lr':               [],
        }

        self.criterion = CTCLoss(blank=0, smoothing=ctc_smoothing)

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
            {'params': bart_params,    'lr': 5e-5, 'weight_decay': 0.01},
        ])

        self.total_steps = len(train_loader) * 150
        self.warmup_steps = len(train_loader) * 5
        self.best_val_loss = float('inf')

    def _set_stage(self, epoch):
        """Manage training stages for BART."""
        if not self.use_bart:
            return
        
        # Glossless mode (ctc_weight=0): skip Stage 1, go straight to BART training
        if self.ctc_weight == 0:
            if self.stage != 3:
                self.stage = 3
                print(f"\n  >>> Glossless mode: BART-only training (no CTC) <<<")
                self.model.unfreeze_translation()
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
        
        total_loss        = 0
        total_ctc         = 0
        total_trans       = 0
        total_contrastive = 0
        total_rdrop       = 0
        num_batches       = 0

        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(self.train_loader):
            keypoints = batch['keypoints'].to(self.device)
            mask = batch['mask'].to(self.device)
            targets = batch['gloss'].to(self.device)

            step = epoch * len(self.train_loader) + batch_idx
            self._adjust_learning_rate(step)

            with torch.cuda.amp.autocast(enabled=self.fp16):
                # ── Glossless mode: BART-only, no CTC ──
                if self.use_bart and self.ctc_weight == 0:
                    trans_targets = batch.get('translation_ids')
                    if trans_targets is None:
                        continue
                    trans_targets = trans_targets.to(self.device)

                    output = self.model(keypoints, mask, translation_targets=trans_targets)

                    if 'translation_loss' not in output:
                        continue
                    trans_loss = output['translation_loss']
                    if torch.isnan(trans_loss) or torch.isinf(trans_loss):
                        continue

                    loss = trans_loss
                    total_trans += trans_loss.item()

                    # ── Contrastive loss (glossless: encoder shaped by BART
                    # gradients only — contrastive adds discriminative pressure) ──
                    if self.contrastive_weight > 0 and hasattr(self.model, 'contrastive_head'):
                        aug1 = augment_keypoints(keypoints, mask)
                        aug2 = augment_keypoints(keypoints, mask)
                        h1, m1 = self.model.encode(aug1, mask.clone())
                        h2, m2 = self.model.encode(aug2, mask.clone())
                        c_loss = self.model.contrastive_head(h1, m1, h2, m2)
                        if not (torch.isnan(c_loss) or torch.isinf(c_loss)):
                            loss = loss + self.contrastive_weight * c_loss
                            total_contrastive += c_loss.item()

                # ── Joint CTC + BART or CTC-only ──
                else:
                    if self.use_bart and self.stage >= 2:
                        trans_targets = batch.get('translation_ids')
                        if trans_targets is not None:
                            trans_targets = trans_targets.to(self.device)

                        output = self.model(keypoints, mask, translation_targets=trans_targets)
                        logits = output['logits']
                        mask_out = output['mask']
                    else:
                        # Stage 1 or non-BART: CTC only
                        if self.use_bart:
                            hidden, mask_out = self.model.encode(keypoints, mask)
                            logits = self.model.head(hidden)
                            output = {'logits': logits, 'mask': mask_out}
                        else:
                            logits, mask_out = self.model(keypoints, mask)

                    # CTC loss
                    logits_ctc = logits.float().permute(1, 0, 2)
                    input_lengths = mask_out.sum(dim=1).long().cpu()
                    target_lengths = (targets != 0).sum(dim=1).long().cpu()

                    if (target_lengths == 0).any() or (input_lengths < target_lengths).any():
                        continue

                    ctc_loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)

                    if torch.isnan(ctc_loss) or torch.isinf(ctc_loss):
                        continue

                    # Combine CTC + translation losses
                    if self.use_bart and self.stage >= 2 and isinstance(output, dict) and 'translation_loss' in output:
                        trans_loss = output['translation_loss']
                        if not (torch.isnan(trans_loss) or torch.isinf(trans_loss)):
                            loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * trans_loss
                            total_trans += trans_loss.item()
                        else:
                            loss = ctc_loss
                    else:
                        loss = ctc_loss

                    total_ctc += ctc_loss.item()

                    # ── R-Drop: symmetric KL between two CTC forward passes ──
                    # (Liang et al., NeurIPS 2021 — α≈5 for sequence models)
                    if self.rdrop_weight > 0:
                        if self.use_bart:
                            h2_rd, _ = self.model.encode(keypoints, mask)
                            logits2_rd = self.model.head(h2_rd)
                        else:
                            logits2_rd, _ = self.model(keypoints, mask)
                        p1 = F.log_softmax(logits.float(),      dim=-1)  # (B, T, V)
                        p2 = F.log_softmax(logits2_rd.float(),  dim=-1)
                        valid = mask_out.bool()
                        kl_12 = F.kl_div(p1, p2.exp(), reduction='none').sum(-1)
                        kl_21 = F.kl_div(p2, p1.exp(), reduction='none').sum(-1)
                        rdrop_loss = 0.5 * (kl_12[valid] + kl_21[valid]).mean()
                        if not (torch.isnan(rdrop_loss) or torch.isinf(rdrop_loss)):
                            loss = loss + self.rdrop_weight * rdrop_loss
                            total_rdrop += rdrop_loss.item()

                    # ── Contrastive loss ──────────────────────────────────────
                    if self.contrastive_weight > 0 and hasattr(self.model, 'contrastive_head'):
                        aug1 = augment_keypoints(keypoints, mask)
                        aug2 = augment_keypoints(keypoints, mask)
                        h1, m1 = self.model.encode(aug1, mask.clone())
                        h2, m2 = self.model.encode(aug2, mask.clone())
                        c_loss = self.model.contrastive_head(h1, m1, h2, m2)
                        if not (torch.isnan(c_loss) or torch.isinf(c_loss)):
                            loss = loss + self.contrastive_weight * c_loss
                            total_contrastive += c_loss.item()

            scaled = loss / self.grad_accum
            if self.fp16:
                self.scaler.scale(scaled).backward()
            else:
                scaled.backward()
            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % self.grad_accum == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.fp16:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if batch_idx % 50 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                msg = f'  Epoch {epoch} [S{self.stage}], Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}'
                if total_ctc > 0:
                    msg += f', CTC: {ctc_loss.item():.4f}'
                if total_trans > 0 and self.use_bart:
                    tl = output.get('translation_loss')
                    if tl is not None:
                        msg += f', Trans: {tl.item():.4f}'
                msg += f', LR: {lr:.2e}'
                print(msg)

        n = max(1, num_batches)
        return {
            'loss':        total_loss        / n,
            'ctc':         total_ctc         / n,
            'trans':       total_trans       / n,
            'contrastive': total_contrastive / n,
            'rdrop':       total_rdrop       / n,
        }

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

    @staticmethod
    def _compute_wer_texts(preds: list, targets: list) -> float:
        """Standard WER (edit distance / reference length)."""
        def _wer(p, t):
            pw, tw = p.split(), t.split()
            if not tw:
                return 0.0
            dp = np.zeros((len(tw) + 1, len(pw) + 1))
            for i in range(len(tw) + 1):
                dp[i, 0] = i
            for j in range(len(pw) + 1):
                dp[0, j] = j
            for i in range(1, len(tw) + 1):
                for j in range(1, len(pw) + 1):
                    cost = 0 if tw[i - 1] == pw[j - 1] else 1
                    dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
            return dp[len(tw), len(pw)] / len(tw)
        if not preds:
            return 1.0
        return float(np.mean([_wer(p, t) for p, t in zip(preds, targets)]))

    @staticmethod
    @staticmethod
    def _normalize_text(text: str) -> str:
        """Lowercase, remove punctuation, collapse whitespace."""
        import re
        text = text.lower()
        text = re.sub(r"[^\w\s']", ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _compute_bleu_texts(self, preds: list, targets: list) -> dict:
        """sacrebleu corpus BLEU — standard 13a tokenization + lowercase.
        Returns scores in [0, 1] range to match the rest of the codebase."""
        from sacrebleu.metrics import BLEU
        bleu = BLEU(tokenize='13a', lowercase=True)
        result = bleu.corpus_score(preds, [targets])
        return {
            'BLEU-1': result.precisions[0] / 100,
            'BLEU-2': result.precisions[1] / 100,
            'BLEU-4': result.precisions[3] / 100,
            'BLEU':   result.score         / 100,
        }

    @torch.no_grad()
    def validate(self, decode_mode='greedy', beam_width=5):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        all_trans_preds = []
        all_trans_targets = []

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

            # CTC val loss (skip in glossless mode)
            if self.ctc_weight > 0:
                logits_ctc = logits.permute(1, 0, 2)
                input_lengths = mask_out.sum(dim=1).long().cpu()
                target_lengths = (targets != 0).sum(dim=1).long().cpu()

                if (target_lengths > 0).all() and (input_lengths >= target_lengths).all():
                    loss = self.criterion(logits_ctc, targets, input_lengths, target_lengths)
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        num_batches += 1

                # CTC decode
                if decode_mode == 'beam':
                    preds = ctc_beam_decode(logits.cpu(), input_lengths, beam_width)
                else:
                    preds = ctc_greedy_decode(logits.cpu(), input_lengths)
                all_predictions.extend(preds)
                all_targets.extend(batch['gloss_text'])

            # BART translation eval (for glossless or joint mode)
            if self.use_bart and self.bart_tokenizer is not None and self.stage >= 2:
                try:
                    token_ids = self.model.translate(keypoints, mask, beam_width=beam_width)
                    for i in range(token_ids.shape[0]):
                        text = self.bart_tokenizer.decode(token_ids[i], skip_special_tokens=True)
                        all_trans_preds.append(text)
                    all_trans_targets.extend(batch['translation'])
                    
                    # For glossless mode, use translation loss as val_loss
                    if self.ctc_weight == 0:
                        trans_targets = batch.get('translation_ids')
                        if trans_targets is not None:
                            trans_targets = trans_targets.to(self.device)
                            t_out = self.model(keypoints, mask, translation_targets=trans_targets)
                            if 'translation_loss' in t_out:
                                tl = t_out['translation_loss']
                                if not (torch.isnan(tl) or torch.isinf(tl)):
                                    total_loss += tl.item()
                                    num_batches += 1
                except Exception:
                    pass

        avg_loss = total_loss / max(1, num_batches)
        
        # Return translation predictions for glossless mode
        if self.ctc_weight == 0 and all_trans_preds:
            return avg_loss, all_trans_preds, all_trans_targets
        
        return avg_loss, all_predictions, all_targets

    def train(self, num_epochs=100, decode_mode='greedy', beam_width=5):
        # ── Disk space guard: abort early rather than crash mid-training ──────
        import shutil as _shutil
        disk = _shutil.disk_usage(self.models_dir)
        free_gb = disk.free / 1e9
        if free_gb < 20:
            raise RuntimeError(
                f"Insufficient disk space: {free_gb:.1f} GB free at "
                f"{self.models_dir}. Need at least 20 GB. Free up space and retry.")
        print(f"  Disk: {free_gb:.0f} GB free at {self.models_dir.parent.parent}")

        for epoch in range(num_epochs):

            metrics = self.train_epoch(epoch)
            val_loss, predictions, targets = self.validate(
                decode_mode=decode_mode, beam_width=beam_width)

            # ── Decode predictions → text for WER / BLEU ──
            pred_texts = [
                p if isinstance(p, str) else self.tokenizer.decode(p)
                for p in predictions
            ]
            val_wer  = self._compute_wer_texts(pred_texts, targets) if pred_texts else 1.0
            val_bleu = self._compute_bleu_texts(pred_texts, targets) if pred_texts else {}

            current_lr = self.optimizer.param_groups[0]['lr']

            # ── Accumulate history ──
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(metrics['loss'])
            self.history['train_ctc'].append(metrics['ctc'])
            self.history['train_trans'].append(metrics['trans'])
            self.history['train_contrastive'].append(metrics['contrastive'])
            self.history['train_rdrop'].append(metrics['rdrop'])
            self.history['val_loss'].append(val_loss)
            self.history['val_wer'].append(val_wer)
            self.history['val_bleu1'].append(val_bleu.get('BLEU-1', 0))
            self.history['val_bleu4'].append(val_bleu.get('BLEU-4', 0))
            self.history['lr'].append(current_lr)

            print(f'Epoch {epoch:3d}: Train={metrics["loss"]:.4f} '
                  f'(CTC={metrics["ctc"]:.4f} Trans={metrics["trans"]:.4f}) '
                  f'Val={val_loss:.4f}  WER={val_wer:.3f}  '
                  f'B1={val_bleu.get("BLEU-1",0):.3f}  B4={val_bleu.get("BLEU-4",0):.3f}  '
                  f'LR={current_lr:.2e}')

            # ── W&B per-epoch metrics ──
            log_dict = {
                'epoch':                    epoch,
                'train/loss':               metrics['loss'],
                'train/ctc_loss':           metrics['ctc'],
                'train/trans_loss':         metrics['trans'],
                'train/contrastive_loss':   metrics['contrastive'],
                'train/rdrop_loss':         metrics['rdrop'],
                'train/lr':                 current_lr,
                'train/stage':              self.stage,
                'val/loss':                 val_loss,
                'val/wer':                  val_wer,
                'val/bleu1':                val_bleu.get('BLEU-1', 0),
                'val/bleu2':                val_bleu.get('BLEU-2', 0),
                'val/bleu4':                val_bleu.get('BLEU-4', 0),
            }
            if wandb.run is not None:
                wandb.log(log_dict, step=epoch)

            # ── Sample predictions every 5 epochs ──
            if epoch % 5 == 0:
                sample_rows = []
                for i in range(min(5, len(pred_texts))):
                    target_text = targets[i] if isinstance(targets[i], str) else str(targets[i])
                    pred_text   = pred_texts[i]
                    print(f'  Target: {self._normalize_text(target_text)}')
                    print(f'  Pred  : {self._normalize_text(pred_text)}')
                    print('  ---')
                    sample_rows.append([epoch, target_text, pred_text])
                if wandb.run is not None and sample_rows:
                    wandb.log({
                        'val/samples': wandb.Table(
                            columns=['epoch', 'target', 'prediction'],
                            data=sample_rows,
                        )
                    }, step=epoch)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = self.models_dir / 'best_model.pt'
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss':             val_loss,
                    'val_wer':              val_wer,
                    'val_bleu4':            val_bleu.get('BLEU-4', 0),
                }, best_path)
                print(f'  ✅ Best model saved (val_loss={val_loss:.4f}  '
                      f'WER={val_wer:.3f}  BLEU-4={val_bleu.get("BLEU-4",0):.4f}) → {best_path}')
                if wandb.run is not None:
                    wandb.log({
                        'val/best_loss':  val_loss,
                        'val/best_wer':   val_wer,
                        'val/best_bleu4': val_bleu.get('BLEU-4', 0),
                    }, step=epoch)


        # ── Plot training curves at end of training ──
        self._plot_training_curves()

    def _plot_training_curves(self):
        """Plot all training metrics as matplotlib figures and log to W&B."""
        h = self.history
        if not h['epoch']:
            return
        ep = h['epoch']

        def _ax_style(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3, linestyle='--')
            ax.spines[['top', 'right']].set_visible(False)

        plots = {}

        # 1. Loss: train + val
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, h['train_loss'], label='Train',  color='royalblue',  linewidth=1.8)
        ax.plot(ep, h['val_loss'],   label='Val',    color='tomato',     linewidth=1.8)
        _ax_style(ax, 'Loss Curve', 'Epoch', 'Loss')
        ax.legend()
        fig.tight_layout()
        plots['plots/loss_curve'] = _fig_to_wandb(fig)

        # 2. CTC vs Translation loss (only when BART is active)
        if any(v > 0 for v in h['train_trans']):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(ep, h['train_ctc'],   label='CTC Loss',         color='steelblue',   linewidth=1.8)
            ax.plot(ep, h['train_trans'], label='Translation Loss',  color='darkorange',  linewidth=1.8)
            _ax_style(ax, 'CTC vs Translation Loss', 'Epoch', 'Loss')
            ax.legend()
            fig.tight_layout()
            plots['plots/ctc_vs_trans_loss'] = _fig_to_wandb(fig)

        # 3. Val WER
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, h['val_wer'], color='crimson', linewidth=1.8)
        _ax_style(ax, 'Validation WER', 'Epoch', 'WER (lower is better)')
        fig.tight_layout()
        plots['plots/val_wer'] = _fig_to_wandb(fig)

        # 4. Val BLEU-1 & BLEU-4
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(ep, h['val_bleu1'], label='BLEU-1', color='mediumseagreen', linewidth=1.8)
        ax.plot(ep, h['val_bleu4'], label='BLEU-4', color='darkgreen',      linewidth=1.8)
        _ax_style(ax, 'Validation BLEU', 'Epoch', 'BLEU (higher is better)')
        ax.legend()
        fig.tight_layout()
        plots['plots/val_bleu'] = _fig_to_wandb(fig)

        # 5. Combined: val WER + BLEU-4 (dual-axis)
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax2 = ax1.twinx()
        ax1.plot(ep, h['val_wer'],   color='crimson',   linewidth=1.8, label='WER')
        ax2.plot(ep, h['val_bleu4'], color='darkgreen', linewidth=1.8, linestyle='--', label='BLEU-4')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('WER',    color='crimson')
        ax2.set_ylabel('BLEU-4', color='darkgreen')
        ax1.set_title('Val WER & BLEU-4 (dual axis)', fontsize=12, fontweight='bold')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        ax1.grid(alpha=0.3, linestyle='--')
        fig.tight_layout()
        plots['plots/val_wer_bleu4_combined'] = _fig_to_wandb(fig)

        # 6. Learning rate schedule
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ep, h['lr'], color='mediumpurple', linewidth=1.8)
        _ax_style(ax, 'Learning Rate Schedule', 'Epoch', 'LR')
        fig.tight_layout()
        plots['plots/lr_schedule'] = _fig_to_wandb(fig)

        if wandb.run is not None:
            wandb.log(plots)
            print(f"  📈 {len(plots)} training curve plots saved to W&B.")


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