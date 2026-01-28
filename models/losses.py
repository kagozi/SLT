# models/losses.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """
    Standard label smoothing over vocabulary, ignoring padding positions.

    - logits: (N, V)
    - target: (N,)
    """
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = int(vocab_size)
        self.padding_idx = int(padding_idx)
        self.smoothing = float(smoothing)
        self.confidence = 1.0 - self.smoothing

        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if not (0 <= self.padding_idx < self.vocab_size):
            raise ValueError("padding_idx out of range")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 2:
            raise ValueError(f"logits must be (N, V), got {tuple(logits.shape)}")
        if target.dim() != 1:
            raise ValueError(f"target must be (N,), got {tuple(target.shape)}")
        if logits.size(0) != target.size(0):
            raise ValueError("logits and target batch dimension mismatch")

        # (N, V)
        log_probs = F.log_softmax(logits, dim=-1)

        # mask out padding positions
        mask = target.ne(self.padding_idx)
        if mask.sum().item() == 0:
            # all padding: return a zero scalar with grad
            return log_probs.sum() * 0.0

        log_probs = log_probs[mask]  # (M, V)
        target = target[mask]        # (M,)

        # Build smoothed distribution: uniform over all non-pad tokens
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)  # (M, V)
            # distribute smoothing over V-1 (excluding pad)
            denom = self.vocab_size - 1
            true_dist.fill_(self.smoothing / denom)
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0.0

        # KL(true || log_probs) is equivalent to cross-entropy with soft targets
        # Using sum over vocab then mean over tokens
        loss = F.kl_div(log_probs, true_dist, reduction="batchmean", log_target=False)
        return loss
