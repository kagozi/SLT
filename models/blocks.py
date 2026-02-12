"""
Core building blocks: ECA, DepthwiseConv1D, Conv1DBlock, TransformerBlock.
Ported from TF reference to PyTorch with improvements.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ECA(nn.Module):
    """
    Efficient Channel Attention (from reference code).
    Performs lightweight channel recalibration via 1D conv on channel-pooled features.
    """

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 1D conv on the channel dimension
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
            mask: (B, T) bool, True = valid
        """
        if mask is not None:
            # Zero out padded positions before pooling
            x_masked = x * mask.unsqueeze(-1).float()
            # Average over valid positions
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).float()  # (B, 1)
            pooled = x_masked.sum(dim=1) / lengths  # (B, C)
        else:
            pooled = x.mean(dim=1)  # (B, C)

        # (B, C) -> (B, 1, C) -> conv -> (B, 1, C) -> sigmoid -> (B, 1, C)
        attn = self.conv(pooled.unsqueeze(1))
        attn = torch.sigmoid(attn)  # (B, 1, C)
        return x * attn


class DepthwiseConv1D(nn.Module):
    """Masked depthwise separable 1D conv (from reference MaskingDWConv1D)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, T, C), mask: (B, T) bool."""
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class Conv1DBlock(nn.Module):
    """
    Efficient Conv1D block from reference: expand → DWConv → BN → ECA → project.
    With residual connection when input/output dims match.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.2,
        expand_ratio: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        expanded = in_channels * expand_ratio
        act_fn = nn.GELU() if activation == "gelu" else nn.Tanh()

        self.expand = nn.Sequential(nn.Linear(in_channels, expanded), act_fn)
        self.dwconv = DepthwiseConv1D(expanded, kernel_size, dilation)
        self.bn = nn.BatchNorm1d(expanded)
        self.eca = ECA(expanded)
        self.project = nn.Linear(expanded, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (in_channels == out_channels)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.expand(x)
        x = self.dwconv(x, mask)
        # BN expects (B, C, T)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.eca(x, mask)
        x = self.project(x)
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional mask (from reference)."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D)
        mask: (B, T) bool — True = valid token
        """
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D_h)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T)
            attn_mask = mask[:, None, None, :].float()
            attn = attn.masked_fill(attn_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, self.dim)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block (from reference): LN → MHSA → residual → LN → FFN → residual.
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 4,
        ff_expand: int = 4,
        attn_dropout: float = 0.2,
        ff_dropout: float = 0.2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, attn_dropout)
        self.drop1 = nn.Dropout(ff_dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ff_expand),
            nn.GELU(),
            nn.Linear(dim * ff_expand, dim),
            nn.Dropout(ff_dropout),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x = x + self.drop1(self.attn(self.norm1(x), mask))
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE (from reference positional_encoding)."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → adds PE."""
        return x + self.pe[:, : x.size(1), :]