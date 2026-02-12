"""
Per-stream encoders for the three modalities:
  - RGB: lightweight CNN backbone → temporal transformer
  - Hands: dual-hand CNN → temporal transformer
  - Keypoints: linear projection → Conv1D + transformer (closest to reference arch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ..models.blocks import (
    Conv1DBlock,
    TransformerBlock,
    SinusoidalPositionalEncoding,
    ECA,
)


# ─────────────────────────────────────────────
# Lightweight CNN feature extractor for RGB/Hands
# ─────────────────────────────────────────────

class LightweightCNN(nn.Module):
    """
    Small CNN to extract spatial features from image frames.
    Processes each frame independently, outputs a feature vector.
    """

    def __init__(self, in_channels: int = 3, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # (B*T, 3, H, W) -> downsample through conv blocks
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # -> (B*T, 256, 1, 1)
        )
        self.proj = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T, C, H, W) -> (B*T, out_dim)"""
        feat = self.features(x).flatten(1)  # (B*T, 256)
        return self.proj(feat)


# ─────────────────────────────────────────────
# Temporal Encoder (shared structure for all streams)
# ─────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Conv1D blocks + Transformer layers operating on (B, T, D).
    architecture: [Conv1D_11, Conv1D_5, Conv1D_3, Transformer] × N.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_expand: int = 2,
        dropout: float = 0.2,
        attn_dropout: float = 0.2,
        conv_kernels: Tuple[int, ...] = (11, 5, 3),
        max_len: int = 300,
        use_eca: bool = True,
    ):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim, bias=False)
        self.pe = SinusoidalPositionalEncoding(max_len, hidden_dim)
        self.stem_bn = nn.BatchNorm1d(hidden_dim)

        layers = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleList()
            for k in conv_kernels:
                block.append(Conv1DBlock(hidden_dim, hidden_dim, kernel_size=k, dropout=dropout))
            block.append(TransformerBlock(hidden_dim, num_heads, ff_expand, attn_dropout, dropout))
            layers.append(block)
        self.layers = layers
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D_in) → (B, T, hidden_dim)
        mask: (B, T) bool
        """
        x = self.stem(x)
        x = self.pe(x)
        x = self.stem_bn(x.transpose(1, 2)).transpose(1, 2)

        for block in self.layers:
            for module in block:
                x = module(x, mask) if hasattr(module, 'forward') else module(x)

        return self.norm(x)


# ─────────────────────────────────────────────
# RGB Stream Encoder
# ─────────────────────────────────────────────

class RGBStreamEncoder(nn.Module):
    """
    Processes RGB frames: CNN per-frame → temporal encoder.
    Input: (B, T, C, H, W)
    Output: (B, T, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.2, pretrained_backbone: bool = False):
        super().__init__()
        self.cnn = LightweightCNN(in_channels=3, out_dim=hidden_dim)
        self.temporal = TemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C, H, W = x.shape
        # Process all frames through CNN
        x = x.reshape(B * T, C, H, W)
        x = self.cnn(x)  # (B*T, hidden_dim)
        x = x.reshape(B, T, -1)  # (B, T, hidden_dim)
        return self.temporal(x, mask)


# ─────────────────────────────────────────────
# Hands Stream Encoder
# ─────────────────────────────────────────────

class HandsStreamEncoder(nn.Module):
    """
    Processes hand crops: separate CNN for left/right → concat → temporal encoder.
    Input: (B, T, 2, C, H, W) — dim 2 = [left_hand, right_hand]
    Output: (B, T, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 256, num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        hand_feat_dim = hidden_dim // 2
        self.left_cnn = LightweightCNN(in_channels=3, out_dim=hand_feat_dim)
        self.right_cnn = LightweightCNN(in_channels=3, out_dim=hand_feat_dim)
        # Fuse left + right hand features
        self.hand_fuse = nn.Sequential(
            nn.Linear(hand_feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.temporal = TemporalEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _, C, H, W = x.shape
        left = x[:, :, 0]  # (B, T, C, H, W)
        right = x[:, :, 1]

        left_feat = self.left_cnn(left.reshape(B * T, C, H, W)).reshape(B, T, -1)
        right_feat = self.right_cnn(right.reshape(B * T, C, H, W)).reshape(B, T, -1)

        combined = torch.cat([left_feat, right_feat], dim=-1)  # (B, T, hidden_dim)
        combined = self.hand_fuse(combined)
        return self.temporal(combined, mask)


# ─────────────────────────────────────────────
# Keypoints Stream Encoder
# ─────────────────────────────────────────────

class KeypointsStreamEncoder(nn.Module):
    """
    Processes keypoint sequences: linear proj → Conv1D + Transformer.
    Closest to the reference TF model architecture.
    Input: (B, T, D_kpts)  where D_kpts=498
    Output: (B, T, hidden_dim)
    """

    def __init__(self, input_dim: int = 498, hidden_dim: int = 256,
                 num_layers: int = 4, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.temporal = TemporalEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.temporal(x, mask)