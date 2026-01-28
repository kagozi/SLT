"""
RGB video encoder for sign language recognition.

Architecture:
    RGB [B, T, 3, H, W]
        → Spatial CNN (per-frame)
        → Temporal Transformer
        → Features [B, T, D]
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torchvision.models as models


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences (batch_first)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, : x.size(1), :].to(dtype=x.dtype, device=x.device)
        return self.dropout(x)


def _get_backbone(backbone: str, pretrained: bool):
    """
    Returns (feature_extractor_module, backbone_out_dim).

    Uses torchvision "weights=" API when available.
    """
    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        out_dim = 512
        # keep avgpool, drop fc
        feat = nn.Sequential(*list(net.children())[:-1])

    elif backbone == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        net = models.resnet34(weights=weights)
        out_dim = 512
        feat = nn.Sequential(*list(net.children())[:-1])

    elif backbone == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        out_dim = 2048
        feat = nn.Sequential(*list(net.children())[:-1])

    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        out_dim = 1280
        # keep features + avgpool, drop classifier
        feat = nn.Sequential(*list(net.children())[:-1])

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return feat, out_dim


class SpatialCNNEncoder(nn.Module):
    """
    Per-frame spatial feature extraction using pretrained CNN backbone.
    """

    def __init__(
        self,
        d_model: int = 512,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone_name = backbone
        self.backbone, backbone_out_dim = _get_backbone(backbone, pretrained=pretrained)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Linear(backbone_out_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, H, W]

        Returns:
            features: [B, T, D]
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)

        feats = self.backbone(x)  # typically [B*T, D_backbone, 1, 1]
        if feats.dim() == 4:
            feats = feats.flatten(1)  # [B*T, D_backbone]

        feats = self.proj(feats)
        feats = self.dropout(feats)

        feats = feats.reshape(B, T, -1)
        return feats


class TemporalTransformerEncoder(nn.Module):
    """Temporal modeling using Transformer encoder (batch_first)."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        norm_first: bool = False,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        return x


class RGBVideoEncoder(nn.Module):
    """Spatial CNN + Temporal Transformer."""

    def __init__(
        self,
        d_model: int = 512,
        spatial_backbone: str = "resnet18",
        spatial_pretrained: bool = True,
        spatial_freeze: bool = False,
        temporal_nhead: int = 8,
        temporal_layers: int = 4,
        temporal_dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.spatial_encoder = SpatialCNNEncoder(
            d_model=d_model,
            backbone=spatial_backbone,
            pretrained=spatial_pretrained,
            freeze_backbone=spatial_freeze,
            dropout=dropout,
        )

        self.temporal_encoder = TemporalTransformerEncoder(
            d_model=d_model,
            nhead=temporal_nhead,
            num_layers=temporal_layers,
            dim_feedforward=temporal_dim_feedforward,
            dropout=dropout,
        )

    def forward(self, rgb: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        spatial = self.spatial_encoder(rgb)  # [B,T,D]
        temporal = self.temporal_encoder(spatial, src_key_padding_mask=src_key_padding_mask)  # [B,T,D]
        return temporal

    def get_feature_dim(self) -> int:
        return self.spatial_encoder.proj.out_features


if __name__ == "__main__":
    encoder = RGBVideoEncoder(
        d_model=256,
        spatial_backbone="resnet18",
        spatial_pretrained=False,
        temporal_layers=2,
    )

    B, T, C, H, W = 2, 64, 3, 256, 256
    rgb = torch.randn(B, T, C, H, W)
    src_mask = torch.zeros(B, T, dtype=torch.bool)
    src_mask[:, -10:] = True

    y = encoder(rgb, src_key_padding_mask=src_mask)
    print("Output:", y.shape)
