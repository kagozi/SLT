"""
Multi-Stream Fusion Module.
Combines encoded representations from RGB, Hands, and Keypoints streams.

Supports three fusion strategies:
  - cross_attention: learnable cross-attention between streams (default, best quality)
  - concat: simple concatenation + linear projection
  - gated: gated mixture with learned stream importance weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from models.blocks import MultiHeadSelfAttention, TransformerBlock


class CrossAttentionFusionLayer(nn.Module):
    """
    Each stream attends to all other streams via cross-attention,
    then results are aggregated.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Cross-attention: query from one stream, key/value from concatenated others
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # FFN after cross-attention
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        query: (B, T, D) — the stream being updated
        context: (B, T_ctx, D) — concatenation of other streams
        mask: (B, T_ctx) — mask for context
        """
        B, T, D = query.shape

        q = self.q_proj(self.norm_q(query))
        kv = self.kv_proj(self.norm_kv(context))

        q = q.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = kv.reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, T, D)
        out = self.out_proj(out)

        # Residual + FFN
        x = query + out
        x = x + self.ffn(x)
        return x


class MultiStreamFusion(nn.Module):
    """
    Fuses three encoded streams into a single representation.

    Strategies:
      - cross_attention: each stream attends to others, then mean pool
      - concat: concat along feature dim, project down
      - gated: learned sigmoid gates per stream
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        strategy: str = "cross_attention",
    ):
        super().__init__()
        self.strategy = strategy
        self.hidden_dim = hidden_dim

        if strategy == "cross_attention":
            self.cross_layers = nn.ModuleList()
            for _ in range(num_layers):
                self.cross_layers.append(nn.ModuleDict({
                    "rgb_cross": CrossAttentionFusionLayer(hidden_dim, num_heads, dropout),
                    "hands_cross": CrossAttentionFusionLayer(hidden_dim, num_heads, dropout),
                    "kpts_cross": CrossAttentionFusionLayer(hidden_dim, num_heads, dropout),
                }))
            # Final aggregation
            self.agg_proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )

        elif strategy == "concat":
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )

        elif strategy == "gated":
            self.gate_rgb = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.gate_hands = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.gate_kpts = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
            self.out_proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )

        # Self-attention refinement after fusion
        self.refine = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_expand=4, attn_dropout=dropout, ff_dropout=dropout)
            for _ in range(2)
        ])

    def forward(
        self,
        rgb_enc: torch.Tensor,
        hands_enc: torch.Tensor,
        kpts_enc: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Each input: (B, T, D)
        Returns: (B, T, D) fused representation
        """
        if self.strategy == "cross_attention":
            fused = self._cross_attention_fusion(rgb_enc, hands_enc, kpts_enc, mask)
        elif self.strategy == "concat":
            fused = self.proj(torch.cat([rgb_enc, hands_enc, kpts_enc], dim=-1))
        elif self.strategy == "gated":
            fused = self._gated_fusion(rgb_enc, hands_enc, kpts_enc)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

        # Refine with self-attention
        for layer in self.refine:
            fused = layer(fused, mask)

        return fused

    def _cross_attention_fusion(
        self, rgb: torch.Tensor, hands: torch.Tensor, kpts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.cross_layers:
            # Each stream attends to the other two (concatenated as context)
            ctx_for_rgb = torch.cat([hands, kpts], dim=1)
            ctx_for_hands = torch.cat([rgb, kpts], dim=1)
            ctx_for_kpts = torch.cat([rgb, hands], dim=1)

            # Build context masks
            ctx_mask = None
            if mask is not None:
                ctx_mask = torch.cat([mask, mask], dim=1)  # doubled since 2 other streams

            rgb = layer["rgb_cross"](rgb, ctx_for_rgb, ctx_mask)
            hands = layer["hands_cross"](hands, ctx_for_hands, ctx_mask)
            kpts = layer["kpts_cross"](kpts, ctx_for_kpts, ctx_mask)

        # Aggregate: concat + project
        return self.agg_proj(torch.cat([rgb, hands, kpts], dim=-1))

    def _gated_fusion(
        self, rgb: torch.Tensor, hands: torch.Tensor, kpts: torch.Tensor,
    ) -> torch.Tensor:
        g_r = self.gate_rgb(rgb)
        g_h = self.gate_hands(hands)
        g_k = self.gate_kpts(kpts)
        # Normalize gates
        g_sum = g_r + g_h + g_k + 1e-8
        fused = (g_r / g_sum) * rgb + (g_h / g_sum) * hands + (g_k / g_sum) * kpts
        return self.out_proj(fused)