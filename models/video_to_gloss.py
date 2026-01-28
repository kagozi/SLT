"""
Complete video-to-gloss (ORTH) model.

Encoder: RGB video → [B,T,D]
Decoder: Transformer Decoder → token logits

Stage 1 target is `orth` (gloss-like sequence).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .video_encoders.rgb_encoder import RGBVideoEncoder, PositionalEncoding 


def generate_square_subsequent_mask(sz: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Standard causal mask for Transformer decoder: [sz, sz]
    mask[i,j] = -inf if j > i else 0
    """
    mask = torch.full((sz, sz), float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=1)
    return mask


class TransformerDecoder(nn.Module):
    """Transformer decoder for ORTH generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_id = pad_id

        self.tgt_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tgt_emb.weight  # tie weights

        nn.init.normal_(self.tgt_emb.weight, mean=0.0, std=d_model ** -0.5)

    def forward(
        self,
        tgt: torch.Tensor,                 # [B, L]
        memory: torch.Tensor,              # [B, T, D]
        tgt_key_padding_mask: torch.Tensor | None = None,      # [B, L]
        memory_key_padding_mask: torch.Tensor | None = None,   # [B, T]
    ) -> torch.Tensor:
        # Embed
        x = self.tgt_emb(tgt)  # [B,L,D]
        x = self.pos_enc(x)

        L = tgt.size(1)
        causal = generate_square_subsequent_mask(L, device=tgt.device, dtype=x.dtype)  # [L,L]

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.lm_head(out)
        return logits


class VideoToGlossModel(nn.Module):
    """RGB-only video-to-ORTH model."""

    def __init__(
        self,
        gloss_vocab_size: int,
        pad_id: int = 0,
        d_model: int = 512,
        # Encoder
        encoder_backbone: str = "resnet18",
        encoder_pretrained: bool = True,
        encoder_freeze: bool = False,
        encoder_temporal_layers: int = 4,
        encoder_nhead: int = 8,
        # Decoder
        decoder_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.encoder = RGBVideoEncoder(
            d_model=d_model,
            spatial_backbone=encoder_backbone,
            spatial_pretrained=encoder_pretrained,
            spatial_freeze=encoder_freeze,
            temporal_nhead=encoder_nhead,
            temporal_layers=encoder_temporal_layers,
            temporal_dim_feedforward=d_model * 4,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            vocab_size=gloss_vocab_size,
            d_model=d_model,
            nhead=decoder_nhead,
            num_layers=decoder_layers,
            dim_feedforward=decoder_dim_feedforward,
            dropout=dropout,
            pad_id=pad_id,
        )

    def forward(
        self,
        rgb: torch.Tensor,  # [B,T,3,H,W]
        tgt: torch.Tensor,  # [B,L]
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        memory = self.encoder(rgb, src_key_padding_mask=src_key_padding_mask)
        logits = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        rgb: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        bos_id: int = 2,
        eos_id: int = 3,
        max_len: int = 100,
    ) -> torch.Tensor:
        self.eval()
        B = rgb.size(0)
        memory = self.encoder(rgb, src_key_padding_mask=src_key_padding_mask)

        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=rgb.device)
        finished = torch.zeros(B, dtype=torch.bool, device=rgb.device)

        for _ in range(max_len - 1):
            logits = self.decoder(
                tgt=generated,
                memory=memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_key_padding_mask,
            )
            next_tok = logits[:, -1].argmax(dim=-1)
            generated = torch.cat([generated, next_tok[:, None]], dim=1)

            finished |= (next_tok == eos_id)
            if finished.all():
                break

        return generated
