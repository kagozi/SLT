"""
MultiStreamSLT: Full multistream Sign Language Translation model.

Architecture:
  RGB frames   ──→ RGBStreamEncoder   ──┐
  Hand crops   ──→ HandsStreamEncoder  ──┼──→ MultiStreamFusion ──→ CTC Gloss Head
  Keypoints    ──→ KptsStreamEncoder   ──┘                     └──→ BART Translation Head

Training modes:
  - Stage 1: CTC-only gloss recognition (freeze BART)
  - Stage 2: Joint CTC + translation (unfreeze BART)
  - Stage 3: Fine-tune end-to-end with reduced CTC weight
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from transformers import BartForConditionalGeneration, BartConfig

from encoding.stream_encoders import RGBStreamEncoder, HandsStreamEncoder, KeypointsStreamEncoder
from encoding.fusion import MultiStreamFusion


class CTCGlossHead(nn.Module):
    """CTC-based gloss recognition head."""

    def __init__(self, hidden_dim: int, vocab_size: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, vocab_size) log-probs for CTC."""
        return F.log_softmax(self.head(x), dim=-1)


class BARTTranslationHead(nn.Module):
    """
    BART-based translation head.
    Takes fused encoder output and generates target text autoregressively.
    """

    def __init__(
        self,
        hidden_dim: int,
        bart_model_name: str = "facebook/bart-base",
        max_len: int = 100,
    ):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        bart_dim = self.bart.config.d_model

        # Project from our hidden_dim to BART's d_model
        self.encoder_proj = nn.Linear(hidden_dim, bart_dim)
        self.norm = nn.LayerNorm(bart_dim)
        self.max_len = max_len

    def forward(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        encoder_hidden: (B, T, hidden_dim) — fused multistream features
        encoder_mask: (B, T) attention mask
        labels: (B, S) target token ids for teacher forcing

        Returns dict with 'loss' and 'logits'.
        """
        # Project to BART dim
        projected = self.norm(self.encoder_proj(encoder_hidden))

        # Use BART decoder with our custom encoder output
        outputs = self.bart(
            encoder_outputs=(projected,),
            attention_mask=encoder_mask,
            labels=labels,
        )

        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(
        self,
        encoder_hidden: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        beam_width: int = 5,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate translations via beam search."""
        projected = self.norm(self.encoder_proj(encoder_hidden))
        max_len = max_len or self.max_len

        return self.bart.generate(
            encoder_outputs=(projected,),
            attention_mask=encoder_mask,
            max_length=max_len,
            num_beams=beam_width,
            early_stopping=True,
        )

    def freeze(self):
        """Freeze BART parameters for stage 1 training."""
        for param in self.bart.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze BART parameters for fine-tuning."""
        for param in self.bart.parameters():
            param.requires_grad = True


class MultiStreamSLT(nn.Module):
    """
    Complete multi-stream Sign Language Translation model.

    Inputs:
      rgb:  (B, T, 3, H, W)     — full-frame video
      hands: (B, T, 2, 3, Hh, Wh) — left/right hand crops
      kpts: (B, T, D_kpts)      — holistic keypoints

    Outputs:
      gloss_logits: (B, T, gloss_vocab) — CTC output
      translation: BART output (loss + logits during training, token ids during inference)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_frames: int = 64,
        kpts_dim: int = 498,
        gloss_vocab_size: int = 1085,
        # Encoder configs
        num_encoder_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        # Fusion
        fusion_strategy: str = "cross_attention",
        fusion_layers: int = 2,
        fusion_heads: int = 8,
        # Translation
        bart_model: str = "facebook/bart-base",
        max_translation_len: int = 100,
        # CTC
        ctc_weight: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ctc_weight = ctc_weight

        # ── Stream Encoders ──
        self.rgb_encoder = RGBStreamEncoder(
            hidden_dim=hidden_dim, num_layers=num_encoder_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.hands_encoder = HandsStreamEncoder(
            hidden_dim=hidden_dim, num_layers=num_encoder_layers,
            num_heads=num_heads, dropout=dropout,
        )
        self.kpts_encoder = KeypointsStreamEncoder(
            input_dim=kpts_dim, hidden_dim=hidden_dim,
            num_layers=num_encoder_layers, num_heads=num_heads, dropout=dropout,
        )

        # ── Fusion ──
        self.fusion = MultiStreamFusion(
            hidden_dim=hidden_dim, num_heads=fusion_heads,
            num_layers=fusion_layers, dropout=dropout / 2,
            strategy=fusion_strategy,
        )

        # ── Heads ──
        self.ctc_head = CTCGlossHead(hidden_dim, gloss_vocab_size, dropout)
        self.translation_head = BARTTranslationHead(
            hidden_dim, bart_model, max_translation_len,
        )

        # CTC loss
        self.ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    def encode(
        self,
        rgb: torch.Tensor,
        hands: torch.Tensor,
        kpts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run all three stream encoders + fusion. Returns (B, T, D)."""
        rgb_enc = self.rgb_encoder(rgb, mask)
        hands_enc = self.hands_encoder(hands, mask)
        kpts_enc = self.kpts_encoder(kpts, mask)
        return self.fusion(rgb_enc, hands_enc, kpts_enc, mask)

    def forward(
        self,
        rgb: torch.Tensor,
        hands: torch.Tensor,
        kpts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gloss_targets: Optional[torch.Tensor] = None,
        gloss_lengths: Optional[torch.Tensor] = None,
        translation_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.

        Args:
            rgb: (B, T, 3, H, W)
            hands: (B, T, 2, 3, Hh, Wh)
            kpts: (B, T, D)
            mask: (B, T) bool mask for valid frames
            gloss_targets: (B, S_gloss) gloss token ids for CTC
            gloss_lengths: (B,) lengths of gloss targets
            translation_targets: (B, S_trans) translation token ids for BART

        Returns:
            Dict with 'loss', 'ctc_loss', 'translation_loss', 'gloss_logits', etc.
        """
        # Encode all streams
        fused = self.encode(rgb, hands, kpts, mask)  # (B, T, D)

        output = {"fused_features": fused}
        total_loss = torch.tensor(0.0, device=fused.device)

        # ── CTC Gloss Loss ──
        if gloss_targets is not None:
            gloss_logits = self.ctc_head(fused)  # (B, T, V)
            output["gloss_logits"] = gloss_logits

            B, T, _ = gloss_logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=fused.device)
            if mask is not None:
                input_lengths = mask.sum(dim=1).long()

            # CTC expects (T, B, V)
            ctc_loss = self.ctc_loss_fn(
                gloss_logits.permute(1, 0, 2),
                gloss_targets,
                input_lengths,
                gloss_lengths,
            )
            output["ctc_loss"] = ctc_loss
            total_loss = total_loss + self.ctc_weight * ctc_loss

        # ── Translation Loss ──
        if translation_targets is not None:
            trans_out = self.translation_head(
                fused, encoder_mask=mask, labels=translation_targets,
            )
            output["translation_loss"] = trans_out["loss"]
            output["translation_logits"] = trans_out["logits"]
            total_loss = total_loss + (1.0 - self.ctc_weight) * trans_out["loss"]

        output["loss"] = total_loss
        return output

    @torch.no_grad()
    def translate(
        self,
        rgb: torch.Tensor,
        hands: torch.Tensor,
        kpts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        beam_width: int = 5,
    ) -> torch.Tensor:
        """Inference: encode → BART beam search → token ids."""
        self.eval()
        fused = self.encode(rgb, hands, kpts, mask)
        return self.translation_head.generate(fused, mask, beam_width)

    @torch.no_grad()
    def recognize_glosses(
        self,
        rgb: torch.Tensor,
        hands: torch.Tensor,
        kpts: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference: encode → CTC decode → gloss token ids."""
        self.eval()
        fused = self.encode(rgb, hands, kpts, mask)
        logits = self.ctc_head(fused)  # (B, T, V)

        # Greedy CTC decode: argmax → collapse repeats → remove blanks
        preds = logits.argmax(dim=-1)  # (B, T)
        return preds

    def freeze_translation(self):
        """Freeze BART for Stage 1 training."""
        self.translation_head.freeze()

    def unfreeze_translation(self):
        """Unfreeze BART for Stage 2+ training."""
        self.translation_head.unfreeze()