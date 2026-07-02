# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


class ECA(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x.transpose(1, 2))
        y = self.conv(y.transpose(1, 2))
        y = self.sigmoid(y)
        return x * y


class MaskedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation, padding='same',
                              groups=groups, bias=bias)
    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        if mask is not None and self.stride > 1:
            mask = mask[:, ::self.stride]
        return x, mask


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size, expand_ratio=2, dropout=0.0, activation='tanh'):
        super().__init__()
        hidden_dim = dim * expand_ratio
        self.expand = nn.Linear(dim, hidden_dim)
        self.conv = MaskedConv1D(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim, momentum=0.95)
        self.eca = ECA(hidden_dim)
        self.project = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)
    def forward(self, x, mask=None):
        identity = x
        x = self.expand(x)
        x, mask = self.conv(x, mask)
        x = x.transpose(1, 2); x = self.bn(x); x = x.transpose(1, 2)
        x = self.eca(x)
        x = self.project(x)
        x = self.dropout(x)
        if identity.shape[-1] == x.shape[-1]:
            x = x + identity
        return x, mask


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, expand=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*expand, bias=False), nn.GELU(),
                                  nn.Linear(dim*expand, dim, bias=False), nn.Dropout(dropout))
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x, mask


class ConformerBlock(nn.Module):
    """Paired Conv + Transformer block (interleaved architecture).

    Interleaves conv and self-attention at every depth level rather than
    running all conv blocks first, then all transformer blocks.
    This block is the unit that gets stacked N times to build that encoder.
    """
    def __init__(self, dim, conv_kernel, num_heads=8, ffn_expand=4, dropout=0.0):
        super().__init__()
        self.conv = ConvBlock(dim, conv_kernel, dropout=dropout)
        self.transformer = TransformerBlock(dim, num_heads, ffn_expand, dropout)

    def forward(self, x, mask=None):
        x, mask = self.conv(x, mask)
        x, mask = self.transformer(x, mask)
        return x, mask


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ─── Contrastive Loss ────────────────────────────────────────────────

class SignContrastiveLoss(nn.Module):
    """NT-Xent (InfoNCE) contrastive loss on pooled encoder representations.

    Two independently augmented views of the same batch are treated as
    positive pairs; all other within-batch pairs are negatives.  A 2-layer
    MLP projection head (following SimCLR best practice) maps pooled encoder
    hiddens to the contrastive space before computing the loss.

    Usage in training:
        aug1 = augment_keypoints(keypoints, mask)
        aug2 = augment_keypoints(keypoints, mask)
        h1, m1 = model.encode(aug1, mask.clone())
        h2, m2 = model.encode(aug2, mask.clone())
        loss += contrastive_weight * contrastive_head(h1, m1, h2, m2)

    References:
        SimCLR    – Chen et al., ICML 2020
        SignCL    – Min et al., ACL 2024
        MSKA      – Wang et al., arXiv 2405.05672
    """
    def __init__(self, encoder_dim: int, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, proj_dim),
        )
        self.temperature = temperature

    def _pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean-pool encoder hiddens → (B, encoder_dim)."""
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        h1: torch.Tensor, m1: torch.Tensor,
        h2: torch.Tensor, m2: torch.Tensor,
    ) -> torch.Tensor:
        z1 = F.normalize(self.proj(self._pool(h1, m1)), dim=-1)   # (B, proj_dim)
        z2 = F.normalize(self.proj(self._pool(h2, m2)), dim=-1)
        B  = z1.size(0)
        z  = torch.cat([z1, z2], dim=0)                            # (2B, proj_dim)
        sim = (z @ z.T) / self.temperature                         # (2B, 2B)
        sim.fill_diagonal_(float('-inf'))                          # mask self-similarity
        # positive for z1[i] is z2[i] = row i+B, and vice-versa
        labels = torch.cat([
            torch.arange(B, 2 * B, device=z.device),
            torch.arange(B,        device=z.device),
        ])
        return F.cross_entropy(sim, labels)


# ─── BART Translation Head ──────────────────────────────────────────

class BARTTranslationHead(nn.Module):
    """Seq2Seq translation head (BART, mBART, Marian, etc.)."""
    def __init__(self, encoder_dim, bart_model="facebook/bart-base", max_len=128,
                 label_smoothing=0.1, source_len_cap=96,
                 no_repeat_ngram_size=3, repetition_penalty=1.15):
        super().__init__()
        from transformers import AutoModelForSeq2SeqLM
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(bart_model)
        self.bart.gradient_checkpointing_enable()
        bart_dim = self.bart.config.d_model
        # Freeze BART's internal encoder — we always pass encoder_outputs directly
        # so it is never called. Freezing it saves ~50% of BART optimizer state.
        for p in self.bart.model.encoder.parameters():
            p.requires_grad = False
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_dim, bart_dim),
            nn.LayerNorm(bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, bart_dim),
            nn.LayerNorm(bart_dim),
        )
        self.max_len = max_len
        self._frozen = False
        self.label_smoothing = label_smoothing
        self.source_len_cap = source_len_cap
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.repetition_penalty = repetition_penalty

    def _compress(self, encoder_hidden, encoder_mask=None):
        if encoder_mask is None:
            encoder_mask = encoder_hidden.new_ones(
                encoder_hidden.shape[:2], dtype=encoder_hidden.dtype
            )

        hidden = encoder_hidden
        mask = encoder_mask.float()

        while hidden.size(1) > self.source_len_cap:
            if hidden.size(1) % 2 == 1:
                hidden = F.pad(hidden, (0, 0, 0, 1))
                mask = F.pad(mask, (0, 1))

            left_h, right_h = hidden[:, 0::2], hidden[:, 1::2]
            left_m, right_m = mask[:, 0::2], mask[:, 1::2]
            pair_mask = (left_m + right_m).clamp(min=1.0)
            hidden = (left_h * left_m.unsqueeze(-1) + right_h * right_m.unsqueeze(-1)) / pair_mask.unsqueeze(-1)
            mask = (left_m + right_m > 0).float()

        return hidden, mask

    def _project(self, encoder_hidden, encoder_mask=None):
        """Project encoder hidden states and wrap for HuggingFace."""
        from transformers.modeling_outputs import BaseModelOutput
        encoder_hidden, encoder_mask = self._compress(encoder_hidden, encoder_mask)
        if self._frozen:
            projected = self.encoder_proj(encoder_hidden.detach())
        else:
            projected = self.encoder_proj(encoder_hidden)
        return BaseModelOutput(last_hidden_state=projected), encoder_mask

    def forward(self, encoder_hidden, encoder_mask=None, labels=None):
        encoder_out, encoder_mask = self._project(encoder_hidden, encoder_mask)
        outputs = self.bart(encoder_outputs=encoder_out,
                            attention_mask=encoder_mask, labels=labels)
        loss = outputs.loss
        # Apply label smoothing on top of BART's cross-entropy
        if loss is not None and self.label_smoothing > 0 and labels is not None:
            vocab_size = outputs.logits.size(-1)
            log_probs  = F.log_softmax(outputs.logits, dim=-1)
            pad_mask   = labels != -100
            smooth_loss = -log_probs.sum(dim=-1) / vocab_size
            loss = (1 - self.label_smoothing) * loss + \
                   self.label_smoothing * smooth_loss[pad_mask].mean()
        return {"loss": loss, "logits": outputs.logits}

    def generate(self, encoder_hidden, encoder_mask=None,
                 beam_width=5, max_len=None, length_penalty=1.0,
                 forced_bos_token_id=None):
        encoder_out, encoder_mask = self._project(encoder_hidden, encoder_mask)
        kwargs = dict(
            encoder_outputs=encoder_out, attention_mask=encoder_mask,
            max_length=max_len or self.max_len, num_beams=beam_width,
            length_penalty=length_penalty, early_stopping=True,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
        )
        if forced_bos_token_id is not None:
            kwargs['forced_bos_token_id'] = forced_bos_token_id
        return self.bart.generate(**kwargs)

    def freeze(self):
        """Freeze BART AND encoder_proj — no BART gradients touch encoder."""
        self._frozen = True
        for p in self.bart.parameters(): p.requires_grad = False
        for p in self.encoder_proj.parameters(): p.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze BART decoder + encoder_proj for joint fine-tuning.
        BART's internal encoder stays frozen — it is never called since we
        always pass encoder_outputs directly."""
        self._frozen = False
        for p in self.bart.model.decoder.parameters(): p.requires_grad = True
        for p in self.bart.lm_head.parameters():       p.requires_grad = True
        for p in self.encoder_proj.parameters():       p.requires_grad = True


# ─── Main Model ──────────────────────────────────────────────────────

class SignLanguageTransformer(nn.Module):
    """
    Sign language recognition + translation model.
    
    Modes:
      --use_bart=False: keypoints → encoder → CTC gloss head (Sign2Gloss)
      --use_bart=True:  keypoints → encoder → CTC glosses + BART translation (Sign2Gloss2Text)
    """
    def __init__(self, input_dim=225, dim=192, num_classes=1085, max_frames=500,
                 dropout=0.2, use_bart=False, bart_model="facebook/bart-base",
                 ctc_weight=0.3, ffn_expand=4, label_smoothing=0.1,
                 use_motion=False, arch='sequential'):
        super().__init__()
        self.use_bart   = use_bart
        self.ctc_weight = ctc_weight
        self.dim        = dim
        self.use_motion = use_motion
        self.arch       = arch

        proj_in = input_dim * 3 if use_motion else input_dim
        self.input_proj = nn.Linear(proj_in, dim, bias=False)
        self.pos_encoding = PositionalEncoding(dim, max_frames)
        self.norm = nn.LayerNorm(dim)

        if arch == 'interleaved':
            # 6 paired (Conv + Transformer) blocks, alternating kernels 11 and 5.
            self.conv_blocks = nn.ModuleList([
                ConformerBlock(dim, k, num_heads=8, ffn_expand=ffn_expand, dropout=dropout)
                for k in [11, 5, 11, 5, 11, 5]
            ])
            # No standalone transformer blocks — transformers are embedded in each ConformerBlock.
            # An empty ModuleList is kept so multistream subclass can safely rebuild it.
            self.transformer_blocks = nn.ModuleList()
        else:
            self.conv_blocks = nn.ModuleList([
                ConvBlock(dim, 11, dropout=dropout),
                ConvBlock(dim, 5, dropout=dropout),
                ConvBlock(dim, 3, dropout=dropout),
            ])
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(dim, num_heads=8, expand=ffn_expand, dropout=dropout)
                for _ in range(4)
            ])

        # CTC gloss head
        self.head = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(dim * 2, num_classes)
        )

        # BART translation head (optional)
        self.translation_head = None
        if use_bart:
            self.translation_head = BARTTranslationHead(
                encoder_dim=dim, bart_model=bart_model,
                label_smoothing=label_smoothing)

    def encode(self, x, mask=None):
        """Encode keypoints → hidden states (B, T, dim).

        When use_motion=True, velocity features (frame differences) are
        concatenated to the raw keypoints before the input projection,
        giving the model explicit access to motion information that is
        otherwise only implicitly available through temporal convolutions.
        This follows the temporal difference stream approach in TwoStream-SLR
        (Zuo et al., NeurIPS 2022).
        """
        if mask is None:
            mask = (x.abs().sum(dim=-1) != 0).float()
        if self.use_motion:
            vel = torch.zeros_like(x)
            vel[:, 1:] = x[:, 1:] - x[:, :-1]
            acc = torch.zeros_like(x)
            acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
            x = torch.cat([x, vel, acc], dim=-1)   # (B, T, input_dim*3)
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.pos_encoding(x)
        for blk in self.conv_blocks:
            x, mask = blk(x, mask)
        # interleaved: transformers already run inside each ConformerBlock above
        if self.arch == 'sequential':
            for blk in self.transformer_blocks:
                x, mask = blk(x, mask)
        return x, mask

    def forward(self, x, mask=None, translation_targets=None):
        hidden, mask = self.encode(x, mask)
        logits = self.head(hidden)

        if not self.use_bart:
            return logits, mask

        output = {'logits': logits, 'mask': mask, 'hidden': hidden}
        if translation_targets is not None and self.translation_head is not None:
            try:
                trans_out = self.translation_head(hidden, encoder_mask=mask,
                                                   labels=translation_targets)
                output['translation_loss'] = trans_out['loss']
            except Exception:
                pass  # Skip translation loss if it fails (e.g. bad targets)
        return output

    @torch.no_grad()
    def translate(self, x, mask=None, beam_width=5, length_penalty=1.0,
                  forced_bos_token_id=None):
        self.eval()
        hidden, mask = self.encode(x, mask)
        if self.translation_head is None:
            return None
        return self.translation_head.generate(hidden, mask,
                                               beam_width=beam_width,
                                               length_penalty=max(length_penalty, 1.1),
                                               forced_bos_token_id=forced_bos_token_id)

    def freeze_translation(self):
        if self.translation_head: self.translation_head.freeze()
    def unfreeze_translation(self):
        if self.translation_head: self.translation_head.unfreeze()

    # ── Transfer learning freeze controls ────────────────────────────

    def freeze_encoder(self):
        """Freeze entire encoder (input_proj, pos_encoding, conv_blocks, transformer_blocks).
        Only the CTC head and optional BART head remain trainable.
        Use for Exp 7 (full freeze transfer)."""
        for p in self.input_proj.parameters():        p.requires_grad = False
        for p in self.norm.parameters():              p.requires_grad = False
        for blk in self.conv_blocks:
            for p in blk.parameters():                p.requires_grad = False
        for blk in self.transformer_blocks:
            for p in blk.parameters():                p.requires_grad = False

    def freeze_convblocks(self):
        """Freeze only the convolutional blocks (low-level temporal features).
        Transformer blocks remain trainable — recommended strategy from ULMFiT/BERT.
        Use for Exp 8 (ConvBlocks freeze transfer)."""
        for p in self.input_proj.parameters():        p.requires_grad = False
        for p in self.norm.parameters():              p.requires_grad = False
        for blk in self.conv_blocks:
            for p in blk.parameters():                p.requires_grad = False

    def unfreeze_encoder(self):
        """Restore full gradient flow across the entire model."""
        for p in self.parameters():                   p.requires_grad = True

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def frozen_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


class LanguageAwareSignTransformer(SignLanguageTransformer):
    """
    Extension of SignLanguageTransformer for joint PHOENIX + How2Sign training.

    Adds a learned language-identity embedding (lang_id ∈ {0, 1}) that is
    summed into the input after positional encoding.  Separate CTC heads are
    used per language so that the shared encoder must produce representations
    that are useful for BOTH gloss vocabularies simultaneously.

    Architecture:
        x (225) → input_proj → norm → pos_encoding
               → + lang_embedding(lang_id)          ← NEW
               → conv_blocks → transformer_blocks
               → head_phoenix  (if lang_id == 0)
               → head_how2sign (if lang_id == 1)
    """

    def __init__(self, input_dim=225, dim=256,
                 num_classes_phoenix=1085, num_classes_how2sign=2048,
                 max_frames=500, dropout=0.3,
                 use_bart=False, bart_model='facebook/bart-base', ctc_weight=0.3,
                 use_motion=False):
        # Build the shared encoder (num_classes is a dummy; we replace head below)
        super().__init__(
            input_dim=input_dim, dim=dim,
            num_classes=num_classes_phoenix,   # head_phoenix
            max_frames=max_frames, dropout=dropout,
            use_bart=use_bart, bart_model=bart_model, ctc_weight=ctc_weight,
            use_motion=use_motion,
        )
        # Language embedding: 0=PHOENIX(DGS), 1=How2Sign(ASL)
        self.lang_embedding = nn.Embedding(2, dim)
        nn.init.normal_(self.lang_embedding.weight, std=0.02)

        # Second CTC head for How2Sign
        self.head_how2sign = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(dim * 2, num_classes_how2sign),
        )
        # Rename base head for clarity
        self.head_phoenix = self.head

    def encode(self, x, mask=None, lang_id=None):
        """Encode with optional language embedding injection."""
        if mask is None:
            mask = (x.abs().sum(dim=-1) != 0).float()
        if self.use_motion:
            vel = torch.zeros_like(x)
            vel[:, 1:] = x[:, 1:] - x[:, :-1]
            acc = torch.zeros_like(x)
            acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
            x = torch.cat([x, vel, acc], dim=-1)
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.pos_encoding(x)

        if lang_id is not None:
            # lang_id: (B,) int tensor
            lang_emb = self.lang_embedding(lang_id)          # (B, dim)
            x = x + lang_emb.unsqueeze(1)                    # broadcast over T

        for blk in self.conv_blocks:
            x, mask = blk(x, mask)
        for blk in self.transformer_blocks:
            x, mask = blk(x, mask)
        return x, mask

    def forward(self, x, mask=None, lang_id=None, translation_targets=None):
        hidden, mask = self.encode(x, mask, lang_id=lang_id)

        # Select CTC head based on language
        if lang_id is not None and (lang_id == 1).all():
            logits = self.head_how2sign(hidden)
        else:
            logits = self.head_phoenix(hidden)

        if not self.use_bart:
            return logits, mask

        output = {'logits': logits, 'mask': mask, 'hidden': hidden}
        if translation_targets is not None and self.translation_head is not None:
            try:
                trans_out = self.translation_head(hidden, encoder_mask=mask,
                                                   labels=translation_targets)
                output['translation_loss'] = trans_out['loss']
            except Exception:
                pass
        return output


# ─── Multi-Stream Encoder ─────────────────────────────────────────────

class MultiStreamSignLanguageTransformer(SignLanguageTransformer):
    """Multi-stream encoder: each body-part group is processed through its own
    projection + ConvBlock stack before cross-stream attention fusion.

    Temporal difference (velocity) features are always computed per-stream,
    giving each stream explicit access to its own motion information.

    Body-part feature slices (raw 225-d keypoints):
        Pose:        x[...,   0: 39]   (13 joints × 3)
        Left hand:   x[...,  39:102]   (21 joints × 3)
        Right hand:  x[..., 102:165]   (21 joints × 3)
        Face:        x[..., 165:225]   (20 joints × 3)

    Per-stream input after velocity concat: pose=78, hand=126, face=120.
    Each stream projects independently to dim; L and R hands share weights
    since they encode structurally symmetric motion patterns.

    Fusion: per-frame cross-stream multi-head attention (4 streams as tokens)
    followed by mean-pool → (B, T, dim) → shared TransformerBlocks.

    Architecture (MSKA / TwoStream-SLR inspired):
        stream_i → proj_i(dim) → LN → ConvBlocks(×3)  ─┐
                                                         ├→ MHA fusion → mean
        stream_j → proj_j(dim) → LN → ConvBlocks(×3)  ─┘
            ↓
        pos_encoding → TransformerBlocks(×4) → CTC / BART heads

    References:
        MSKA          – Wang et al., arXiv 2405.05672
        TwoStream-SLR – Zuo et al., NeurIPS 2022
    """

    _POSE_S  = slice(0,   39)
    _LHAND_S = slice(39,  102)
    _RHAND_S = slice(102, 165)
    _FACE_S  = slice(165, 225)

    def __init__(self, input_dim: int = 225, dim: int = 192,
                 num_classes: int = 1085, max_frames: int = 500,
                 dropout: float = 0.2, use_bart: bool = False,
                 bart_model: str = 'facebook/bart-base',
                 ctc_weight: float = 0.3, ffn_expand: int = 4,
                 label_smoothing: float = 0.1,
                 use_motion: bool = True,
                 arch: str = 'sequential'):
        # Bootstrap shared components (pos_encoding, transformer_blocks, head,
        # translation_head) via parent with a dummy input_dim=1.
        super().__init__(
            input_dim=1, dim=dim, num_classes=num_classes,
            max_frames=max_frames, dropout=dropout, use_bart=use_bart,
            bart_model=bart_model, ctc_weight=ctc_weight,
            ffn_expand=ffn_expand, label_smoothing=label_smoothing,
            use_motion=False,   # velocity handled per-stream in encode()
            arch=arch,
        )
        self.use_motion = use_motion
        # Remove flat-encoder components (replaced by per-stream equivalents)
        del self.input_proj, self.norm, self.conv_blocks

        # For interleaved arch the parent left transformer_blocks empty (transformers
        # are embedded per-stream in ConformerBlocks).  Rebuild them here for the
        # shared global-context stage that runs after cross-stream fusion.
        if arch == 'interleaved':
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(dim, num_heads=8, expand=ffn_expand, dropout=dropout)
                for _ in range(4)
            ])

        # Per-stream input dims: ×3 when use_motion (raw + vel + acc), ×1 otherwise
        _m = 3 if use_motion else 1
        pose_d = 39 * _m
        hand_d = 63 * _m
        face_d = 60 * _m

        # Stream projections + normalisation
        self.proj_pose  = nn.Linear(pose_d, dim, bias=False)
        self.proj_lhand = nn.Linear(hand_d, dim, bias=False)
        self.proj_rhand = nn.Linear(hand_d, dim, bias=False)
        self.proj_face  = nn.Linear(face_d, dim, bias=False)
        self.norm_pose  = nn.LayerNorm(dim)
        self.norm_lhand = nn.LayerNorm(dim)
        self.norm_rhand = nn.LayerNorm(dim)
        self.norm_face  = nn.LayerNorm(dim)

        # Per-stream block stacks.
        # interleaved: 3 ConformerBlocks (Conv+Transformer) per stream — each stream
        #   gets interleaved local context before cross-stream fusion.
        # sequential: 3 ConvBlocks per stream (current default).
        def _conv_stack():
            if arch == 'interleaved':
                return nn.ModuleList([
                    ConformerBlock(dim, k, num_heads=8, ffn_expand=ffn_expand, dropout=dropout)
                    for k in [11, 5, 11]
                ])
            return nn.ModuleList([
                ConvBlock(dim, 11, dropout=dropout),
                ConvBlock(dim, 5,  dropout=dropout),
                ConvBlock(dim, 3,  dropout=dropout),
            ])
        self.conv_pose  = _conv_stack()
        self.conv_lhand = _conv_stack()
        self.conv_rhand = _conv_stack()
        self.conv_face  = _conv_stack()

        # Cross-stream attention fusion: 4 stream tokens per frame → 1
        self.fusion_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=4, dropout=dropout, batch_first=True)
        self.fusion_norm = nn.LayerNorm(dim)

    # ── Internal helpers ─────────────────────────────────────────────

    @staticmethod
    def _add_velocity(feat: torch.Tensor) -> torch.Tensor:
        """Concat velocity + acceleration features: (B,T,D) → (B,T,3D)."""
        vel = torch.zeros_like(feat)
        vel[:, 1:] = feat[:, 1:] - feat[:, :-1]
        acc = torch.zeros_like(feat)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]
        return torch.cat([feat, vel, acc], dim=-1)

    def _encode_stream(self, raw, proj, norm, conv_blocks, mask):
        x = norm(proj(self._add_velocity(raw) if self.use_motion else raw))
        for blk in conv_blocks:
            x, mask = blk(x, mask)
        return x, mask

    # ── Core encode (overrides flat version) ─────────────────────────

    def encode(self, x: torch.Tensor,
               mask: Optional[torch.Tensor] = None) -> tuple:
        if mask is None:
            mask = (x.abs().sum(dim=-1) != 0).float()

        # Encode each body-part stream independently
        s_pose,  m  = self._encode_stream(
            x[..., self._POSE_S],  self.proj_pose,  self.norm_pose,  self.conv_pose,  mask.clone())
        s_lhand, _  = self._encode_stream(
            x[..., self._LHAND_S], self.proj_lhand, self.norm_lhand, self.conv_lhand, mask.clone())
        s_rhand, _  = self._encode_stream(
            x[..., self._RHAND_S], self.proj_rhand, self.norm_rhand, self.conv_rhand, mask.clone())
        s_face,  _  = self._encode_stream(
            x[..., self._FACE_S],  self.proj_face,  self.norm_face,  self.conv_face,  mask.clone())

        # Cross-stream attention: treat 4 streams as a 4-token sequence per frame
        B, T, D = s_pose.shape
        flat = torch.stack([s_pose, s_lhand, s_rhand, s_face], dim=2)  # (B, T, 4, D)
        flat = flat.reshape(B * T, 4, D)
        attended, _ = self.fusion_attn(flat, flat, flat)
        fused = self.fusion_norm(flat + attended).mean(dim=1)           # (B*T, D)
        x = fused.reshape(B, T, D)

        # Shared positional encoding + transformer blocks
        x = self.pos_encoding(x)
        for blk in self.transformer_blocks:
            x, m = blk(x, m)
        return x, m

    # ── Freeze controls for multi-stream layout ───────────────────────

    def freeze_encoder(self):
        for mod in [
            self.proj_pose, self.proj_lhand, self.proj_rhand, self.proj_face,
            self.norm_pose, self.norm_lhand, self.norm_rhand, self.norm_face,
            self.conv_pose, self.conv_lhand, self.conv_rhand, self.conv_face,
            self.fusion_attn, self.fusion_norm, self.pos_encoding,
        ] + list(self.transformer_blocks):
            for p in mod.parameters():
                p.requires_grad = False

    def freeze_convblocks(self):
        """Freeze per-stream projections, norms, convs, and fusion.
        Transformer blocks remain trainable (ULMFiT-style)."""
        for mod in [
            self.proj_pose, self.proj_lhand, self.proj_rhand, self.proj_face,
            self.norm_pose, self.norm_lhand, self.norm_rhand, self.norm_face,
            self.conv_pose, self.conv_lhand, self.conv_rhand, self.conv_face,
            self.fusion_attn, self.fusion_norm,
        ]:
            for p in mod.parameters():
                p.requires_grad = False
