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


# ─── BART Translation Head ──────────────────────────────────────────

class BARTTranslationHead(nn.Module):
    """BART-based Gloss→Text translation head."""
    def __init__(self, encoder_dim, bart_model="facebook/bart-base", max_len=128):
        super().__init__()
        from transformers import BartForConditionalGeneration
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model)
        bart_dim = self.bart.config.d_model
        self.encoder_proj = nn.Sequential(
            nn.Linear(encoder_dim, bart_dim),
            nn.LayerNorm(bart_dim),
            nn.GELU(),
            nn.Linear(bart_dim, bart_dim),
            nn.LayerNorm(bart_dim),
        )
        self.max_len = max_len

    def forward(self, encoder_hidden, encoder_mask=None, labels=None):
        projected = self.encoder_proj(encoder_hidden)
        outputs = self.bart(encoder_outputs=(projected,),
                            attention_mask=encoder_mask, labels=labels)
        return {"loss": outputs.loss, "logits": outputs.logits}

    def generate(self, encoder_hidden, encoder_mask=None,
                 beam_width=5, max_len=None, length_penalty=1.0):
        projected = self.encoder_proj(encoder_hidden)
        return self.bart.generate(
            encoder_outputs=(projected,), attention_mask=encoder_mask,
            max_length=max_len or self.max_len, num_beams=beam_width,
            length_penalty=length_penalty, early_stopping=True)

    def freeze(self):
        for p in self.bart.parameters(): p.requires_grad = False
    def unfreeze(self):
        for p in self.bart.parameters(): p.requires_grad = True


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
                 ctc_weight=0.3):
        super().__init__()
        self.use_bart = use_bart
        self.ctc_weight = ctc_weight
        self.dim = dim

        self.input_proj = nn.Linear(input_dim, dim, bias=False)
        self.pos_encoding = PositionalEncoding(dim, max_frames)
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)

        self.conv_blocks = nn.ModuleList([
            ConvBlock(dim, 11, dropout=dropout),
            ConvBlock(dim, 5, dropout=dropout),
            ConvBlock(dim, 3, dropout=dropout),
        ])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=4, expand=2, dropout=dropout)
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
                encoder_dim=dim, bart_model=bart_model)

    def encode(self, x, mask=None):
        """Encode keypoints → hidden states (B, T, dim)."""
        if mask is None:
            mask = (x.abs().sum(dim=-1) != 0).float()
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = x.transpose(1, 2); x = self.bn(x); x = x.transpose(1, 2)
        for blk in self.conv_blocks:
            x, mask = blk(x, mask)
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
            trans_out = self.translation_head(hidden, encoder_mask=mask,
                                               labels=translation_targets)
            output['translation_loss'] = trans_out['loss']
        return output

    @torch.no_grad()
    def translate(self, x, mask=None, beam_width=5, length_penalty=1.0):
        self.eval()
        hidden, mask = self.encode(x, mask)
        if self.translation_head is None:
            return None
        return self.translation_head.generate(hidden, mask,
                                               beam_width=beam_width,
                                               length_penalty=length_penalty)

    def freeze_translation(self):
        if self.translation_head: self.translation_head.freeze()
    def unfreeze_translation(self):
        if self.translation_head: self.translation_head.unfreeze()