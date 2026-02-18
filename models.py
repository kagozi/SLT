import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA(nn.Module):
    """Efficient Channel Attention module"""
    
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        y = self.avg_pool(x.transpose(1, 2))  # (batch, channels, 1)
        y = self.conv(y.transpose(1, 2))      # (batch, 1, channels)
        y = self.sigmoid(y)                    # (batch, 1, channels)
        return x * y


class MaskedConv1D(nn.Module):
    """1D convolution with masking support"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, padding='same',
            groups=groups, bias=bias
        )
        
    def forward(self, x, mask=None):
        # x: (batch, seq_len, channels)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            
        x = x.transpose(1, 2)  # (batch, channels, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (batch, seq_len, channels)
        
        if mask is not None and self.stride > 1:
            mask = mask[:, ::self.stride]
            
        return x, mask


class ConvBlock(nn.Module):
    """Efficient Conv1D block with expansion and ECA"""
    
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
        
        # Expand
        x = self.expand(x)
        
        # Depthwise conv
        x, mask = self.conv(x, mask)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        
        # Attention
        x = self.eca(x)
        
        # Project
        x = self.project(x)
        x = self.dropout(x)
        
        # Residual
        if identity.shape[-1] == x.shape[-1]:
            x = x + identity
            
        return x, mask


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module"""
    
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm"""
    
    def __init__(self, dim, num_heads=4, expand=4, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * expand, bias=False),
            nn.GELU(),
            nn.Linear(dim * expand, dim, bias=False),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), mask)
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x, mask


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
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


class SignLanguageTransformer(nn.Module):
    """Complete model for sign language recognition"""
    
    def __init__(self, input_dim=225, dim=192, num_classes=1085, max_frames=250, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, dim, bias=False)
        self.pos_encoding = PositionalEncoding(dim, max_len=max_frames * 2)
        self.bn = nn.BatchNorm1d(dim, momentum=0.95)
        
        # Convolutional blocks
        self.conv_blocks = nn.ModuleList([
            ConvBlock(dim, 11, dropout=dropout),
            ConvBlock(dim, 5, dropout=dropout),
            ConvBlock(dim, 3, dropout=dropout),
        ])
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads=4, expand=2, dropout=dropout)
            for _ in range(4)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(dim * 2, num_classes)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim) - keypoints
            mask: (batch, seq_len) - padding mask (1 for valid, 0 for padding)
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        B, T, _ = x.shape
        
        # Create mask if not provided
        if mask is None:
            mask = (x.abs().sum(dim=-1) != 0).float()
            
        # Project to model dimension
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        
        # Apply batch norm (transpose for BN)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x, mask = conv_block(x, mask)
            
        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x, mask = transformer_block(x, mask)
            
        # Output head
        logits = self.head(x)
        
        return logits, mask