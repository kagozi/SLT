"""
Multistream SLT Configuration
All hyperparameters for preprocessing, model architecture, and training.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional


# ─────────────────────────────────────────────
# Preprocessing Configs
# ─────────────────────────────────────────────

@dataclass
class RGBConfig:
    target_size: Tuple[int, int] = (224, 224)
    num_frames: int = 64
    sampling_strategy: str = "uniform"  # "uniform" | "adaptive"
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)
    std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    normalize: bool = True
    save_fp16: bool = True


@dataclass
class HandsConfig:
    crop_size: Tuple[int, int] = (112, 112)
    min_hand_conf: float = 0.4
    padding_factor: float = 1.5
    interpolate_missing: bool = True
    max_interpolation_frames: int = 10
    smoothing_window: int = 5
    save_fp16: bool = True


@dataclass
class KeypointsConfig:
    use_pose: bool = True
    use_hands: bool = True
    use_face: bool = True
    static_image_mode: bool = False
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    temporal_smoothing: bool = True
    smoothing_window: int = 3
    # Dimensions: pose=33*4=132, hands=2*21*3=126, face=80*3=240 → total=498
    total_dim: int = 498


@dataclass
class PreprocessingConfig:
    dataset: str = "phoenix2014t"
    data_root: str = ""
    output_dir: str = "data_cache"
    rgb: RGBConfig = field(default_factory=RGBConfig)
    hands: HandsConfig = field(default_factory=HandsConfig)
    keypoints: KeypointsConfig = field(default_factory=KeypointsConfig)


# ─────────────────────────────────────────────
# Model Architecture Configs
# ─────────────────────────────────────────────

@dataclass
class StreamEncoderConfig:
    """Per-stream encoder config (shared structure, different input dims)."""
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    ff_expand: int = 2
    dropout: float = 0.2
    attn_dropout: float = 0.2
    conv_kernels: List[int] = field(default_factory=lambda: [11, 5, 3])
    use_eca: bool = True


@dataclass
class FusionConfig:
    strategy: str = "cross_attention"  # "cross_attention" | "concat" | "gated"
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class CTCHeadConfig:
    enabled: bool = True
    weight: float = 0.3  # CTC loss weight in joint CTC + CE training


@dataclass
class TranslationConfig:
    bart_model: str = "facebook/bart-base"
    max_gloss_len: int = 75
    max_translation_len: int = 100
    label_smoothing: float = 0.1
    beam_width: int = 5
    freeze_bart_epochs: int = 5  # freeze BART decoder for N epochs


@dataclass
class ModelConfig:
    # Input dimensions (derived from preprocessing)
    rgb_input_shape: Tuple[int, ...] = (64, 3, 224, 224)  # T, C, H, W
    hands_input_shape: Tuple[int, ...] = (64, 2, 3, 112, 112)  # T, 2, C, H, W
    kpts_input_shape: Tuple[int, ...] = (64, 498)  # T, D

    num_frames: int = 64

    # Per-stream encoders
    rgb_encoder: StreamEncoderConfig = field(default_factory=StreamEncoderConfig)
    hands_encoder: StreamEncoderConfig = field(default_factory=StreamEncoderConfig)
    kpts_encoder: StreamEncoderConfig = field(default_factory=StreamEncoderConfig)

    # Fusion
    fusion: FusionConfig = field(default_factory=FusionConfig)

    # Heads
    ctc: CTCHeadConfig = field(default_factory=CTCHeadConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)

    # Gloss vocabulary (for CTC)
    gloss_vocab_size: int = 1085


# ─────────────────────────────────────────────
# Training Configs
# ─────────────────────────────────────────────

@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 100
    warmup_epochs: int = 5
    lr_max: float = 1e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = True
    seed: int = 42
    save_dir: str = "checkpoints"
    log_every: int = 50
    eval_every_epoch: int = 1


@dataclass
class FullConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def get_default_config(data_root: str = "") -> FullConfig:
    cfg = FullConfig()
    cfg.preprocessing.data_root = data_root
    return cfg