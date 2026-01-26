# aslg_translation/preprocessing/config.py
"""
Configuration for preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, List


@dataclass
class RGBConfig:
    """Configuration for RGB stream extraction."""
    target_size: Tuple[int, int] = (256, 256)
    num_frames: int = 64  # target number of frames
    sampling_strategy: Literal["uniform", "adaptive"] = "uniform"

    # Person detection (not used for Phoenix v3 frames, but kept for other datasets)
    use_person_detection: bool = True
    detection_model: str = "yolov8n"
    min_detection_conf: float = 0.5
    tracking_enabled: bool = True

    # Preprocessing
    normalize: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Storage
    save_fp16: bool = False

    # Phoenix: frames already centered on signer
    phoenix_centered: bool = True


@dataclass
class KeypointsConfig:
    """Configuration for keypoint extraction."""
    # MediaPipe settings
    use_mediapipe: bool = True
    static_image_mode: bool = False
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    # Landmark selection
    use_pose: bool = True
    use_hands: bool = True
    use_face: bool = True

    # Face landmark reduction
    face_regions: Optional[List[List[int]]] = None

    # Smoothing
    temporal_smoothing: bool = True
    smoothing_window: int = 5

    # Storage
    include_confidence: bool = True
    normalize_coords: bool = True  # (mediapipe x/y already in [0,1])

    def __post_init__(self):
        if self.face_regions is None:
            self.face_regions = [
                list(range(0, 17)),
                list(range(61, 68)),
                list(range(78, 82)),
                list(range(133, 145)),
                list(range(362, 374)),
                list(range(70, 80)),
                list(range(300, 310)),
                list(range(1, 5)),
                list(range(195, 199)),
            ]


@dataclass
class HandCropConfig:
    """Configuration for hand crop extraction."""
    crop_size: Tuple[int, int] = (128, 128)
    padding_factor: float = 1.5

    # Tracking
    smoothing_window: int = 5
    max_hand_distance: float = 0.3  # normalized

    # Quality
    min_hand_conf: float = 0.5
    min_visible_landmarks: int = 10

    # Occlusion handling
    interpolate_missing: bool = True
    max_interpolation_frames: int = 10

    # Storage
    save_fp16: bool = False
    include_bbox_info: bool = True


@dataclass
class PreprocessingConfig:
    """Master preprocessing configuration."""
    dataset: Literal["phoenix2014t", "aslg_pc12"] = "phoenix2014t"
    data_root: str = "/path/to/PHOENIX-2014-T"
    output_dir: str = "data_cache"

    # Processing settings
    num_workers: int = 8
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    batch_size: int = 1

    # Stream configs
    rgb: RGBConfig = field(default_factory=RGBConfig)
    keypoints: KeypointsConfig = field(default_factory=KeypointsConfig)
    hands: HandCropConfig = field(default_factory=HandCropConfig)

    # Quality control
    min_video_frames: int = 10
    max_video_frames: int = 300
    skip_corrupted: bool = True

    # Debugging
    visualize: bool = False
    vis_output_dir: str = "preprocessing_debug"
    save_intermediate: bool = False


def phoenix2014t_config(data_root: str) -> PreprocessingConfig:
    """Optimized config for PHOENIX-2014T."""
    cfg = PreprocessingConfig(
        dataset="phoenix2014t",
        data_root=data_root,
        output_dir="data_cache",
        num_workers=8,
    )
    cfg.rgb.phoenix_centered = True
    cfg.rgb.use_person_detection = False
    cfg.rgb.num_frames = 64
    cfg.rgb.sampling_strategy = "uniform"

    cfg.keypoints.model_complexity = 1
    cfg.keypoints.temporal_smoothing = True

    cfg.hands.crop_size = (128, 128)
    cfg.hands.padding_factor = 1.5
    return cfg
