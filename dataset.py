import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from pathlib import Path
import numpy as np
from preprocessing import PhoenixKeypointExtractor


class PhoenixSignDataset(Dataset):
    """PyTorch Dataset for PHOENIX-2014T with augmentation"""
    
    # Number of coordinates per joint (x, y, z)
    COORDS_PER_JOINT = 3
    
    def __init__(self, root_dir, split='train', max_frames=250, augment=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_frames = max_frames
        self.augment = augment and (split == 'train')
        
        # Load annotations
        csv_path = self.root_dir / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
        self.df = pd.read_csv(csv_path, sep='|')
        
        # Setup tokenizer for glosses
        self.glosses = sorted(self.df['orth'].unique())
        self.gloss_to_idx = {g: i+1 for i, g in enumerate(self.glosses)}  # 0 reserved for padding
        
        # Cache for pre-extracted keypoints
        self.kps_dir = self.root_dir / 'features' / 'keypoints' / split
        self.kps_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize keypoint extractor (lazy loading)
        self.extractor = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence_name = row['name']
        
        # Try to load cached keypoints
        kps_path = self.kps_dir / f"{sequence_name}.npy"
        
        if kps_path.exists():
            keypoints = np.load(kps_path)
        else:
            # Extract keypoints from frames
            if self.extractor is None:
                self.extractor = PhoenixKeypointExtractor()
                
            frame_dir = self.root_dir / 'features' / 'fullFrame-210x260px' / self.split / sequence_name
            keypoints = self.extractor.extract_from_frames(frame_dir, self.max_frames)
            
            # Cache for future use
            np.save(kps_path, keypoints)
            
        # Convert to tensor
        keypoints = torch.FloatTensor(keypoints)
        
        # Apply augmentation for training
        if self.augment:
            keypoints = self._augment(keypoints)
            
        # Get gloss label
        gloss = row['orth']
        gloss_indices = [self.gloss_to_idx[gloss]]
        
        return {
            'keypoints': keypoints,
            'gloss': torch.LongTensor(gloss_indices),
            'gloss_text': gloss,
            'translation': row['translation'],
            'name': sequence_name
        }
    
    def _to_3d(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Reshape flattened (T, D) keypoints to (T, num_joints, 3) for augmentation."""
        T, D = keypoints.shape
        assert D % self.COORDS_PER_JOINT == 0, (
            f"Feature dim {D} not divisible by {self.COORDS_PER_JOINT}. "
            f"Check that keypoints are stored as [x,y,z] per joint."
        )
        num_joints = D // self.COORDS_PER_JOINT
        return keypoints.reshape(T, num_joints, self.COORDS_PER_JOINT)
    
    def _to_flat(self, keypoints: torch.Tensor) -> torch.Tensor:
        """Reshape (T, num_joints, 3) back to (T, D)."""
        T = keypoints.shape[0]
        return keypoints.reshape(T, -1)
    
    def _augment(self, keypoints):
        """Apply augmentation similar to the TensorFlow reference code."""
        # Reshape to (T, num_joints, 3) for spatial transforms
        kps_3d = self._to_3d(keypoints)
        
        # Spatial affine augmentation
        if random.random() < 0.5:
            kps_3d = self._spatial_random_affine(kps_3d)
            
        # Flatten back to (T, D) before temporal resampling
        keypoints = self._to_flat(kps_3d)
        
        # Temporal resampling (operates on flattened format fine)
        if random.random() < 0.5:
            keypoints = self._resample(keypoints, rate=(0.8, 1.2))
            
        return keypoints
    
    def _spatial_random_affine(self, keypoints, scale_range=(0.8, 1.2), 
                               rotation_range=(-30, 30), 
                               translation_range=(-0.1, 0.1)):
        """
        Apply random affine transformation to keypoints.
        
        Args:
            keypoints: (T, num_joints, 3) tensor
        Returns:
            Transformed keypoints with same shape
        """
        # Scale
        if scale_range:
            scale = random.uniform(*scale_range)
            keypoints = keypoints * scale
            
        # Rotation (around center 0.5, 0.5 on xy plane)
        if rotation_range and random.random() < 0.5:
            angle = random.uniform(*rotation_range) * np.pi / 180
            c, s = np.cos(angle), np.sin(angle)
            rot_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
            
            # Center, rotate, uncenter — only on x,y (first 2 coords)
            center = torch.tensor([0.5, 0.5], dtype=torch.float32)
            xy = keypoints[..., :2] - center      # (T, J, 2)
            xy = xy @ rot_matrix.T                 # rotate
            keypoints = torch.cat([xy + center, keypoints[..., 2:]], dim=-1)
            
        # Translation: broadcast (1, 1, 3) over (T, J, 3)
        if translation_range:
            translation = torch.FloatTensor(1, 1, 3).uniform_(*translation_range)
            keypoints = keypoints + translation
            
        return keypoints
    
    def _resample(self, keypoints, rate=(0.8, 1.2)):
        """Resample sequence in time. Works on (T, D) flattened format."""
        rate_val = random.uniform(*rate)
        current_len = keypoints.shape[0]
        new_len = max(1, int(current_len * rate_val))
            
        # Linear interpolation along time axis
        indices = torch.linspace(0, current_len - 1, new_len)
        indices_floor = indices.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=current_len - 1)
        alpha = (indices - indices_floor.float()).unsqueeze(1)  # (new_len, 1)
        
        keypoints_resampled = (1 - alpha) * keypoints[indices_floor] + alpha * keypoints[indices_ceil]
        
        return keypoints_resampled