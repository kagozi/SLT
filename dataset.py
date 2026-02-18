import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from pathlib import Path
import numpy as np
from preprocessing import PhoenixKeypointExtractor


class PhoenixSignDataset(Dataset):
    """PyTorch Dataset for PHOENIX-2014T with augmentation"""
    
    COORDS_PER_JOINT = 3
    
    def __init__(self, root_dir, split='train', max_frames=250, augment=True,
                 tokenizer=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_frames = max_frames
        self.augment = augment and (split == 'train')
        self.tokenizer = tokenizer  # Must be set before training
        
        # Load annotations
        csv_path = self.root_dir / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
        self.df = pd.read_csv(csv_path, sep='|')
        
        # Collect all unique gloss WORDS (not sentences) for vocab building
        all_words = set()
        for orth in self.df['orth']:
            if isinstance(orth, str):
                all_words.update(orth.strip().upper().split())
        self.gloss_words = sorted(all_words)
        
        # Cache for pre-extracted keypoints
        self.kps_dir = self.root_dir / 'features' / 'keypoints' / split
        self.kps_dir.mkdir(parents=True, exist_ok=True)
        
        # No extractor in dataset — use preextract_keypoints.py first
        self.extractor = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence_name = row['name']
        
        # Load cached keypoints
        kps_path = self.kps_dir / f"{sequence_name}.npy"
        
        if kps_path.exists():
            keypoints = np.load(kps_path)
        else:
            # Lazy fallback extraction (shouldn't happen if preextracted)
            if self.extractor is None:
                self.extractor = PhoenixKeypointExtractor()
            frame_dir = self.root_dir / 'features' / 'fullFrame-210x260px' / self.split / sequence_name
            keypoints = self.extractor.extract_from_frames(frame_dir, self.max_frames)
            np.save(kps_path, keypoints)
        
        # Count actual non-padding frames BEFORE any augmentation
        # (frames with all zeros are padding from preprocessing)
        frame_norms = np.linalg.norm(keypoints, axis=-1)
        num_real_frames = int((frame_norms != 0).sum())
        if num_real_frames == 0:
            num_real_frames = 1  # at least 1 frame
            
        # Convert to tensor
        keypoints = torch.FloatTensor(keypoints)
        
        # Apply augmentation
        if self.augment:
            keypoints = self._augment(keypoints)
        
        # Enforce max_frames after augmentation (resampling can change length)
        T = keypoints.shape[0]
        if T > self.max_frames:
            keypoints = keypoints[:self.max_frames]
            num_real_frames = min(num_real_frames, self.max_frames)
        elif T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, keypoints.shape[1])
            keypoints = torch.cat([keypoints, pad], dim=0)
            
        # Get gloss label — WORD-LEVEL tokenization
        gloss_text = row['orth'] if isinstance(row['orth'], str) else ""
        
        if self.tokenizer is not None:
            gloss_indices = self.tokenizer.encode(gloss_text)
        else:
            # Fallback: return dummy
            gloss_indices = torch.tensor([1], dtype=torch.long)
        
        return {
            'keypoints': keypoints,
            'gloss': gloss_indices,
            'gloss_text': gloss_text,
            'translation': row['translation'] if isinstance(row['translation'], str) else "",
            'name': sequence_name,
            'num_frames': num_real_frames,
        }
    
    def _to_3d(self, keypoints):
        T, D = keypoints.shape
        if D % self.COORDS_PER_JOINT != 0:
            return keypoints.unsqueeze(-1)  # safety fallback
        num_joints = D // self.COORDS_PER_JOINT
        return keypoints.reshape(T, num_joints, self.COORDS_PER_JOINT)
    
    def _to_flat(self, keypoints):
        T = keypoints.shape[0]
        return keypoints.reshape(T, -1)
    
    def _augment(self, keypoints):
        kps_3d = self._to_3d(keypoints)
        
        if random.random() < 0.5:
            kps_3d = self._spatial_random_affine(kps_3d)
            
        keypoints = self._to_flat(kps_3d)
        
        if random.random() < 0.5:
            keypoints = self._resample(keypoints, rate=(0.8, 1.2))
            
        return keypoints
    
    def _spatial_random_affine(self, keypoints, scale_range=(0.8, 1.2), 
                               rotation_range=(-30, 30), 
                               translation_range=(-0.1, 0.1)):
        if scale_range:
            scale = random.uniform(*scale_range)
            keypoints = keypoints * scale
            
        if rotation_range and random.random() < 0.5:
            angle = random.uniform(*rotation_range) * np.pi / 180
            c, s = np.cos(angle), np.sin(angle)
            rot_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
            center = torch.tensor([0.5, 0.5], dtype=torch.float32)
            xy = keypoints[..., :2] - center
            xy = xy @ rot_matrix.T
            keypoints = torch.cat([xy + center, keypoints[..., 2:]], dim=-1)
            
        if translation_range:
            translation = torch.FloatTensor(1, 1, 3).uniform_(*translation_range)
            keypoints = keypoints + translation
            
        return keypoints
    
    def _resample(self, keypoints, rate=(0.8, 1.2)):
        rate_val = random.uniform(*rate)
        current_len = keypoints.shape[0]
        new_len = max(1, int(current_len * rate_val))
            
        indices = torch.linspace(0, current_len - 1, new_len)
        indices_floor = indices.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=current_len - 1)
        alpha = (indices - indices_floor.float()).unsqueeze(1)
        
        return (1 - alpha) * keypoints[indices_floor] + alpha * keypoints[indices_ceil]