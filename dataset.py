import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random

class PhoenixSignDataset(Dataset):
    """PyTorch Dataset for PHOENIX-2014T with augmentation"""
    
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
        gloss_indices = [self.gloss_to_idx[gloss]]  # Single gloss for now (will extend to sequences later)
        
        return {
            'keypoints': keypoints,
            'gloss': torch.LongTensor(gloss_indices),
            'gloss_text': gloss,
            'translation': row['translation'],
            'name': sequence_name
        }
    
    def _augment(self, keypoints):
        """Apply augmentation similar to the TensorFlow code"""
        
        # Spatial affine augmentation
        if random.random() < 0.5:
            keypoints = self._spatial_random_affine(keypoints)
            
        # Temporal resampling
        if random.random() < 0.5:
            keypoints = self._resample(keypoints, rate=(0.8, 1.2))
            
        return keypoints
    
    def _spatial_random_affine(self, keypoints, scale_range=(0.8, 1.2), 
                               rotation_range=(-30, 30), 
                               translation_range=(-0.1, 0.1)):
        """Apply random affine transformation to keypoints"""
        
        # Scale
        if scale_range:
            scale = random.uniform(*scale_range)
            keypoints = keypoints * scale
            
        # Rotation (around center 0.5, 0.5)
        if rotation_range and random.random() < 0.5:
            angle = random.uniform(*rotation_range) * np.pi / 180
            c, s = np.cos(angle), np.sin(angle)
            rot_matrix = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)
            
            # Apply rotation to x,y coordinates (first 2 dims)
            keypoints_reshaped = keypoints.reshape(-1, 3)
            keypoints_reshaped[:, :2] = keypoints_reshaped[:, :2] @ rot_matrix[:2, :2].T
            keypoints = keypoints_reshaped.reshape(keypoints.shape)
            
        # Translation
        if translation_range:
            translation = torch.FloatTensor(3).uniform_(*translation_range)
            keypoints = keypoints + translation
            
        return keypoints
    
    def _resample(self, keypoints, rate=(0.8, 1.2)):
        """Resample sequence in time"""
        rate = random.uniform(*rate)
        current_len = keypoints.shape[0]
        new_len = int(current_len * rate)
        
        if new_len <= 0:
            return keypoints
            
        # Simple linear interpolation
        indices = torch.linspace(0, current_len - 1, new_len)
        indices_floor = indices.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=current_len - 1)
        alpha = (indices - indices_floor.float()).unsqueeze(1)
        
        keypoints_resampled = (1 - alpha) * keypoints[indices_floor] + alpha * keypoints[indices_ceil]
        
        return keypoints_resampled