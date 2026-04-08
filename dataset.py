import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from pathlib import Path
import numpy as np
from preprocessing import PhoenixKeypointExtractor
from keypoint_utils import normalize_flat_keypoints
from translation_utils import encode_translation_target


class PhoenixSignDataset(Dataset):
    """PyTorch Dataset for PHOENIX-2014T with augmentation and optional translation targets."""
    
    COORDS_PER_JOINT = 3
    
    def __init__(self, root_dir, split='train', max_frames=250, augment=True,
                 tokenizer=None, bart_tokenizer=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_frames = max_frames
        self.augment = augment and (split == 'train')
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer  # For BART translation targets
        
        csv_path = self.root_dir / 'annotations' / 'manual' / f'PHOENIX-2014-T.{split}.corpus.csv'
        self.df = pd.read_csv(csv_path, sep='|')
        
        all_words = set()
        for orth in self.df['orth']:
            if isinstance(orth, str):
                all_words.update(orth.strip().upper().split())
        self.gloss_words = sorted(all_words)
        
        self.kps_dir = self.root_dir / 'features' / 'keypoints' / split
        self.kps_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence_name = row['name']
        
        kps_path = self.kps_dir / f"{sequence_name}.npy"
        if kps_path.exists():
            keypoints = np.load(kps_path)
        else:
            if self.extractor is None:
                self.extractor = PhoenixKeypointExtractor()
            frame_dir = self.root_dir / 'features' / 'fullFrame-210x260px' / self.split / sequence_name
            keypoints = self.extractor.extract_from_frames(frame_dir, self.max_frames)
            np.save(kps_path, keypoints)

        keypoints = normalize_flat_keypoints(keypoints.astype(np.float32))
        
        frame_norms = np.linalg.norm(keypoints, axis=-1)
        num_real_frames = max(1, int((frame_norms != 0).sum()))
        
        keypoints = torch.FloatTensor(keypoints)
        
        if self.augment:
            keypoints = self._augment(keypoints)
        
        T = keypoints.shape[0]
        if T > self.max_frames:
            keypoints = keypoints[:self.max_frames]
            num_real_frames = min(num_real_frames, self.max_frames)
        elif T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, keypoints.shape[1])
            keypoints = torch.cat([keypoints, pad], dim=0)
        
        # Gloss tokens
        gloss_text = row['orth'] if isinstance(row['orth'], str) else ""
        gloss_indices = self.tokenizer.encode(gloss_text) if self.tokenizer else torch.tensor([1], dtype=torch.long)
        
        # Translation text
        translation_text = row['translation'] if isinstance(row['translation'], str) else ""
        
        # BART translation token ids (if tokenizer provided)
        translation_ids = None
        if self.bart_tokenizer is not None:
            translation_ids = encode_translation_target(self.bart_tokenizer, translation_text, max_length=128)
        
        result = {
            'keypoints': keypoints,
            'gloss': gloss_indices,
            'gloss_text': gloss_text,
            'translation': translation_text,
            'name': sequence_name,
            'num_frames': num_real_frames,
        }
        if translation_ids is not None:
            result['translation_ids'] = translation_ids
        
        return result
    
    # ─── Augmentation ────────────────────────────────────────────────
    
    def _to_3d(self, kps):
        T, D = kps.shape
        if D % self.COORDS_PER_JOINT != 0:
            return kps.unsqueeze(-1)
        return kps.reshape(T, D // self.COORDS_PER_JOINT, self.COORDS_PER_JOINT)
    
    def _to_flat(self, kps):
        return kps.reshape(kps.shape[0], -1)
    
    def _augment(self, keypoints):
        kps_3d = self._to_3d(keypoints)
        if random.random() < 0.5:
            kps_3d = self._spatial_random_affine(kps_3d)
        keypoints = self._to_flat(kps_3d)
        if random.random() < 0.5:
            keypoints = self._resample(keypoints, rate=(0.8, 1.2))
        return keypoints
    
    def _spatial_random_affine(self, kps, scale_range=(0.8, 1.2),
                               rotation_range=(-30, 30),
                               translation_range=(-0.1, 0.1)):
        if scale_range:
            kps = kps * random.uniform(*scale_range)
        if rotation_range and random.random() < 0.5:
            angle = random.uniform(*rotation_range) * np.pi / 180
            c, s = np.cos(angle), np.sin(angle)
            rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
            center = torch.zeros(2, dtype=torch.float32)
            xy = kps[..., :2] - center
            kps = torch.cat([xy @ rot.T + center, kps[..., 2:]], dim=-1)
        if translation_range:
            kps = kps + torch.FloatTensor(1, 1, 3).uniform_(*translation_range)
        return kps
    
    def _resample(self, kps, rate=(0.8, 1.2)):
        T = kps.shape[0]
        new_len = max(1, int(T * random.uniform(*rate)))
        idx = torch.linspace(0, T - 1, new_len)
        fl = idx.long()
        cl = torch.clamp(fl + 1, max=T - 1)
        alpha = (idx - fl.float()).unsqueeze(1)
        return (1 - alpha) * kps[fl] + alpha * kps[cl]
