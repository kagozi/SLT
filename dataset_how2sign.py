"""
How2Sign Dataset loader for the PSewmuthu/How2Sign_Holistic dataset.

Loads pre-extracted MediaPipe Holistic .npy files (543 landmarks x 3 coords)
and selects 75 landmarks (33 pose + 21 left hand + 21 right hand) to produce
(T, 225) feature vectors -- identical to the PHOENIX pipeline.

Usage:
    ds = How2SignDataset(root_dir='path/to/how2sign_holistic_features',
                         split='train', max_frames=300, bart_tokenizer=bart_tok)
    sample = ds[0]
    # sample['keypoints'] -> (max_frames, 225) tensor
    # sample['translation'] -> "And I call them decorative elements..."
    # sample['gloss_text'] -> ""  (no glosses available)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import random
from pathlib import Path
import numpy as np


# --- MediaPipe Holistic landmark indices ---

# Full Holistic: 543 landmarks
#   Pose:        0 - 32   (33 landmarks)
#   Face:       33 - 500  (468 landmarks)
#   Left hand: 501 - 521  (21 landmarks)
#   Right hand: 522 - 542 (21 landmarks)

POSE_IDX  = list(range(0, 33))      # 33 pose landmarks
LHAND_IDX = list(range(501, 522))    # 21 left hand landmarks
RHAND_IDX = list(range(522, 543))    # 21 right hand landmarks

# 75 landmarks total -> 75 x 3 = 225 features (matches PHOENIX)
SELECT_IDX = np.array(POSE_IDX + LHAND_IDX + RHAND_IDX)  # (75,)


def holistic_to_pose_hands(data):
    """Select 75 landmarks from 543 and flatten to (T, 225).

    Args:
        data: np.ndarray of shape (T, 543, 3)
    Returns:
        np.ndarray of shape (T, 225)
    """
    selected = data[:, SELECT_IDX, :]  # (T, 75, 3)
    return selected.reshape(data.shape[0], -1)  # (T, 225)


class How2SignDataset(Dataset):
    """PyTorch Dataset for How2Sign with pre-extracted MediaPipe Holistic keypoints.

    Compatible with the same collate_fn and Trainer as PhoenixSignDataset.
    Since How2Sign has no gloss annotations, gloss fields return dummy values.
    """

    COORDS_PER_JOINT = 3

    def __init__(self, root_dir, split='train', max_frames=300, augment=True,
                 tokenizer=None, bart_tokenizer=None):
        """
        Args:
            root_dir: Path to how2sign_holistic_features/
            split: 'train', 'val', or 'test'
            max_frames: Maximum sequence length (pad/truncate)
            augment: Enable augmentation (train only)
            tokenizer: GlossTokenizer (unused for How2Sign, kept for API compat)
            bart_tokenizer: BartTokenizer for translation targets
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.max_frames = max_frames
        self.augment = augment and (split == 'train')
        self.tokenizer = tokenizer
        self.bart_tokenizer = bart_tokenizer

        # -- Load metadata CSV --
        csv_name = f"how2sign_realigned_{split}.csv"
        csv_path = self.root_dir / "metadata" / csv_name
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata not found: {csv_path}")

        self.df = pd.read_csv(csv_path, sep='\t')

        # -- NPY directory --
        self.npy_dir = self.root_dir / split / "frontal"
        if not self.npy_dir.exists():
            raise FileNotFoundError(f"NPY directory not found: {self.npy_dir}")

        # -- Build filename mapping --
        # CSV SENTENCE_NAME: "--7E2sU6zP4_10-5-rgb_front"
        # NPY filename:      "--7E2sU6zP4_10-5-rgb_front_holistic.npy"
        available_npys = {f.stem: f for f in self.npy_dir.glob("*.npy")}

        # Map CSV rows to npy files
        self.samples = []
        missing = 0
        for _, row in self.df.iterrows():
            sentence_name = row['SENTENCE_NAME']
            # Try with _holistic suffix first, then direct
            npy_stem = f"{sentence_name}_holistic"

            if npy_stem in available_npys:
                npy_path = available_npys[npy_stem]
            elif sentence_name in available_npys:
                npy_path = available_npys[sentence_name]
            else:
                missing += 1
                continue

            self.samples.append({
                'npy_path': npy_path,
                'sentence_name': sentence_name,
                'sentence': str(row['SENTENCE']) if pd.notna(row['SENTENCE']) else "",
            })

        if missing > 0:
            print(f"  Warning: How2Sign [{split}]: {missing}/{len(self.df)} CSV rows have no matching .npy")
        print(f"  How2Sign [{split}]: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # -- Load and adapt keypoints --
        raw = np.load(sample['npy_path'])  # (T, 543, 3)

        # Select 75 landmarks -> (T, 225)
        keypoints = holistic_to_pose_hands(raw)

        # Count real frames (non-zero)
        frame_norms = np.linalg.norm(keypoints, axis=-1)
        num_real_frames = max(1, int((frame_norms != 0).sum()))

        keypoints = torch.FloatTensor(keypoints)

        # -- Augment --
        if self.augment:
            keypoints = self._augment(keypoints)

        # -- Pad / truncate --
        T = keypoints.shape[0]
        if T > self.max_frames:
            keypoints = keypoints[:self.max_frames]
            num_real_frames = min(num_real_frames, self.max_frames)
        elif T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, keypoints.shape[1])
            keypoints = torch.cat([keypoints, pad], dim=0)

        # -- Gloss (dummy -- How2Sign has no gloss annotations) --
        gloss_text = ""
        gloss_indices = torch.tensor([1], dtype=torch.long)  # dummy blank token
        if self.tokenizer:
            gloss_indices = self.tokenizer.encode(gloss_text)

        # -- Translation --
        translation_text = sample['sentence']

        translation_ids = None
        if self.bart_tokenizer is not None:
            encoded = self.bart_tokenizer(
                translation_text, max_length=128, truncation=True,
                padding=False, return_tensors='pt'
            )
            translation_ids = encoded['input_ids'].squeeze(0)

        result = {
            'keypoints': keypoints,          # (max_frames, 225)
            'gloss': gloss_indices,           # dummy for How2Sign
            'gloss_text': gloss_text,         # "" always
            'translation': translation_text,  # English sentence
            'name': sample['sentence_name'],
            'num_frames': num_real_frames,
        }
        if translation_ids is not None:
            result['translation_ids'] = translation_ids

        return result

    # --- Augmentation (same as PHOENIX) ---

    def _to_3d(self, kps):
        T, D = kps.shape
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
            center = torch.tensor([0.5, 0.5], dtype=torch.float32)
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