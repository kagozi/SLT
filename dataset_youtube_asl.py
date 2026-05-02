"""
YouTube-ASL Dataset loader.

Loads raw (T, 543, 3) MediaPipe Holistic keypoints saved by
download_youtube_asl.py and applies the same landmark selection +
normalisation pipeline used by How2SignDataset (process_holistic).
The __getitem__ output is identical to How2SignDataset so the same
Trainer, collate_fn, and model code works unchanged.

Expected layout (written by download_youtube_asl.py):
    <root_dir>/
        metadata/
            youtube_asl_train.csv    (tab-sep)
            youtube_asl_val.csv
            youtube_asl_test.csv
        keypoints/
            train/{SENTENCE_NAME}.npy   # (T, 543, 3) float32
            val/
            test/

CSV columns:
    VIDEO_ID, SEGMENT_ID, SENTENCE_NAME, START, END, SENTENCE, PSEUDOGLOSS
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset_how2sign import process_holistic      # reuse exact same pipeline
from keypoint_utils import normalize_flat_keypoints
from translation_utils import encode_translation_target


class YouTubeASLDataset(Dataset):
    """
    PyTorch Dataset for YouTube-ASL.

    Args:
        root_dir:        Root of the YouTube-ASL data tree
                         (e.g. /data/youtube_asl).
        split:           'train', 'val', or 'test'.
        max_frames:      Pad / truncate to this many frames.
        augment:         Spatial + temporal augmentation (train only).
        tokenizer:       GlossTokenizer for pseudo-gloss encoding (optional).
        bart_tokenizer:  HuggingFace BartTokenizer for translation targets (optional).
    """

    COORDS_PER_JOINT = 3

    def __init__(self, root_dir, split='train', max_frames=300, augment=True,
                 tokenizer=None, bart_tokenizer=None):
        self.root_dir       = Path(root_dir)
        self.split          = split
        self.max_frames     = max_frames
        self.augment        = augment and (split == 'train')
        self.tokenizer      = tokenizer
        self.bart_tokenizer = bart_tokenizer

        meta_dir = self.root_dir / 'metadata'
        csv_path = meta_dir / f'youtube_asl_{split}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f'YouTube-ASL metadata not found: {csv_path}')

        df = pd.read_csv(csv_path, sep='\t')
        df.columns = [c.strip() for c in df.columns]

        kp_dir = self.root_dir / 'keypoints' / split

        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            name     = str(row['SENTENCE_NAME']).strip()
            npy_path = kp_dir / f'{name}.npy'
            if not npy_path.exists():
                missing += 1
                continue
            self.samples.append({
                'npy_path':      npy_path,
                'sentence_name': name,
                'sentence':      str(row['SENTENCE']) if pd.notna(row['SENTENCE']) else '',
                'pseudogloss':   str(row.get('PSEUDOGLOSS', ''))
                                 if pd.notna(row.get('PSEUDOGLOSS', float('nan')))
                                 else '',
            })

        if missing:
            print(f'  Warning: YouTubeASL [{split}]: '
                  f'{missing}/{len(df)} rows have no matching .npy')
        print(f'  YouTubeASL [{split}]: {len(self.samples)} samples loaded')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        raw = np.load(sample['npy_path'])              # (T, 543, 3)
        kps = process_holistic(raw, self.max_frames)   # (max_frames, 225)
        kps = normalize_flat_keypoints(kps.astype(np.float32))

        num_real_frames = max(
            1, int((np.linalg.norm(kps, axis=-1) != 0).sum())
        )

        keypoints = torch.FloatTensor(kps)

        if self.augment:
            keypoints = self._augment(keypoints)

        gloss_text    = sample['pseudogloss']
        gloss_indices = torch.tensor([1], dtype=torch.long)
        if self.tokenizer and gloss_text:
            gloss_indices = self.tokenizer.encode(gloss_text)

        translation_text = sample['sentence']
        translation_ids  = None
        if self.bart_tokenizer is not None:
            translation_ids = encode_translation_target(
                self.bart_tokenizer, translation_text, max_length=128
            )

        result = {
            'keypoints':   keypoints,
            'gloss':       gloss_indices,
            'gloss_text':  gloss_text,
            'translation': translation_text,
            'name':        sample['sentence_name'],
            'num_frames':  num_real_frames,
        }
        if translation_ids is not None:
            result['translation_ids'] = translation_ids

        return result

    # ── Augmentation (identical to How2SignDataset) ──────────────────────────

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
        fl  = idx.long()
        cl  = torch.clamp(fl + 1, max=T - 1)
        alpha = (idx - fl.float()).unsqueeze(1)
        return (1 - alpha) * kps[fl] + alpha * kps[cl]
