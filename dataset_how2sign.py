"""
How2Sign Dataset loader — RGB-extracted keypoints.

Expected layout on the PVC:
    <root_dir>/
        annotations/
            how2sign_train.csv   (tab-sep, cols: SENTENCE_NAME, SENTENCE, [PSEUDOGLOSS], …)
            how2sign_val.csv
            how2sign_test.csv
        keypoints/
            train/{SENTENCE_NAME}.npy   (T, 225) float32 — pre-extracted by MediaPipe
            val/
            test/
        train/                          (pre-trimmed mp4 clips, available for future use)
            {SENTENCE_NAME}.mp4
        val/
        test/

Produces (max_frames, 225) tensors identical to the PHOENIX pipeline.
Compatible with the existing collate_fn and Trainer.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class How2SignDataset(Dataset):
    """
    PyTorch Dataset for How2Sign (RGB-extracted keypoints).

    Args:
        root_dir:        Root of the how2sign_rgb data tree.
        split:           'train', 'val', or 'test'.
        max_frames:      Pad / truncate to this many frames.
        augment:         Spatial + temporal augmentation (train split only).
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

        csv_path = self.root_dir / 'annotations' / f'how2sign_{split}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Annotations CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, sep='\t')

        kps_dir   = self.root_dir / 'keypoints' / split
        video_dir = self.root_dir / split

        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            sentence_name = str(row['SENTENCE_NAME'])
            npy_path = kps_dir / f"{sentence_name}.npy"

            if not npy_path.exists():
                missing += 1
                continue

            self.samples.append({
                'npy_path':      npy_path,
                'video_path':    video_dir / f"{sentence_name}.mp4",
                'sentence_name': sentence_name,
                'sentence':      str(row['SENTENCE']) if pd.notna(row['SENTENCE']) else "",
                'pseudogloss':   str(row['PSEUDOGLOSS'])
                                 if 'PSEUDOGLOSS' in row and pd.notna(row['PSEUDOGLOSS'])
                                 else "",
            })

        if missing:
            print(f"  Warning: How2Sign [{split}]: "
                  f"{missing}/{len(df)} rows have no matching .npy")
        print(f"  How2Sign [{split}]: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        keypoints = np.load(sample['npy_path']).astype(np.float32)   # (T, 225)

        num_real_frames = max(
            1, int((np.linalg.norm(keypoints, axis=-1) != 0).sum())
        )

        keypoints = torch.FloatTensor(keypoints)

        if self.augment:
            keypoints = self._augment(keypoints)

        # Pad / truncate
        T = keypoints.shape[0]
        if T > self.max_frames:
            keypoints = keypoints[:self.max_frames]
            num_real_frames = min(num_real_frames, self.max_frames)
        elif T < self.max_frames:
            pad = torch.zeros(self.max_frames - T, keypoints.shape[1])
            keypoints = torch.cat([keypoints, pad], dim=0)

        # Gloss (pseudo-gloss or dummy)
        gloss_text    = sample['pseudogloss']
        gloss_indices = torch.tensor([1], dtype=torch.long)
        if self.tokenizer and gloss_text:
            gloss_indices = self.tokenizer.encode(gloss_text)

        # Translation
        translation_text = sample['sentence']
        translation_ids  = None
        if self.bart_tokenizer is not None:
            encoded = self.bart_tokenizer(
                translation_text, max_length=128, truncation=True,
                padding=False, return_tensors='pt'
            )
            translation_ids = encoded['input_ids'].squeeze(0)

        result = {
            'keypoints':   keypoints,           # (max_frames, 225)
            'gloss':       gloss_indices,
            'gloss_text':  gloss_text,
            'translation': translation_text,
            'name':        sample['sentence_name'],
            'num_frames':  num_real_frames,
            'video_path':  str(sample['video_path']),
        }
        if translation_ids is not None:
            result['translation_ids'] = translation_ids

        return result

    # ── Augmentation ────────────────────────────────────────────────────────

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
