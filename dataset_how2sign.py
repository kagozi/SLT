"""
How2Sign Dataset loader.

Loads raw (T, 543, 3) MediaPipe Holistic keypoints directly from the
HuggingFace How2Sign_Holistic cache and applies landmark selection +
normalization on-the-fly, matching the PHOENIX pipeline exactly.

Correct MediaPipe Holistic layout (543 landmarks):
    Pose        :   0 -  32  (33 landmarks)
    Face mesh   :  33 - 500  (468 landmarks,  face mesh idx i = holistic 33+i)
    Left hand   : 501 - 521  (21 landmarks)
    Right hand  : 522 - 542  (21 landmarks)

Selected 75 landmarks -> 225 features:
    13 pose  x 3 =  39
    21 lhand x 3 =  63
    21 rhand x 3 =  63
    20 face  x 3 =  60
    total: 75 joints x 3 = 225

Expected layout on the PVC (HF cache, no pre-processing needed):
    <root_dir>/                    # e.g. /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features
        metadata/
            how2sign_realigned_train.csv
            how2sign_realigned_val.csv
            how2sign_realigned_test.csv
        train/frontal/{SENTENCE_NAME}_holistic.npy   (T, 543, 3) float32
        val/frontal/
        test/frontal/

CSV columns expected: VIDEO_ID, VIDEO_NAME, SENTENCE_ID, SENTENCE_NAME,
                      START_REALIGNED, END_REALIGNED, SENTENCE
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from keypoint_utils import normalize_flat_keypoints, normalize_selected_keypoints
from translation_utils import encode_translation_target


# ── Landmark selection ───────────────────────────────────────────────────────

# 13 upper-body pose landmarks (indices into the 33-landmark pose block, 0-32)
POSE_SELECT = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# 21 hand landmarks each (correct holistic indices)
LHAND_SELECT = list(range(501, 522))
RHAND_SELECT = list(range(522, 543))

# 20 face landmarks — face mesh indices mapped to holistic space (33 + mesh_idx)
FACE_MESH_IDS = [
    61, 291, 0, 17, 39, 269, 181, 405,   # lips (8)
    70, 63, 105,                           # left eyebrow (3)
    300, 293, 334,                         # right eyebrow (3)
    33, 133,                               # left eye (2)
    263, 362,                              # right eye (2)
    1, 2,                                  # nose (2)
]
FACE_SELECT = [33 + i for i in FACE_MESH_IDS]

# Combined 75-landmark selection array (indices into 543-landmark holistic)
ALL_SELECT = np.array(POSE_SELECT + LHAND_SELECT + RHAND_SELECT + FACE_SELECT)
assert len(ALL_SELECT) == 75

# ── Positions within the 75-landmark array after selection ──────────────────
# Pose:       indices  0-12  (13 landmarks)
# Left hand:  indices 13-33  (21 landmarks)
# Right hand: indices 34-54  (21 landmarks)
# Face:       indices 55-74  (20 landmarks)
_P  = len(POSE_SELECT)    # 13
_LH = len(LHAND_SELECT)   # 21
_RH = len(RHAND_SELECT)   # 21

LHAND_OFFSET = _P            # 13
RHAND_OFFSET = _P + _LH     # 34

# Wrist positions in the selected 13-pose block
# POSE_SELECT[9]  = 15 = left wrist
# POSE_SELECT[10] = 16 = right wrist
LWRIST_IDX  = 9
RWRIST_IDX  = 10

# Shoulder positions in the selected 13-pose block
# POSE_SELECT[5] = 11 = left shoulder
# POSE_SELECT[6] = 12 = right shoulder
LSHOULDER_IDX = 5
RSHOULDER_IDX = 6


# ── Processing ───────────────────────────────────────────────────────────────

def _interpolate(arr: np.ndarray) -> np.ndarray:
    """Fill all-zero frames by linear interpolation. arr: (T, J, 3)."""
    T = len(arr)
    for i in range(T):
        if np.all(arr[i] == 0):
            last = i - 1
            while last >= 0 and np.all(arr[last] == 0):
                last -= 1
            nxt = i + 1
            while nxt < T and np.all(arr[nxt] == 0):
                nxt += 1
            if last >= 0 and nxt < T:
                alpha = (i - last) / (nxt - last)
                arr[i] = (1 - alpha) * arr[last] + alpha * arr[nxt]
            elif last >= 0:
                arr[i] = arr[last]
            elif nxt < T:
                arr[i] = arr[nxt]
    return arr


def process_holistic(raw: np.ndarray, max_frames: int) -> np.ndarray:
    """
    (T, 543, 3) -> (max_frames, 225) float32.

    Pipeline:
      1. Select 75 landmarks with correct indices
      2. Interpolate missing (all-zero) frames
      3. Wrist-relative hands
      4. Neck normalization (subtract shoulder midpoint from all joints)
      5. Flatten + pad/truncate to max_frames
    """
    T = raw.shape[0]

    # 1. Select 75 landmarks
    sel = raw[:, ALL_SELECT, :].copy()   # (T, 75, 3)

    # 2. Interpolate missing frames
    sel = _interpolate(sel)

    # 3. Wrist-relative hands (vectorized)
    lw = sel[:, LWRIST_IDX, :]           # (T, 3)
    rw = sel[:, RWRIST_IDX, :]
    lh = sel[:, LHAND_OFFSET:LHAND_OFFSET + _LH, :]
    rh = sel[:, RHAND_OFFSET:RHAND_OFFSET + _RH, :]
    lh_missing = np.all(lh == 0, axis=(1, 2))  # (T,)
    rh_missing = np.all(rh == 0, axis=(1, 2))
    sel[:, LHAND_OFFSET:LHAND_OFFSET + _LH, :] = np.where(
        lh_missing[:, None, None], lh, lh - lw[:, None, :])
    sel[:, RHAND_OFFSET:RHAND_OFFSET + _RH, :] = np.where(
        rh_missing[:, None, None], rh, rh - rw[:, None, :])

    # 4. Neck normalization (vectorized)
    neck = (sel[:, LSHOULDER_IDX, :] + sel[:, RSHOULDER_IDX, :]) * 0.5  # (T, 3)
    sel -= neck[:, None, :]

    # 5. Flatten + pad/truncate
    sel = normalize_selected_keypoints(sel)
    flat = sel.reshape(T, -1).astype(np.float32)   # (T, 225)
    flat = np.nan_to_num(flat)

    if T < max_frames:
        pad = np.zeros((max_frames - T, 225), dtype=np.float32)
        flat = np.vstack([flat, pad])
    else:
        flat = flat[:max_frames]

    return flat   # (max_frames, 225)


# ── Dataset ──────────────────────────────────────────────────────────────────

class How2SignDataset(Dataset):
    """
    PyTorch Dataset for How2Sign.

    Loads raw (T, 543, 3) holistic keypoints from raw/{split}/ and applies
    landmark selection + normalization on-the-fly via process_holistic().

    Args:
        root_dir:        Root of the HF holistic features tree
                         (e.g. /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features).
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

        # ── CSV: realigned only ───────────────────────────────────────────────
        meta_dir = self.root_dir / 'metadata'
        csv_path = meta_dir / f'how2sign_realigned_{split}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Realigned CSV not found: {csv_path}")

        df = pd.read_csv(csv_path, sep='\t')
        df.columns = [c.strip() for c in df.columns]

        # ── NPY files: {split}/frontal/{SENTENCE_NAME}_holistic.npy ──────────
        frontal_dir = self.root_dir / split / 'frontal'

        self.samples = []
        missing = 0
        for _, row in df.iterrows():
            sentence_name = str(row['SENTENCE_NAME']).strip()
            npy_path = frontal_dir / f"{sentence_name}_holistic.npy"

            if not npy_path.exists():
                missing += 1
                continue

            self.samples.append({
                'npy_path':      npy_path,
                'sentence_name': sentence_name,
                'sentence':      str(row['SENTENCE']) if pd.notna(row['SENTENCE']) else "",
                'pseudogloss':   str(row.get('PSEUDOGLOSS', ''))
                                 if pd.notna(row.get('PSEUDOGLOSS', float('nan')))
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

        raw = np.load(sample['npy_path'])             # (T, 543, 3)
        kps = process_holistic(raw, self.max_frames)  # (max_frames, 225)
        kps = normalize_flat_keypoints(kps.astype(np.float32))

        num_real_frames = max(
            1, int((np.linalg.norm(kps, axis=-1) != 0).sum())
        )

        keypoints = torch.FloatTensor(kps)

        if self.augment:
            keypoints = self._augment(keypoints)

        # Gloss (pseudo-gloss or dummy)
        gloss_text    = sample['pseudogloss']
        gloss_indices = torch.tensor([1], dtype=torch.long)
        if self.tokenizer and gloss_text:
            gloss_indices = self.tokenizer.encode(gloss_text)

        # Translation
        translation_text = sample['sentence']
        translation_ids  = None
        if self.bart_tokenizer is not None:
            translation_ids = encode_translation_target(self.bart_tokenizer, translation_text, max_length=128)

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
