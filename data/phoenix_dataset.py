"""
PyTorch Dataset for multistream SLT.
Loads preprocessed .pt files (rgb, hands, kpts) aligned by manifest frame_indices.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


class PhoenixMultiStreamDataset(Dataset):
    """
    Loads aligned RGB, Hands, and Keypoints tensors from preprocessed .pt files.

    Expected directory structure (from preprocessing scripts):
      data_cache/<dataset>/rgb/<split>/<video_id>.pt      → [T, C, H, W]
      data_cache/<dataset>/hands/<split>/<video_id>.pt    → [T, 2, C, Hh, Wh]
      data_cache/<dataset>/kpts/<split>/<video_id>.pt     → [T, D]

    Manifest (from RGB preprocessing) provides:
      video_id, orth (glosses), translation, frame_indices
    """

    def __init__(
        self,
        data_cache_dir: str,
        dataset: str,
        split: str,
        gloss_tokenizer=None,
        translation_tokenizer=None,
        num_frames: int = 64,
        max_gloss_len: int = 75,
        max_translation_len: int = 100,
        augment: bool = False,
    ):
        super().__init__()
        self.data_cache = Path(data_cache_dir) / dataset
        self.split = split
        self.num_frames = num_frames
        self.max_gloss_len = max_gloss_len
        self.max_translation_len = max_translation_len
        self.augment = augment
        self.gloss_tokenizer = gloss_tokenizer
        self.translation_tokenizer = translation_tokenizer

        # Load RGB manifest (primary — has annotations)
        rgb_manifest_path = self.data_cache / "manifests" / f"{split}_rgb_manifest.json"
        with open(rgb_manifest_path, "r") as f:
            rgb_manifest = json.load(f)

        # Build sample list from successful entries
        self.samples: List[Dict] = []
        for entry in rgb_manifest:
            if not entry.get("success", False):
                continue

            vid = entry["video_id"]
            rgb_path = self.data_cache / "rgb" / split / f"{vid}.pt"
            hands_path = self.data_cache / "hands" / split / f"{vid}.pt"
            kpts_path = self.data_cache / "kpts" / split / f"{vid}.pt"

            # All three modalities must exist
            if not (rgb_path.exists() and hands_path.exists() and kpts_path.exists()):
                continue

            self.samples.append({
                "video_id": vid,
                "rgb_path": str(rgb_path),
                "hands_path": str(hands_path),
                "kpts_path": str(kpts_path),
                "orth": entry.get("orth", ""),         # gloss string
                "translation": entry.get("translation", ""),
                "signer": entry.get("signer", ""),
            })

        print(f"[{split}] Loaded {len(self.samples)} multistream samples "
              f"(from {len(rgb_manifest)} RGB manifest entries)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Load preprocessed tensors
        rgb = torch.load(sample["rgb_path"], map_location="cpu", weights_only=True)      # [T, C, H, W]
        hands = torch.load(sample["hands_path"], map_location="cpu", weights_only=True)  # [T, 2, C, Hh, Wh]
        kpts = torch.load(sample["kpts_path"], map_location="cpu", weights_only=True)    # [T, D]


        # Ensure float32
        rgb = rgb.float()
        hands = hands.float()
        kpts = kpts.float()

        # Augmentation (training only)
        if self.augment:
            rgb, hands, kpts = self._augment(rgb, hands, kpts)

        # Normalize keypoints per-frame (from reference: mean/var normalization)
        kpts = self._normalize_kpts(kpts)

        # Prepare output dict
        out = {
            "video_id": sample["video_id"],
            "rgb": rgb,         # (T, C, H, W)
            "hands": hands,     # (T, 2, C, Hh, Wh)
            "kpts": kpts,       # (T, D)
            "num_frames": torch.tensor(rgb.shape[0], dtype=torch.long),
        }

        # Tokenize glosses (for CTC target)
        if self.gloss_tokenizer is not None and sample["orth"]:
            gloss_ids = self.gloss_tokenizer.encode(sample["orth"])
            out["gloss_ids"] = torch.tensor(gloss_ids[:self.max_gloss_len], dtype=torch.long)
            out["gloss_length"] = torch.tensor(len(gloss_ids[:self.max_gloss_len]), dtype=torch.long)

        # Tokenize translation (for BART target)
        if self.translation_tokenizer is not None and sample["translation"]:
            enc = self.translation_tokenizer(
                sample["translation"],
                max_length=self.max_translation_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            out["translation_ids"] = enc["input_ids"].squeeze(0)
            out["translation_mask"] = enc["attention_mask"].squeeze(0)

        return out

    def _normalize_kpts(self, kpts: torch.Tensor) -> torch.Tensor:
        """Per-frame normalization (from reference normalize function)."""
        mean = kpts.mean(dim=-1, keepdim=True)
        var = kpts.var(dim=-1, keepdim=True)
        return (kpts - mean) / (var.sqrt() + 1e-6)

    def _augment(
        self, rgb: torch.Tensor, hands: torch.Tensor, kpts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Temporal augmentation (from reference resample + spatial_random_affine).
        Applied consistently across all three streams.
        """
        T = rgb.shape[0]

        # Temporal resampling (stretch/compress uniformly across all streams)
        if torch.rand(1).item() < 0.5:
            rate = 0.8 + torch.rand(1).item() * 0.4  # [0.8, 1.2]
            new_T = max(1, int(T * rate))
            indices = torch.linspace(0, T - 1, new_T).long().clamp(0, T - 1)
            rgb = rgb[indices]
            hands = hands[indices]
            kpts = kpts[indices]

        # Spatial augmentation on keypoints (from reference spatial_random_affine)
        if torch.rand(1).item() < 0.5:
            # Scale
            scale = 0.9 + torch.rand(1).item() * 0.2
            kpts = kpts * scale

            # Small translation
            shift = (torch.rand(1, kpts.shape[-1]) - 0.5) * 0.1
            kpts = kpts + shift

        return rgb, hands, kpts


def collate_multistream(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function that pads variable-length sequences.
    """
    max_T = max(item["rgb"].shape[0] for item in batch)
    B = len(batch)

    # Get shapes from first item
    _, C_rgb, H_rgb, W_rgb = batch[0]["rgb"].shape
    _, _, C_h, H_h, W_h = batch[0]["hands"].shape
    D_kpts = batch[0]["kpts"].shape[-1]

    # Pre-allocate padded tensors
    rgb_padded = torch.zeros(B, max_T, C_rgb, H_rgb, W_rgb)
    hands_padded = torch.zeros(B, max_T, 2, C_h, H_h, W_h)
    kpts_padded = torch.zeros(B, max_T, D_kpts)
    mask = torch.zeros(B, max_T, dtype=torch.bool)

    video_ids = []
    gloss_ids_list = []
    gloss_lengths = []
    trans_ids_list = []
    trans_mask_list = []

    for i, item in enumerate(batch):
        T = item["rgb"].shape[0]
        rgb_padded[i, :T] = item["rgb"]
        hands_padded[i, :T] = item["hands"]
        kpts_padded[i, :T] = item["kpts"]
        mask[i, :T] = True

        video_ids.append(item["video_id"])

        if "gloss_ids" in item:
            gloss_ids_list.append(item["gloss_ids"])
            gloss_lengths.append(item["gloss_length"])

        if "translation_ids" in item:
            trans_ids_list.append(item["translation_ids"])
            trans_mask_list.append(item["translation_mask"])

    out = {
        "video_ids": video_ids,
        "rgb": rgb_padded,
        "hands": hands_padded,
        "kpts": kpts_padded,
        "mask": mask,
    }

    if gloss_ids_list:
        # Pad gloss targets to same length
        max_gl = max(g.shape[0] for g in gloss_ids_list)
        gloss_padded = torch.zeros(B, max_gl, dtype=torch.long)
        for i, g in enumerate(gloss_ids_list):
            gloss_padded[i, :g.shape[0]] = g
        out["gloss_targets"] = gloss_padded
        out["gloss_lengths"] = torch.stack(gloss_lengths)

    if trans_ids_list:
        out["translation_targets"] = torch.stack(trans_ids_list)
        out["translation_mask"] = torch.stack(trans_mask_list)

    return out


def build_dataloaders(
    data_cache_dir: str,
    dataset: str,
    gloss_tokenizer=None,
    translation_tokenizer=None,
    batch_size: int = 8,
    num_workers: int = 4,
    num_frames: int = 64,
) -> Dict[str, DataLoader]:
    """Build train/dev/test DataLoaders."""
    loaders = {}
    for split in ["train", "dev", "test"]:
        ds = PhoenixMultiStreamDataset(
            data_cache_dir=data_cache_dir,
            dataset=dataset,
            split=split,
            gloss_tokenizer=gloss_tokenizer,
            translation_tokenizer=translation_tokenizer,
            num_frames=num_frames,
            augment=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_multistream,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders