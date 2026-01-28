"""
PyTorch Dataset for PHOENIX-2014T video data.

Loads preprocessed tensors (RGB, keypoints, hands) and annotations.
Supports flexible stream selection for ablation studies.

IMPORTANT:
- In your manifests, `orth` is the gloss-like sequence (target for S2G/V2G).
- `translation` is the spoken language text (not used in Stage 1 V2G).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class PhoenixVideoDataset(Dataset):
    """
    PHOENIX-2014T dataset for video-to-gloss (orth) or video-to-text translation.

    Loads preprocessed tensors from:
        <data_cache_dir>/phoenix2014t/{rgb,kpts,hands}/{split}/<video_id>.pt

    Loads annotations from:
        <data_cache_dir>/phoenix2014t/manifests/{split}_rgb_manifest.json
    """

    def __init__(
        self,
        data_cache_dir: str,
        split: str = "train",
        gloss_vocab=None,  # Vocab object
        use_rgb: bool = True,
        use_keypoints: bool = False,
        use_hands: bool = False,
        max_length: Optional[int] = None,
        strict: bool = True,
    ):
        """
        Args:
            data_cache_dir: Path to preprocessed data root (e.g., "../data_cache")
            split: "train", "dev", or "test"
            gloss_vocab: Vocabulary for gloss tokens (built from `orth`)
            use_rgb: Load RGB stream
            use_keypoints: Load keypoints stream
            use_hands: Load hands stream
            max_length: Max token length for orth (None = no limit)
            strict: If True, raise on missing tensor files. If False, return zero tensors.
        """
        self.data_cache = Path(data_cache_dir) / "phoenix2014t"
        self.split = split
        self.gloss_vocab = gloss_vocab

        self.use_rgb = use_rgb
        self.use_keypoints = use_keypoints
        self.use_hands = use_hands
        self.max_length = max_length
        self.strict = strict

        manifest_path = self.data_cache / "manifests" / f"{split}_rgb_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # Keep only successful samples with valid orth
        def _valid_orth(s: Dict) -> bool:
            orth = s.get("orth", None)
            if orth is None:
                return False
            orth = str(orth).strip()
            if orth == "" or orth == "-1":
                return False
            return True

        self.samples = [s for s in manifest if s.get("success", False) and _valid_orth(s)]

        # Optional length filter
        if max_length is not None:
            self.samples = [s for s in self.samples if len(str(s["orth"]).split()) <= max_length]

        print(f"[PhoenixVideoDataset] Loaded {len(self.samples)} samples from split='{split}'")

        # Stream dirs
        self.rgb_dir = self.data_cache / "rgb" / split
        self.kpts_dir = self.data_cache / "kpts" / split
        self.hands_dir = self.data_cache / "hands" / split

    def __len__(self) -> int:
        return len(self.samples)

    def _load_tensor(self, path: Path, fallback_shape: Optional[tuple] = None) -> torch.Tensor:
        if path.exists():
            return torch.load(path, map_location="cpu")
        if self.strict:
            raise FileNotFoundError(f"Missing tensor: {path}")
        if fallback_shape is None:
            raise FileNotFoundError(f"Missing tensor: {path} (no fallback_shape provided)")
        return torch.zeros(*fallback_shape)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        video_id = sample["video_id"]

        orth = str(sample.get("orth", "")).strip()
        translation = str(sample.get("translation", "")).strip()

        data: Dict = {
            "video_id": video_id,
            "orth": orth,  # gloss-like target
            "translation": translation,
            "split": sample.get("split", self.split),
            "sequence_dir": sample.get("sequence_dir", ""),
            "frame_indices": sample.get("frame_indices", None),
        }

        # Load streams
        if self.use_rgb:
            rgb_path = self.rgb_dir / f"{video_id}.pt"
            # Expected shape: [T, 3, 256, 256]
            data["rgb"] = self._load_tensor(rgb_path, fallback_shape=(64, 3, 256, 256))

        if self.use_keypoints:
            kpts_path = self.kpts_dir / f"{video_id}.pt"
            # Expected shape: [T, D] e.g., [T, 498]
            data["keypoints"] = self._load_tensor(kpts_path, fallback_shape=(64, 498))

        if self.use_hands:
            hands_path = self.hands_dir / f"{video_id}.pt"
            # Expected shape: [T, 2, 3, 128, 128]
            data["hands"] = self._load_tensor(hands_path, fallback_shape=(64, 2, 3, 128, 128))

        # Encode orth → ids
        if self.gloss_vocab is not None:
            from vocab import encode
            orth_ids = encode(orth, self.gloss_vocab, add_bos_eos=True)
            data["orth_ids"] = torch.tensor(orth_ids, dtype=torch.long)

        return data

    def get_vocab_size(self) -> int:
        if self.gloss_vocab is None:
            raise ValueError("gloss_vocab is not set")
        return len(self.gloss_vocab.tokens)


def collate_video_batch(batch: List[Dict], pad_id: int = 0) -> Dict:
    """
    Collate function for video batches.

    Pads:
    - rgb: [B, T_max, 3, H, W]
    - orth_ids: [B, L_max]

    Creates:
    - src_key_padding_mask: [B, T_max] (True = padding)
    - tgt_key_padding_mask: [B, L_max] (True = padding)
    """
    video_ids = [b["video_id"] for b in batch]
    orth_texts = [b["orth"] for b in batch]
    translations = [b.get("translation", "") for b in batch]

    collated: Dict = {
        "video_ids": video_ids,
        "orth_texts": orth_texts,
        "translations": translations,
    }

    # RGB
    if "rgb" in batch[0]:
        rgb_list = [b["rgb"] for b in batch]  # [T, 3, H, W]
        rgb_padded = pad_sequence(rgb_list, batch_first=True, padding_value=0.0)
        collated["rgb"] = rgb_padded

        lengths = torch.tensor([x.size(0) for x in rgb_list], dtype=torch.long)
        T_max = rgb_padded.size(1)
        src_mask = torch.arange(T_max)[None, :] >= lengths[:, None]
        collated["src_key_padding_mask"] = src_mask

    # Keypoints
    if "keypoints" in batch[0]:
        kpts_list = [b["keypoints"] for b in batch]
        kpts_padded = pad_sequence(kpts_list, batch_first=True, padding_value=0.0)
        collated["keypoints"] = kpts_padded

    # Hands
    if "hands" in batch[0]:
        hands_list = [b["hands"] for b in batch]
        hands_padded = pad_sequence(hands_list, batch_first=True, padding_value=0.0)
        collated["hands"] = hands_padded

    # orth_ids
    if "orth_ids" in batch[0]:
        tgt_list = [b["orth_ids"] for b in batch]
        tgt_padded = pad_sequence(tgt_list, batch_first=True, padding_value=pad_id)
        collated["orth_ids"] = tgt_padded

        tgt_lengths = torch.tensor([x.size(0) for x in tgt_list], dtype=torch.long)
        L_max = tgt_padded.size(1)
        tgt_mask = torch.arange(L_max)[None, :] >= tgt_lengths[:, None]
        collated["tgt_key_padding_mask"] = tgt_mask

    return collated


if __name__ == "__main__":
    from .vocab import build_word_vocab

    manifest_path = Path("data_cache") / "phoenix2014t" / "manifests" / "train_rgb_manifest.json"
    manifest = json.load(open(manifest_path, "r", encoding="utf-8"))

    orths = [s["orth"] for s in manifest if s.get("success") and str(s.get("orth", "")).strip() not in ("", "-1")]
    gloss_vocab = build_word_vocab(
        orths,
        specials=["<pad>", "<unk>", "<start>", "<end>"],
        min_freq=1,
    )

    dataset = PhoenixVideoDataset(
        data_cache_dir="data_cache",
        split="dev",
        gloss_vocab=gloss_vocab,
        use_rgb=True,
        use_keypoints=False,
        use_hands=False,
        strict=False,
    )

    sample = dataset[0]
    print("Video ID:", sample["video_id"])
    print("RGB shape:", sample["rgb"].shape)
    print("ORTH:", sample["orth"])
    print("ORTH IDs:", sample["orth_ids"])

    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_video_batch(b, pad_id=gloss_vocab.pad_id),
    )
    batch = next(iter(loader))
    print("Batch RGB:", batch["rgb"].shape)
    print("Batch ORTH IDs:", batch["orth_ids"].shape)
    print("Src mask:", batch["src_key_padding_mask"].shape)
    print("Tgt mask:", batch["tgt_key_padding_mask"].shape)
