"""
Vocabulary building and token encoding/decoding utilities.

Supports building vocabularies from text corpora and converting between
text and token IDs.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Iterable, Any


@dataclass
class Vocab:
    """
    Vocabulary object for token-to-ID mapping.

    Attributes:
        tokens: List of all tokens (including special tokens)
        token_to_id: Map from token string to ID
        id_to_token: Map from ID to token string
        pad_id: ID for padding token
        unk_id: ID for unknown token
        bos_id: ID for begin-of-sequence token (optional)
        eos_id: ID for end-of-sequence token (optional)
    """
    tokens: List[str]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    pad_id: int
    unk_id: int
    bos_id: int | None = None
    eos_id: int | None = None

    def __len__(self):
        """Return vocabulary size."""
        return len(self.tokens)

    def __repr__(self):
        return (
            f"Vocab(size={len(self.tokens)}, "
            f"pad_id={self.pad_id}, unk_id={self.unk_id}, "
            f"bos_id={self.bos_id}, eos_id={self.eos_id})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize vocab to a JSON-safe dict.
        NOTE: We only need tokens + special ids; mappings can be reconstructed.
        """
        return {
            "tokens": list(self.tokens),
            "pad_id": int(self.pad_id),
            "unk_id": int(self.unk_id),
            "bos_id": None if self.bos_id is None else int(self.bos_id),
            "eos_id": None if self.eos_id is None else int(self.eos_id),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Vocab":
        """
        Restore vocab from dict produced by to_dict().
        Rebuild mappings deterministically from tokens.
        """
        tokens = data["tokens"]
        token_to_id = {tok: i for i, tok in enumerate(tokens)}
        id_to_token = {i: tok for tok, i in token_to_id.items()}

        pad_id = int(data.get("pad_id", token_to_id.get("<pad>", 0)))
        unk_id = int(data.get("unk_id", token_to_id.get("<unk>", 1)))
        bos_id = data.get("bos_id", token_to_id.get("<start>", None))
        eos_id = data.get("eos_id", token_to_id.get("<end>", None))

        bos_id = None if bos_id is None else int(bos_id)
        eos_id = None if eos_id is None else int(eos_id)

        return Vocab(
            tokens=tokens,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )


def build_word_vocab(
    sentences: Iterable[str],
    specials: List[str],
    min_freq: int = 1,
) -> Vocab:
    """
    Build word-level vocabulary from sentences.

    Args:
        sentences: Iterable of sentences (strings)
        specials: List of special tokens (e.g., ["<pad>", "<unk>", "<start>", "<end>"])
        min_freq: Minimum frequency for a token to be included

    Returns:
        Vocab object with token mappings
    """
    counts = Counter()
    for sentence in sentences:
        if not sentence:
            continue
        s = str(sentence).strip()
        if not s:
            continue
        counts.update(s.split())

    tokens = list(specials)

    for word, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if count >= min_freq and word not in specials:
            tokens.append(word)

    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    pad_id = token_to_id.get("<pad>", 0)
    unk_id = token_to_id.get("<unk>", 1)
    bos_id = token_to_id.get("<start>")
    eos_id = token_to_id.get("<end>")

    return Vocab(
        tokens=tokens,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        pad_id=pad_id,
        unk_id=unk_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )


def encode(sentence: str, vocab: Vocab, add_bos_eos: bool = False) -> List[int]:
    """
    Encode sentence to list of token IDs.
    """
    s = "" if sentence is None else str(sentence).strip()
    if not s:
        ids: List[int] = []
    else:
        words = s.split()
        ids = [vocab.token_to_id.get(word, vocab.unk_id) for word in words]

    if add_bos_eos:
        if vocab.bos_id is None or vocab.eos_id is None:
            raise ValueError("Vocab missing <start> or <end> tokens")
        ids = [vocab.bos_id] + ids + [vocab.eos_id]

    return ids


def decode(ids: List[int], vocab: Vocab, skip_special: bool = True) -> str:
    """
    Decode list of token IDs back to sentence.
    """
    tokens: List[str] = []
    for token_id in ids:
        token = vocab.id_to_token.get(int(token_id), "<unk>")

        if skip_special and token in {"<pad>", "<start>", "<end>"}:
            continue

        tokens.append(token)

    return " ".join(tokens)


def save_vocab(vocab: Vocab, path: str):
    """
    Save vocabulary to JSON file.
    """
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"Saved vocabulary ({len(vocab.tokens)} tokens) to {path}")


def load_vocab(path: str) -> Vocab:
    """
    Load vocabulary from JSON file.
    """
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab = Vocab.from_dict(data)
    print(f"Loaded vocabulary ({len(vocab.tokens)} tokens) from {path}")
    return vocab


if __name__ == "__main__":
    print("Testing vocabulary module...")

    sentences = [
        "HELLO WORLD",
        "HELLO THERE",
        "WORLD PEACE",
        "HELLO HELLO HELLO",
    ]

    vocab = build_word_vocab(
        sentences,
        specials=["<pad>", "<unk>", "<start>", "<end>"],
        min_freq=1,
    )

    print(vocab)
    test_sentence = "HELLO WORLD"
    ids = encode(test_sentence, vocab)
    ids_with = encode(test_sentence, vocab, add_bos_eos=True)

    print("Encoded:", ids)
    print("Encoded BOS/EOS:", ids_with)
    print("Decoded skip special:", decode(ids_with, vocab, skip_special=True))
    print("Decoded keep special:", decode(ids_with, vocab, skip_special=False))

    # to_dict/from_dict sanity
    d = vocab.to_dict()
    vocab2 = Vocab.from_dict(d)
    assert vocab.tokens == vocab2.tokens
    assert vocab.pad_id == vocab2.pad_id
    assert vocab.unk_id == vocab2.unk_id
    assert vocab.bos_id == vocab2.bos_id
    assert vocab.eos_id == vocab2.eos_id
    print("✓ to_dict/from_dict passed")

    # save/load sanity
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        save_vocab(vocab, tmp)
        vocab3 = load_vocab(tmp)
        assert vocab.tokens == vocab3.tokens
        print("✓ save/load passed")
    finally:
        os.unlink(tmp)

    print("✅ All tests passed!")
