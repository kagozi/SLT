"""
Tokenizers and decoding utilities for Sign Language Translation.
"""

import json
from typing import List, Dict, Optional
from pathlib import Path
from collections import Counter
import torch


class GlossTokenizer:
    """
    Simple whitespace-based gloss tokenizer with vocabulary.
    Maps gloss strings like "ICH OSTERN WETTER" to integer sequences.
    Token 0 = <blank> (CTC blank), Token 1 = <unk>.
    """

    def __init__(self):
        self.word2idx: Dict[str, int] = {"<blank>": 0, "<unk>": 1}
        self.idx2word: Dict[int, str] = {0: "<blank>", 1: "<unk>"}
        self._next_idx = 2

    def build_vocab(self, gloss_strings: List[str], min_freq: int = 1):
        """Build vocabulary from a list of gloss strings."""
        counter = Counter()
        for s in gloss_strings:
            tokens = s.strip().upper().split()
            counter.update(tokens)

        for word, freq in counter.most_common():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self._next_idx
                self.idx2word[self._next_idx] = word
                self._next_idx += 1

        print(f"GlossTokenizer: {len(self.word2idx)} tokens (min_freq={min_freq})")

    def encode(self, text: str) -> List[int]:
        """Encode a gloss string to token ids."""
        tokens = text.strip().upper().split()
        return [self.word2idx.get(t, 1) for t in tokens]

    def decode(self, ids: List[int]) -> str:
        """Decode token ids back to gloss string."""
        words = [self.idx2word.get(i, "<unk>") for i in ids if i > 0]
        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx}, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.word2idx = data["word2idx"]
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self._next_idx = max(self.idx2word.keys()) + 1


def ctc_greedy_decode(logits: torch.Tensor, blank: int = 0) -> List[List[int]]:
    """
    Greedy CTC decoding: argmax → collapse consecutive duplicates → remove blanks.
    
    Args:
        logits: (B, T, V) log-probabilities
        blank: blank token index
    
    Returns:
        List of decoded token id sequences (one per batch item)
    """
    preds = logits.argmax(dim=-1)  # (B, T)
    decoded = []

    for seq in preds:
        result = []
        prev = -1
        for token in seq.tolist():
            if token != prev:
                if token != blank:
                    result.append(token)
                prev = token
        decoded.append(result)

    return decoded


def ctc_beam_decode(
    logits: torch.Tensor,
    beam_width: int = 10,
    blank: int = 0,
) -> List[List[int]]:
    """
    Simple prefix beam search CTC decoding.
    For production, consider using torchaudio.models.decoder.ctc_decoder.
    
    Falls back to greedy if beam_width <= 1.
    """
    if beam_width <= 1:
        return ctc_greedy_decode(logits, blank)

    try:
        # Try using torchaudio's CTC decoder if available
        import torchaudio
        from torchaudio.models.decoder import ctc_decoder

        # This requires compilation of kenlm, so fallback to greedy
        raise ImportError("Using simple beam search")
    except ImportError:
        pass

    # Simple beam search implementation
    B, T, V = logits.shape
    log_probs = logits  # already log-softmax'd from CTCGlossHead

    decoded = []
    for b in range(B):
        # beam: list of (prefix_tuple, log_prob)
        beams = [((), 0.0)]

        for t in range(T):
            new_beams = {}
            lp = log_probs[b, t]  # (V,)

            for prefix, score in beams:
                for c in lp.topk(beam_width).indices.tolist():
                    new_score = score + lp[c].item()
                    if c == blank:
                        key = prefix
                    elif len(prefix) > 0 and prefix[-1] == c:
                        key = prefix  # collapse duplicate
                    else:
                        key = prefix + (c,)

                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            # Prune to beam_width
            beams = sorted(new_beams.items(), key=lambda x: -x[1])[:beam_width]

        best_prefix = beams[0][0] if beams else ()
        decoded.append(list(best_prefix))

    return decoded