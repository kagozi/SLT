"""
Beam search decoder for video-to-gloss model.

Significantly better than greedy decoding.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


class BeamSearchDecoder:
    """Beam search for sequence generation."""
    
    def __init__(
        self,
        model,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        beam_size: int = 5,
        max_len: int = 100,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.0,
    ):
        self.model = model
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.beam_size = beam_size
        self.max_len = max_len
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
    
    def _apply_repetition_penalty(
        self, 
        logits: torch.Tensor, 
        generated_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Reduce logits for tokens that have already been generated.
        
        Args:
            logits: [beam_size, vocab_size]
            generated_ids: [beam_size, seq_len]
        
        Returns:
            logits: [beam_size, vocab_size]
        """
        if self.repetition_penalty == 1.0:
            return logits
        
        for beam_idx in range(logits.size(0)):
            for token_id in generated_ids[beam_idx].unique():
                if token_id != self.pad_id:
                    logits[beam_idx, token_id] /= self.repetition_penalty
        
        return logits
    
    def _length_penalty_fn(self, length: int) -> float:
        """
        Length penalty from Google's NMT paper.
        
        lp = ((5 + length) / 6) ^ alpha
        """
        return ((5.0 + length) / 6.0) ** self.length_penalty
    
    @torch.no_grad()
    def decode(
        self,
        rgb: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> Tuple[List[int], float]:
        """
        Beam search for a single sample.
        
        Args:
            rgb: [1, T, 3, H, W]
            src_key_padding_mask: [1, T]
        
        Returns:
            best_sequence: List of token IDs
            best_score: Log probability score
        """
        device = rgb.device
        
        # Encode once
        memory = self.model.encoder(rgb, src_key_padding_mask=src_key_padding_mask)  # [1, T, D]
        memory = memory.expand(self.beam_size, -1, -1)  # [beam_size, T, D]
        
        if src_key_padding_mask is not None:
            src_mask = src_key_padding_mask.expand(self.beam_size, -1)
        else:
            src_mask = None
        
        # Initialize beams
        # Shape: [beam_size, seq_len]
        sequences = torch.full(
            (self.beam_size, 1), 
            self.bos_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Scores for each beam
        scores = torch.zeros(self.beam_size, device=device)
        
        # Track which beams are finished
        finished = torch.zeros(self.beam_size, dtype=torch.bool, device=device)
        
        for step in range(self.max_len - 1):
            # Get logits for current sequences
            # [beam_size, seq_len, vocab_size]
            logits = self.model.decoder(
                tgt=sequences,
                memory=memory,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=src_mask,
            )
            
            # Get logits for next token: [beam_size, vocab_size]
            next_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if self.repetition_penalty > 1.0:
                next_logits = self._apply_repetition_penalty(next_logits, sequences)
            
            # Convert to log probabilities
            log_probs = F.log_softmax(next_logits, dim=-1)  # [beam_size, vocab_size]
            
            # For finished beams, force to select EOS (or pad)
            log_probs[finished] = float('-inf')
            log_probs[finished, self.eos_id] = 0.0
            
            # Expand scores: [beam_size, vocab_size]
            # For each beam, compute score for each possible next token
            candidate_scores = scores.unsqueeze(1) + log_probs  # [beam_size, vocab_size]
            
            # Flatten to [beam_size * vocab_size]
            candidate_scores = candidate_scores.view(-1)
            
            # Get top beam_size candidates
            top_scores, top_indices = torch.topk(
                candidate_scores, 
                k=self.beam_size, 
                sorted=True
            )
            
            # Convert flat indices back to (beam_idx, token_idx)
            beam_indices = top_indices // log_probs.size(1)
            token_indices = top_indices % log_probs.size(1)
            
            # Update sequences
            new_sequences = sequences[beam_indices]  # [beam_size, seq_len]
            new_sequences = torch.cat(
                [new_sequences, token_indices.unsqueeze(1)], 
                dim=1
            )  # [beam_size, seq_len+1]
            
            sequences = new_sequences
            scores = top_scores
            
            # Update finished beams
            finished = finished[beam_indices]
            finished |= (token_indices == self.eos_id)
            
            # If all beams finished, stop
            if finished.all():
                break
        
        # Apply length penalty to scores
        lengths = (sequences != self.pad_id).sum(dim=1)
        normalized_scores = scores / self._length_penalty_fn(lengths.float())
        
        # Get best sequence
        best_idx = normalized_scores.argmax()
        best_sequence = sequences[best_idx].tolist()
        best_score = scores[best_idx].item()
        
        # Remove BOS
        if best_sequence[0] == self.bos_id:
            best_sequence = best_sequence[1:]
        
        # Remove EOS and everything after
        if self.eos_id in best_sequence:
            eos_idx = best_sequence.index(self.eos_id)
            best_sequence = best_sequence[:eos_idx]
        
        return best_sequence, best_score
    
    @torch.no_grad()
    def decode_batch(
        self,
        rgb: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> List[Tuple[List[int], float]]:
        """
        Beam search for a batch of samples.
        
        Args:
            rgb: [B, T, 3, H, W]
            src_key_padding_mask: [B, T]
        
        Returns:
            results: List of (sequence, score) for each sample in batch
        """
        batch_size = rgb.size(0)
        results = []
        
        for i in range(batch_size):
            rgb_i = rgb[i:i+1]  # [1, T, 3, H, W]
            mask_i = src_key_padding_mask[i:i+1] if src_key_padding_mask is not None else None
            
            seq, score = self.decode(rgb_i, mask_i)
            results.append((seq, score))
        
        return results


def add_beam_search_to_model(model, vocab):
    """
    Add beam search method to VideoToGlossModel.
    
    Usage:
        model = VideoToGlossModel(...)
        add_beam_search_to_model(model, vocab)
        
        # Now you can call:
        sequences = model.beam_search(rgb, beam_size=5)
    """
    
    def beam_search(
        self,
        rgb: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
        beam_size: int = 5,
        max_len: int = 100,
        length_penalty: float = 0.6,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """
        Beam search decoding.
        
        Returns:
            sequences: [B, max_len] tensor of token IDs (padded)
        """
        self.eval()
        
        decoder = BeamSearchDecoder(
            model=self,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            pad_id=vocab.pad_id,
            beam_size=beam_size,
            max_len=max_len,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        
        results = decoder.decode_batch(rgb, src_key_padding_mask)
        
        # Convert to padded tensor
        sequences = []
        for seq, _ in results:
            sequences.append(torch.tensor(seq, dtype=torch.long, device=rgb.device))
        
        # Pad
        from torch.nn.utils.rnn import pad_sequence
        padded = pad_sequence(sequences, batch_first=True, padding_value=vocab.pad_id)
        
        return padded
    
    # Attach method to model
    import types
    model.beam_search = types.MethodType(beam_search, model)
    
    return model


if __name__ == "__main__":
    # Example usage
    from models.video_to_gloss import VideoToGlossModel
    from data.vocab import Vocab
    
    # Create dummy vocab
    vocab_dict = {
        "tokens": ["<pad>", "<unk>", "<start>", "<end>", "HELLO", "WORLD"],
        "pad_id": 0,
        "unk_id": 1,
        "bos_id": 2,
        "eos_id": 3,
    }
    vocab = Vocab.from_dict(vocab_dict)
    
    # Create model
    model = VideoToGlossModel(
        gloss_vocab_size=len(vocab.tokens),
        pad_id=vocab.pad_id,
        d_model=128,
    )
    model.eval()
    
    # Add beam search
    add_beam_search_to_model(model, vocab)
    
    # Test
    rgb = torch.randn(2, 10, 3, 256, 256)  # [B=2, T=10, C, H, W]
    
    # Greedy
    greedy_output = model.generate(rgb, bos_id=vocab.bos_id or 2, eos_id=vocab.eos_id or 3)
    
    print("Greedy output shape:", greedy_output.shape)
    
    # Beam search
    beam_output = model.beam_search(rgb, beam_size=5)
    print("Beam search output shape:", beam_output.shape)
    
    print("\nBeam search decoding works!")