"""
Evaluation metrics for Sign Language Translation.
WER (Word Error Rate) for gloss recognition.
BLEU for translation quality.
"""

import collections
import math
import numpy as np
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────
# WER (from reference, with configurable costs)
# ─────────────────────────────────────────────

def compute_wer(
    references: List[str],
    hypotheses: List[str],
    cost_del: int = 3,
    cost_ins: int = 3,
    cost_sub: int = 4,
) -> float:
    """
    Compute average Word Error Rate between reference and hypothesis sequences.
    Uses edit distance with configurable costs (from reference code).
    """
    total_wer = 0.0
    for ref_str, hyp_str in zip(references, hypotheses):
        ref = ref_str.strip().split()
        hyp = hyp_str.strip().split()

        if len(ref) == 0:
            total_wer += 1.0
            continue

        d = np.zeros((len(ref) + 1, len(hyp) + 1))
        for i in range(len(ref) + 1):
            d[i][0] = i * cost_del
        for j in range(len(hyp) + 1):
            d[0][j] = j * cost_ins

        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                sub_cost = d[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else cost_sub)
                ins_cost = d[i][j - 1] + cost_ins
                del_cost = d[i - 1][j] + cost_del
                d[i][j] = min(sub_cost, ins_cost, del_cost)

        max_cost = max(cost_del, cost_ins, cost_sub)
        wer = d[len(ref)][len(hyp)] / (max(len(ref), len(hyp)) * max_cost)
        total_wer += min(1.0, wer)

    return total_wer / max(1, len(references))


# ─────────────────────────────────────────────
# BLEU (from reference compute_bleu, unchanged logic)
# ─────────────────────────────────────────────

def _get_ngrams(segment: List[str], max_order: int) -> collections.Counter:
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(
    references: List[str],
    hypotheses: List[str],
    max_order: int = 4,
    smooth: bool = True,
) -> Dict[str, float]:
    """
    Compute BLEU-1 through BLEU-{max_order}.
    Returns dict: {"bleu-1": ..., "bleu-2": ..., "bleu-3": ..., "bleu-4": ..., "bleu": <bleu-4>}
    """
    ref_corpus = [[r.strip().split()] for r in references]
    hyp_corpus = [h.strip().split() for h in hypotheses]

    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    ref_length = 0
    hyp_length = 0

    for refs, hyp in zip(ref_corpus, hyp_corpus):
        ref_length += min(len(r) for r in refs)
        hyp_length += len(hyp)

        merged_ref = collections.Counter()
        for ref in refs:
            merged_ref |= _get_ngrams(ref, max_order)
        hyp_ngrams = _get_ngrams(hyp, max_order)
        overlap = hyp_ngrams & merged_ref

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible = len(hyp) - order + 1
            if possible > 0:
                possible_by_order[order - 1] += possible

    results = {}
    for n in range(1, max_order + 1):
        precisions = [0.0] * n
        for i in range(n):
            if smooth:
                precisions[i] = (matches_by_order[i] + 1.0) / (possible_by_order[i] + 1.0)
            else:
                precisions[i] = (float(matches_by_order[i]) / possible_by_order[i]) if possible_by_order[i] > 0 else 0.0

        if min(precisions) > 0:
            log_avg = sum(math.log(p) for p in precisions) / n
            geo_mean = math.exp(log_avg)
        else:
            geo_mean = 0.0

        ratio = float(hyp_length) / max(1, ref_length)
        bp = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / max(ratio, 1e-10))

        results[f"bleu-{n}"] = geo_mean * bp

    results["bleu"] = results[f"bleu-{max_order}"]
    return results


def evaluate_model(
    gloss_refs: List[str],
    gloss_hyps: List[str],
    trans_refs: List[str],
    trans_hyps: List[str],
) -> Dict[str, float]:
    """Run full evaluation: WER on glosses + BLEU on translations."""
    metrics = {}
    if gloss_refs and gloss_hyps:
        metrics["wer"] = compute_wer(gloss_refs, gloss_hyps)
    if trans_refs and trans_hyps:
        bleu = compute_bleu(trans_refs, trans_hyps)
        metrics.update(bleu)
    return metrics