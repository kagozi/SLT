# aslg_translation/evaluation/metrics.py

from __future__ import annotations

from typing import Dict, List

import sacrebleu


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def corpus_bleu(preds: List[str], refs: List[str]) -> float:
    bleu = sacrebleu.corpus_bleu(preds, [refs])
    return float(bleu.score)


def corpus_bleu_ngram(preds: List[str], refs: List[str], n: int) -> float:
    """
    Returns BLEU-n (%), using SacreBLEU with only up-to-n n-gram order.
    """
    bleu = sacrebleu.metrics.BLEU(effective_order=True, max_ngram_order=int(n))
    return float(bleu.corpus_score(preds, [refs]).score)


def corpus_chrf(preds: List[str], refs: List[str]) -> float:
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    return float(chrf.score)


_METEOR_WARNED = False


def corpus_meteor(preds: List[str], refs: List[str]) -> float:
    """
    Corpus METEOR as mean of sentence METEOR.
    Requires nltk + wordnet resources installed on the machine.

    We DO NOT download resources at runtime (SLURM-friendly).
    """
    global _METEOR_WARNED
    try:
        from nltk.translate.meteor_score import meteor_score
        scores = []
        for hyp, ref in zip(preds, refs):
            scores.append(meteor_score([ref.split()], hyp.split()))
        return float(sum(scores) / max(1, len(scores)))
    except Exception as e:
        if not _METEOR_WARNED:
            _METEOR_WARNED = True
            print(
                "[metrics] METEOR unavailable (missing nltk data/resources). "
                "Install nltk + wordnet in the environment, or pre-download corpora. "
                f"Error: {type(e).__name__}: {e}"
            )
        return 0.0


_WER_WARNED = False


def word_error_rate(preds: List[str], refs: List[str]) -> float:
    """
    WER using jiwer.
    """
    global _WER_WARNED
    try:
        from jiwer import wer
        return float(wer(refs, preds))
    except Exception as e:
        if not _WER_WARNED:
            _WER_WARNED = True
            print(
                "[metrics] WER unavailable (missing jiwer). "
                "Install jiwer in the environment. "
                f"Error: {type(e).__name__}: {e}"
            )
        return 0.0


def exact_match(preds: List[str], refs: List[str]) -> float:
    correct = sum(int(p.strip() == r.strip()) for p, r in zip(preds, refs))
    return _safe_div(correct, len(refs))


def compute_all_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    return {
        "bleu1": corpus_bleu_ngram(preds, refs, 1),
        "bleu2": corpus_bleu_ngram(preds, refs, 2),
        "bleu3": corpus_bleu_ngram(preds, refs, 3),
        "bleu4": corpus_bleu_ngram(preds, refs, 4),
        "meteor": corpus_meteor(preds, refs),
        "chrf": corpus_chrf(preds, refs),
        "wer": word_error_rate(preds, refs),
        "exact_match": exact_match(preds, refs),
    }
