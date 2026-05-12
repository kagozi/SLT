from typing import Optional
import re
import torch


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "he", "her", "hers", "him", "his", "i",
    "if", "in", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "ours", "she", "so", "that", "the", "their", "theirs", "them", "then",
    "there", "these", "they", "this", "those", "to", "was", "we", "were",
    "what", "when", "where", "which", "who", "will", "with", "you", "your",
    "yours",
    # High-frequency instructional/caption filler in How2Sign/YouTube-ASL.
    "actually", "again", "also", "basically", "can", "could", "gonna", "going",
    "got", "guess", "just", "kind", "let", "let's", "like", "little", "look",
    "make", "maybe", "mean", "need", "now", "okay", "really", "right", "say",
    "see", "sort", "thing", "things", "think", "try", "use", "want", "way",
    "we're", "i'm", "you're", "that's", "it's", "don't", "doesn't", "can't",
    "i'll", "we'll", "you'll",
}


def normalize_translation_text(text: str) -> str:
    """Normalize spoken-language text for word-level CTC targets."""
    text = str(text or "").lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.upper()


def translation_keywords(text: str, min_tokens: int = 1) -> str:
    """Lightweight content-word target for unglossed CTC training."""
    normalized = normalize_translation_text(text)
    tokens = [
        tok for tok in normalized.split()
        if tok.lower() not in _STOPWORDS and not tok.isdigit()
    ]
    if len(tokens) < min_tokens:
        tokens = normalized.split()
    return " ".join(tokens)


def resolve_translation_config(dataset: str, requested_model: str = "auto"):
    """Pick a stronger dataset-aware seq2seq decoder default."""
    dataset = dataset.lower()
    if requested_model and requested_model != "auto":
        return {"bart_model": requested_model, "target_lang": None}

    if dataset == "phoenix":
        return {
            "bart_model": "facebook/mbart-large-50-many-to-many-mmt",
            "target_lang": "de_DE",
        }

    return {
        "bart_model": "facebook/bart-base",
        "target_lang": None,
    }


def configure_tokenizer_for_target(tokenizer, target_lang: Optional[str]):
    """Set target-language metadata for multilingual tokenizers when supported."""
    forced_bos_token_id = None
    if target_lang and hasattr(tokenizer, "tgt_lang"):
        tokenizer.tgt_lang = target_lang
    if target_lang and hasattr(tokenizer, "src_lang") and getattr(tokenizer, "src_lang", None) is None:
        tokenizer.src_lang = target_lang
    if target_lang and hasattr(tokenizer, "lang_code_to_id"):
        forced_bos_token_id = tokenizer.lang_code_to_id.get(target_lang)
    return forced_bos_token_id


def encode_translation_target(tokenizer, text: str, max_length: int = 128) -> torch.Tensor:
    """Encode translation labels, using text_target for seq2seq tokenizers when available."""
    kwargs = dict(max_length=max_length, truncation=True, padding=False, return_tensors='pt')
    try:
        encoded = tokenizer(text_target=text, **kwargs)
    except TypeError:
        encoded = tokenizer(text, **kwargs)
    return encoded['input_ids'].squeeze(0)
