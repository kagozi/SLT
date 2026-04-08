from typing import Optional
import torch


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
