"""
POS-based pseudogloss generation for How2Sign (Sign2GPT approach).

For each English sentence, extract content words (NOUN, VERB, ADJ, ADV)
using spaCy lemmatisation and POS filtering. These serve as a weak
gloss-like intermediate supervision signal for CTC training.

Reads:   <root_dir>/metadata/how2sign_realigned_{split}.csv
Writes:  PSEUDOGLOSS column back into the same CSV in-place.
         How2SignDataset picks it up automatically on the next training run.

Logs 200 random samples per split to a W&B Table for manual review.

Usage (local):
    python generate_pseudoglosses_how2sign.py \
        --root_dir /data/hf_cache/How2Sign_Holistic/how2sign_holistic_features

Usage (NRP job):
    kubectl apply -f nautilius/generate-pseudoglosses-job.yaml
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import spacy
import wandb
from tqdm import tqdm

# POS tags treated as sign-relevant content words
CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}
WANDB_SAMPLES = 200   # rows logged per split


def extract_pseudogloss(doc) -> str:
    """
    Extract content words from a spaCy Doc and return as an uppercase
    space-separated pseudogloss string.

    Rules (Sign2GPT-style):
      - Keep NOUN, VERB, ADJ, ADV tokens only
      - Exclude stopwords (e.g. 'be', 'have', 'do' as auxiliaries)
      - Exclude non-alpha tokens (punctuation, numbers)
      - Lemmatise so inflected forms collapse to the same pseudogloss
        (e.g. CUTTING -> CUT, GLASSES -> GLASS)
    """
    tokens = [
        tok.lemma_.upper()
        for tok in doc
        if tok.pos_ in CONTENT_POS
        and not tok.is_stop
        and tok.is_alpha
    ]
    return ' '.join(tokens) if tokens else 'UNKNOWN'


def process_split(root_dir: Path, split: str, nlp) -> None:
    csv_path = root_dir / 'annotations' / f'how2sign_{split}.csv'
    if not csv_path.exists():
        print(f"  [{split}] CSV not found: {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path, sep='\t')

    if 'PSEUDOGLOSS' in df.columns:
        already = df['PSEUDOGLOSS'].notna().sum()
        print(f"  [{split}] PSEUDOGLOSS column already exists "
              f"({already}/{len(df)} filled). Re-generating...")

    sentences = df['SENTENCE'].fillna('').astype(str).tolist()

    print(f"  [{split}] Processing {len(sentences)} sentences with spaCy...")
    docs = list(tqdm(nlp.pipe(sentences, batch_size=256),
                     total=len(sentences), desc=f"  POS [{split}]"))
    pseudoglosses = [extract_pseudogloss(doc) for doc in docs]

    df['PSEUDOGLOSS'] = pseudoglosses
    df.to_csv(csv_path, sep='\t', index=False)

    # ── Summary stats ────────────────────────────────────────────────────────
    n_unknown = sum(1 for p in pseudoglosses if p == 'UNKNOWN')
    lengths   = [len(p.split()) for p in pseudoglosses]
    avg_len   = sum(lengths) / max(1, len(lengths))
    print(f"  [{split}] Done. avg length: {avg_len:.1f} tokens, "
          f"UNKNOWN: {n_unknown}/{len(df)}")

    # ── Console sample ───────────────────────────────────────────────────────
    print(f"\n  Sample pseudoglosses [{split}]:")
    for i in range(min(5, len(df))):
        print(f"    Sentence:    {sentences[i][:80]}")
        print(f"    Pseudogloss: {pseudoglosses[i]}")
        print()

    # ── W&B Table ────────────────────────────────────────────────────────────
    sentence_names = df['SENTENCE_NAME'].astype(str).tolist()
    n_sample = min(WANDB_SAMPLES, len(sentences))
    indices  = random.sample(range(len(sentences)), n_sample)

    table = wandb.Table(columns=["split", "sentence_name", "sentence", "pseudogloss",
                                 "gloss_length"])
    for i in indices:
        table.add_data(split, sentence_names[i], sentences[i],
                       pseudoglosses[i], lengths[i])

    wandb.log({
        f"pseudoglosses/{split}": table,
        f"pseudoglosses/{split}_avg_length": avg_len,
        f"pseudoglosses/{split}_unknown_pct": 100.0 * n_unknown / max(1, len(df)),
    })
    print(f"  [{split}] Logged {n_sample} samples to W&B.")


def main():
    parser = argparse.ArgumentParser(
        description='Generate POS-based pseudoglosses for How2Sign'
    )
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to HF holistic features root '
                             '(contains metadata/how2sign_realigned_{split}.csv)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    wandb.init(
        project="slt",
        name="pseudogloss-generation-how2sign",
        config={
            "root_dir":     str(root_dir),
            "splits":       args.splits,
            "content_pos":  sorted(CONTENT_POS),
            "wandb_samples": WANDB_SAMPLES,
        },
    )

    print("Loading spaCy en_core_web_sm...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    print("spaCy loaded.\n")

    for split in args.splits:
        process_split(root_dir, split, nlp)

    wandb.finish()
    print("\nAll splits done. PSEUDOGLOSS column written to annotations CSVs.")
    print("How2SignDataset will load pseudoglosses automatically on next training run.")


if __name__ == '__main__':
    main()
