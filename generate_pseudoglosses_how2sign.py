"""
Option A: POS-based pseudogloss generation for How2Sign (Sign2GPT approach).

For each English sentence, extract content words (NOUN, VERB, ADJ, ADV)
using spaCy lemmatisation and POS filtering. These serve as a weak
gloss-like intermediate supervision signal for CTC training.

Writes a PSEUDOGLOSS column directly into each metadata CSV, so the
existing How2SignDataset loader picks it up automatically.

Usage (local):
    pip install spacy && python -m spacy download en_core_web_sm
    python generate_pseudoglosses_how2sign.py --root_dir /data/how2sign

Usage (NRP job):
    See nautilius/generate-pseudoglosses-job.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import spacy
from tqdm import tqdm

# POS tags treated as sign-relevant content words
CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV'}


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
    csv_path = root_dir / 'metadata' / f'how2sign_realigned_{split}.csv'
    if not csv_path.exists():
        print(f"  [{split}] CSV not found at {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path, sep='\t')

    if 'PSEUDOGLOSS' in df.columns:
        already = df['PSEUDOGLOSS'].notna().sum()
        print(f"  [{split}] PSEUDOGLOSS column already exists ({already}/{len(df)} filled). Re-generating...")

    sentences = df['SENTENCE'].fillna('').astype(str).tolist()

    print(f"  [{split}] Processing {len(sentences)} sentences with spaCy...")
    docs = list(tqdm(nlp.pipe(sentences, batch_size=256), total=len(sentences), desc=f"  POS [{split}]"))
    pseudoglosses = [extract_pseudogloss(doc) for doc in docs]

    df['PSEUDOGLOSS'] = pseudoglosses
    df.to_csv(csv_path, sep='\t', index=False)

    # Summary statistics
    n_unknown = sum(1 for p in pseudoglosses if p == 'UNKNOWN')
    avg_len = sum(len(p.split()) for p in pseudoglosses) / max(1, len(pseudoglosses))
    print(f"  [{split}] Done. avg pseudogloss length: {avg_len:.1f} tokens, "
          f"{n_unknown}/{len(df)} UNKNOWN sentences.")

    print(f"\n  Sample pseudoglosses [{split}]:")
    for i in range(min(5, len(df))):
        print(f"    Sentence:    {sentences[i][:80]}")
        print(f"    Pseudogloss: {pseudoglosses[i]}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Generate POS-based pseudoglosses for How2Sign')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='Path to How2Sign root (contains metadata/ subdirectory)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    print("Loading spaCy en_core_web_sm...")
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    print("spaCy loaded.\n")

    for split in args.splits:
        process_split(root_dir, split, nlp)

    print("\nAll splits done. PSEUDOGLOSS column written to metadata CSVs.")
    print("How2SignDataset will now load pseudoglosses automatically.")


if __name__ == '__main__':
    main()
