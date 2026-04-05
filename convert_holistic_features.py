"""
Extract PSewmuthu/How2Sign_Holistic (HuggingFace) raw keypoints.

Saves the raw (T, 543, 3) MediaPipe Holistic arrays directly — no
landmark selection or normalization.  All processing (selection,
wrist-relative coords, neck normalization) happens at data-load time
in dataset_how2sign.py so it can be iterated without re-extraction.

Correct MediaPipe Holistic layout (543 landmarks):
    Pose        :   0 –  32  (33 landmarks)
    Face mesh   :  33 – 500  (468 landmarks,  face mesh idx i = holistic 33+i)
    Left hand   : 501 – 521  (21 landmarks)
    Right hand  : 522 – 542  (21 landmarks)

Output layout:
    /data/how2sign_hf/
        annotations/how2sign_{train,val,test}.csv
        raw/{train,val,test}/{SENTENCE_NAME}.npy     <- (T, 543, 3) float32

Usage:
    python convert_holistic_features.py \\
        --hf_cache /data/hf_cache/How2Sign_Holistic \\
        --out_dir  /data/how2sign_hf
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# ── PSEUDOGLOSS ──────────────────────────────────────────────────────────────

def make_pseudogloss_fn():
    import spacy
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError:
        import sys
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'],
                       check=True)
        nlp = spacy.load('en_core_web_sm')

    POS_KEEP = {'NOUN', 'VERB', 'ADJ', 'ADV'}

    def pseudogloss(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ''
        doc = nlp(text)
        tokens = [t.text.upper() for t in doc
                  if t.pos_ in POS_KEEP and not t.is_stop and t.is_alpha]
        return ' '.join(tokens) if tokens else text.upper()

    return pseudogloss


# ── Annotations ──────────────────────────────────────────────────────────────

def convert_annotations(meta_dir: Path, anno_out: Path, pseudogloss_fn) -> dict:
    anno_out.mkdir(parents=True, exist_ok=True)
    dfs = {}
    for split in ('train', 'val', 'test'):
        src = meta_dir / f'how2sign_realigned_{split}.csv'
        if not src.exists():
            src = meta_dir / f'how2sign_{split}.csv'
        if not src.exists():
            print(f'  WARNING: no CSV for {split}')
            continue

        df = pd.read_csv(src, sep='\t')
        df.columns = [c.strip() for c in df.columns]

        if 'PSEUDOGLOSS' not in df.columns:
            print(f'  [{split}] generating PSEUDOGLOSS ...')
            df['PSEUDOGLOSS'] = df['SENTENCE'].apply(pseudogloss_fn)

        out = anno_out / f'how2sign_{split}.csv'
        df.to_csv(out, sep='\t', index=False)
        print(f'  [{split}] {len(df)} rows → {out}')
        dfs[split] = df

    return dfs


# ── Raw extraction ────────────────────────────────────────────────────────────

def extract_split(split: str, df: pd.DataFrame,
                  rar_path: Path, raw_out: Path,
                  extract_tmp: Path) -> None:
    """
    1. Extract frontal.rar -> extract_tmp/{split}/
    2. Save each (T, 543, 3) array as float32 to raw_out/{SENTENCE_NAME}.npy
    3. Delete extract_tmp/{split}/
    """
    raw_out.mkdir(parents=True, exist_ok=True)

    if not rar_path.exists():
        print(f'  [{split}] WARNING: RAR not found: {rar_path}')
        return

    needed = {}
    for _, row in df.iterrows():
        name = str(row['SENTENCE_NAME']).strip()
        out_path = raw_out / f'{name}.npy'
        if not out_path.exists():
            needed[name] = out_path

    already_done = len(df) - len(needed)
    print(f'  [{split}] {len(df)} total, {already_done} already done, '
          f'{len(needed)} to extract')

    if not needed:
        print(f'  [{split}] nothing to do.')
        return

    extract_dir = extract_tmp / split
    extract_dir.mkdir(parents=True, exist_ok=True)
    rar_gb = rar_path.stat().st_size / 1e9
    print(f'  [{split}] extracting {rar_path.name} ({rar_gb:.1f} GB) -> {extract_dir} ...',
          flush=True)
    subprocess.run(
        ['unrar', 'x', '-y', str(rar_path), str(extract_dir) + '/'],
        check=True
    )
    print(f'  [{split}] extraction complete.', flush=True)

    npy_lookup = {p.stem.replace('_holistic', ''): p
                  for p in extract_dir.rglob('*_holistic.npy')}
    print(f'  [{split}] {len(npy_lookup)} npy files found')

    saved = skipped = errors = 0
    for name, out_path in tqdm(needed.items(), desc=f'  {split}'):
        src_path = npy_lookup.get(name)
        if src_path is None:
            tqdm.write(f'    WARNING: {name} not found')
            skipped += 1
            continue

        try:
            raw = np.load(src_path)
            if raw.ndim == 3 and raw.shape[1] == 543 and raw.shape[2] == 3:
                np.save(out_path, raw.astype(np.float32))
                saved += 1
            else:
                tqdm.write(f'    WARNING: unexpected shape {raw.shape} for {name}')
                errors += 1
        except Exception as e:
            tqdm.write(f'    ERROR {name}: {e}')
            errors += 1

    print(f'  [{split}] saved={saved}, skipped={skipped}, errors={errors}')
    print(f'  [{split}] cleaning up {extract_dir} ...', flush=True)
    shutil.rmtree(extract_dir)
    print(f'  [{split}] done.', flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract How2Sign_Holistic raw (T, 543, 3) keypoints.'
    )
    parser.add_argument('--hf_cache',    type=str,
                        default='/data/hf_cache/How2Sign_Holistic')
    parser.add_argument('--out_dir',     type=str, default='/data/how2sign_hf')
    parser.add_argument('--extract_tmp', type=str, default='/data/how2sign_hf_extract')
    parser.add_argument('--splits',      nargs='+', default=['train', 'val', 'test'])
    args = parser.parse_args()

    hf_root     = Path(args.hf_cache) / 'how2sign_holistic_features'
    out_dir     = Path(args.out_dir)
    extract_tmp = Path(args.extract_tmp)
    meta_dir    = hf_root / 'metadata'

    print(f'HF root     : {hf_root}')
    print(f'Out dir     : {out_dir}')
    print(f'Extract tmp : {extract_tmp}')

    print('\n=== Step 1: Annotations ===')
    pseudogloss_fn = make_pseudogloss_fn()
    dfs = convert_annotations(meta_dir, out_dir / 'annotations', pseudogloss_fn)

    print('\n=== Step 2: Extract raw (T, 543, 3) keypoints ===')
    for split in args.splits:
        if split not in dfs:
            print(f'  [{split}] no annotation DataFrame, skipping.')
            continue
        rar_path = hf_root / split / 'frontal.rar'
        raw_out  = out_dir / 'raw' / split
        extract_split(split, dfs[split], rar_path, raw_out, extract_tmp)

    if extract_tmp.exists():
        shutil.rmtree(extract_tmp, ignore_errors=True)

    print('\n Done.')
    print(f'   Annotations  : {out_dir}/annotations/')
    print(f'   Raw keypoints: {out_dir}/raw/{{train,val,test}}/')


if __name__ == '__main__':
    main()
