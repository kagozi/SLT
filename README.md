# Sign Language Translation (SLT)

End-to-end sign language recognition and translation using body keypoints extracted with MediaPipe. Trains on PHOENIX-2014-T (German Sign Language) and How2Sign (American Sign Language) datasets.

**Thesis project — University of South Dakota**

---

## Table of Contents

- [Architecture](#architecture)
- [Experiments](#experiments)
- [Transfer Learning](#transfer-learning)
- [Pseudogloss Methods](#pseudogloss-methods)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [NRP / Nautilus Cluster](#nrp--nautilus-cluster)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [CI/CD](#cicd)

---

## Architecture

```
Video Frames
     │
     ▼
MediaPipe Keypoint Extractor
     │  75 joints × 3 coords = 225-dim feature vector per frame
     │  (13 pose + 21 left hand + 21 right hand + 20 face)
     ▼
Input Projection  (225 → dim)
     │
Positional Encoding + BatchNorm
     │
Conv Encoder (3 × ConvBlock)
     │  Depthwise conv with ECA channel attention
     │  Kernel sizes: 11 → 5 → 3
     ▼
Transformer Encoder (4 × TransformerBlock)
     │  Multi-head self-attention + FFN
     ▼
     ├──► CTC Gloss Head  →  WER / BLEU gloss metrics
     │
     └──► BART Translation Head (optional)
               │  Projects encoder states → BART d_model
               ▼
          BART Decoder  →  German / English text
               │
               ▼
          BLEU-1/4 translation metrics
```

**Three training modes** (controlled by CLI flags):

| Mode | `--use_bart` | `--ctc_weight` | Description |
|---|---|---|---|
| Sign2Gloss | off | — | CTC only, predicts gloss sequence |
| Sign2Gloss2Text | on | 0.1–0.9 | Joint CTC + BART, gloss-supervised translation |
| Glossless Sign2Text | on | 0.0 | BART only, no gloss labels needed |

---

## Experiments

### Base Experiments (Exps 1–6)

| # | Dataset | Mode | Pseudoglosses | Key Flags |
|---|---|---|---|---|
| 1 | PHOENIX-2014-T | Sign2Gloss | Ground-truth glosses | *(default)* |
| 2 | PHOENIX-2014-T | Sign2Gloss2Text | Ground-truth glosses | `--use_bart --ctc_weight 0.5 --freeze_bart_epochs 20` |
| 3 | PHOENIX-2014-T | Glossless Sign2Text | None | `--use_bart --ctc_weight 0.0` |
| 4 | How2Sign | Glossless Sign2Text | None (baseline) | `--dataset how2sign --use_bart --ctc_weight 0.0` |
| 5 | How2Sign | Sign2Gloss2Text | POS pseudo-glosses | `--dataset how2sign --use_bart --ctc_weight 0.5` |
| 6 | How2Sign | Sign2Gloss | POS pseudo-glosses | `--dataset how2sign` |

### Transfer Learning Experiments (Exps 7–14)

See [TRANSFER_LEARNING.md](TRANSFER_LEARNING.md) for full details.

| # | Dataset | Pretrained From | Freeze Strategy | Notes |
|---|---|---|---|---|
| 7 | How2Sign | Exp 1 (PHOENIX) | Full encoder frozen | CTC head only trained |
| 8 | How2Sign | Exp 1 (PHOENIX) | ConvBlocks frozen | Fine-tune Transformer + CTC |
| 9 | How2Sign | Exp 1 (PHOENIX) | None (full fine-tune) | Low LR (5e-5) |
| 10 | How2Sign | — | None | Scratch control baseline |
| 11 | PHOENIX + How2Sign | — | — | Joint training, language-aware encoder |
| 12 | How2Sign (10% data) | Exp 1 (PHOENIX) | None | Low-data adaptation |
| 13 | How2Sign (5% data) | Exp 1 (PHOENIX) | None | Low-data adaptation |
| 14 | How2Sign (1% data) | Exp 1 (PHOENIX) | None | Extreme low-resource |

---

## Transfer Learning

**Research Question:** Does pretraining on PHOENIX-2014-T (German Sign Language / DGS) improve generalisation on How2Sign (American Sign Language / ASL), and which representations transfer across typologically distinct sign languages?

Both datasets use identical 225-dim MediaPipe keypoints, removing the modality gap. DGS and ASL are linguistically unrelated, making any positive transfer evidence of **language-agnostic gesture representations**.

### Architecture Extensions

**Freeze strategies** (`--freeze_strategy full|convblocks|none`):
```
Full freeze:       [input_proj] [conv_blocks] [transformer_blocks] [head ← TRAIN ONLY]
ConvBlocks freeze: [input_proj] [conv_blocks] [transformer_blocks ← TRAIN] [head ← TRAIN]
Full fine-tune:    all layers ← TRAIN (low LR: 5e-5 to prevent catastrophic forgetting)
```

**Language-Aware Joint Encoder** (`LanguageAwareSignTransformer` in `models.py`):
```
x (225-dim) → input_proj → pos_encoding → + lang_embedding(0=DGS, 1=ASL)
            → conv_blocks → transformer_blocks
            → head_phoenix (if lang=0) | head_how2sign (if lang=1)
```

### Running Transfer Experiments

```bash
# Prerequisites: Exp 1 must finish first
kubectl apply -f nautilius/train-exp1-phoenix-sign2gloss.yaml
# Wait for completion, then:

# Launch all transfer experiments at once
bash run_transfer_experiments.sh

# Or selectively:
bash run_transfer_experiments.sh --freeze    # Exps 7-10 (freeze strategy comparison)
bash run_transfer_experiments.sh --joint     # Exp 11 (joint training)
bash run_transfer_experiments.sh --lowdata   # Exps 12-14 (low-data adaptation)
bash run_transfer_experiments.sh --features  # Feature extraction + UMAP/t-SNE (run last)
```

### Representation Analysis

`extract_features.py` visualises what the encoder learns:
- **UMAP/t-SNE** colored by DGS vs ASL — if clusters intermix, encoder is language-agnostic
- **Cross-lingual cosine similarity** — intra-DGS vs intra-ASL vs cross-lingual distance
- **Attention heatmaps** — which frames attend to which
- All plots logged to W&B under `representations/*`

---

## Pseudogloss Methods

How2Sign has no gloss annotations. This project investigates three approaches to automatically generate pseudoglosses as intermediate supervision signals.

### Option A — POS-based Pseudoglosses (implemented)

Extracts content words (NOUN, VERB, ADJ, ADV) from English translations using spaCy lemmatisation and POS filtering. Inspired by [Sign2GPT (ICLR 2024)](https://openreview.net/forum?id=LqaEEs3UxU).

- **Script:** `generate_pseudoglosses_how2sign.py`
- **Input:** How2Sign English sentences from metadata CSVs
- **Output:** `PSEUDOGLOSS` column written in-place to `metadata/how2sign_realigned_{split}.csv`
- **Tokenizer:** Built from pseudogloss vocab at training time (`min_freq=2`)
- **NRP job:** `nautilius/generate-pseudoglosses-job.yaml`

Example:
```
Sentence:    "And I'm going to show you how to make a layered cake."
Pseudogloss: SHOW MAKE LAYER CAKE
```

### Option B — LLM-generated Pseudoglosses (planned)

Few-shot prompting an LLM (GPT-4 / Gemma2) with 30 text-gloss pairs from PHOENIX-2014-T to generate sign-order pseudogloss sequences for each How2Sign sentence. Includes a frame-level reordering classifier to match sign order. Inspired by [PGG-SLT (NeurIPS 2025)](https://arxiv.org/abs/2505.15438) — achieves **13.7 BLEU-4** on How2Sign.

### Option C — Clustering-based Pseudoglosses (planned)

K-means clustering (k=256) on per-stream keypoint features (left hand, right hand, face, body) to produce per-frame discrete pseudo-labels without any annotation. Inspired by [SHuBERT (ACL 2025)](https://arxiv.org/abs/2411.16765) — current **SOTA at 16.2 BLEU-4** on How2Sign.

---

## Datasets

### PHOENIX-2014-T

German Weather Forecast Sign Language dataset. ~8,000 video sequences with gloss annotations and German text translations.

- **Source:** [RWTH-PHOENIX-Weather 2014-T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- **PVC path:** `/data/phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/`
- **Keypoints cached at:** `.../features/keypoints/{train,dev,test}/`

### How2Sign

Large-scale American Sign Language dataset (~80 hours of instructional video). No gloss annotations.

- **Source:** [How2Sign](https://how2sign.github.io/) — official RGB front clips
- **PVC path:** `/data/how2sign_hf/`
- **Pseudoglosses at:** `/data/how2sign_hf/annotations/how2sign_{split}.csv` (`PSEUDOGLOSS` column, generated by `generate_pseudoglosses_how2sign.py`)

---

## Project Structure

```
slt/
├── train.py                          # Main training + evaluation entry point
├── train_joint.py                    # Joint PHOENIX+How2Sign training (Exp 11)
├── extract_features.py               # UMAP/t-SNE encoder representation analysis
├── models.py                         # SignLanguageTransformer, LanguageAwareSignTransformer
├── dataset.py                        # PhoenixSignDataset
├── dataset_how2sign.py               # How2SignDataset (loads PSEUDOGLOSS column)
├── preprocessing.py                  # PhoenixKeypointExtractor (MediaPipe)
├── preextract_keypoints.py           # Offline Phoenix keypoint extraction
├── generate_pseudoglosses_how2sign.py# Option A: POS-based pseudogloss generation
├── analyze_datasets.py               # Dataset analysis + W&B logging
├── evaluate.py                       # Standalone evaluation
├── utils.py                          # GlossTokenizer, Trainer, collate_fn, CTC decoding
├── TRANSFER_LEARNING.md              # Full transfer learning experiment plan
├── run_transfer_experiments.sh       # Master runner for Exps 7–14
├── Dockerfile                        # Image: ghcr.io/kagozi/slt:latest
├── environment.yaml                  # Conda environment spec
│
├── .github/
│   └── workflows/
│       └── docker.yaml               # CI/CD: build + push on push to main
│
└── nautilius/                        # NRP / Kubernetes manifests
    ├── slt-data-pvc.yaml                 # 500Gi ReadWriteMany CephFS PVC
    ├── data-uploader.yaml                # Pod for manual data uploads via kubectl cp
    ├── pvc-inspector.yaml                # Pod for browsing PVC contents
    ├── extract-phoenix.yaml              # Extracts phoenix-2014-T.v3.tar.gz on PVC
    ├── create-kaggle-secret.sh           # Creates Kaggle K8s secret from .env
    ├── create-wandb-secret.sh            # Creates W&B K8s secret from .env
    ├── download-how2sign-job.yaml        # DEPRECATED — How2Sign now uploaded manually to /data/how2sign_hf/
    ├── preextract-phoenix-job.yaml       # Job: MediaPipe keypoint extraction
    ├── generate-pseudoglosses-job.yaml   # Job: Option A POS pseudogloss generation
    ├── analyze-datasets-job.yaml         # Job: dataset analysis + W&B logging
    ├── train-exp1-phoenix-sign2gloss.yaml
    ├── train-exp2-phoenix-sign2gloss2text.yaml
    ├── train-exp3-phoenix-glossless.yaml
    ├── train-exp4-how2sign-glossless.yaml
    ├── train-exp5-how2sign-pseudogloss.yaml      # How2Sign Sign2Gloss2Text (CTC+BART)
    ├── train-exp6-how2sign-sign2gloss.yaml       # How2Sign Sign2Gloss (CTC-only)
    ├── train-exp7-transfer-full-freeze.yaml      # Transfer: full encoder freeze
    ├── train-exp8-transfer-conv-freeze.yaml      # Transfer: ConvBlocks frozen
    ├── train-exp9-transfer-full-finetune.yaml    # Transfer: full fine-tune (low LR)
    ├── train-exp10-transfer-scratch.yaml         # Transfer: scratch control
    ├── train-exp11-joint-training.yaml           # Joint PHOENIX+How2Sign
    ├── train-exp12-lowdata-10pct.yaml            # Low-data: 10% of How2Sign
    ├── train-exp13-lowdata-5pct.yaml             # Low-data: 5% of How2Sign
    ├── train-exp14-lowdata-1pct.yaml             # Low-data: 1% of How2Sign
    └── extract-features-job.yaml                 # UMAP/t-SNE feature analysis
```

---

## Setup

### Local

```bash
git clone https://github.com/kagozi/slt.git
cd slt

conda env create -f environment.yaml
conda activate slt-multistream

# Extract Phoenix keypoints once before training
python preextract_keypoints.py \
  --root_dir /path/to/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
  --max_frames 250

# Generate How2Sign pseudoglosses once before training
python generate_pseudoglosses_how2sign.py \
  --root_dir /path/to/how2sign
```

### Environment variables

Create a `.env` file (never committed):

```
WANDB_API_KEY=...
KAGGLE_USER_NAME=...
KAGGE_API_KEY=...
```

---

## NRP / Nautilus Cluster

All jobs target namespace `gai-lina-group`. The PVC uses `rook-cephfs` (ReadWriteMany) so multiple jobs can mount it simultaneously.

### First-time setup

```bash
# 1. Create PVC (ReadWriteMany, 500Gi)
kubectl apply -f nautilius/slt-data-pvc.yaml

# 2. Register secrets
chmod +x nautilius/create-kaggle-secret.sh nautilius/create-wandb-secret.sh
./nautilius/create-kaggle-secret.sh
./nautilius/create-wandb-secret.sh

# 3. Upload Phoenix tarball
kubectl apply -f nautilius/data-uploader.yaml
kubectl wait --for=condition=Ready pod/data-uploader -n gai-lina-group --timeout=60s
kubectl cp /path/to/phoenix-2014-T.v3.tar.gz \
  gai-lina-group/data-uploader:/data/phoenix-2014-T.v3.tar.gz
```

### Data pipeline (run in parallel — RWX PVC supports it)

```bash
kubectl apply -f nautilius/extract-phoenix.yaml
kubectl apply -f nautilius/download-how2sign-job.yaml
```

### Preprocessing (run in parallel after data jobs complete)

```bash
kubectl apply -f nautilius/preextract-phoenix-job.yaml
kubectl apply -f nautilius/generate-pseudoglosses-job.yaml
```

### Launch base experiments (Exps 1–6)

```bash
kubectl apply -f nautilius/train-exp1-phoenix-sign2gloss.yaml
kubectl apply -f nautilius/train-exp2-phoenix-sign2gloss2text.yaml
kubectl apply -f nautilius/train-exp3-phoenix-glossless.yaml
kubectl apply -f nautilius/train-exp4-how2sign-glossless.yaml
kubectl apply -f nautilius/train-exp5-how2sign-pseudogloss.yaml
kubectl apply -f nautilius/train-exp6-how2sign-sign2gloss.yaml
```

### Launch transfer learning experiments (Exps 7–14)

```bash
# Run AFTER Exp 1 completes (provides the pretrained checkpoint)
bash run_transfer_experiments.sh

# After all training jobs finish, run representation analysis:
bash run_transfer_experiments.sh --features
```

### Monitor

```bash
kubectl get pods -n gai-lina-group
kubectl logs -f job/<job-name> -n gai-lina-group
```

### Inspect PVC contents

```bash
kubectl apply -f nautilius/pvc-inspector.yaml
kubectl exec -it pvc-inspector -n gai-lina-group -- ls /data
kubectl delete pod pvc-inspector -n gai-lina-group
```

### PVC layout after full pipeline

```
/data/
├── phoenix-2014-T.v3.tar.gz
├── phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
│   ├── annotations/manual/          # corpus CSVs
│   └── features/
│       ├── fullFrame-210x260px/     # original frames
│       └── keypoints/{train,dev,test}/  # cached .npy (preextract job)
├── how2sign/
│   ├── how2sign_holistic_features/metadata/how2sign_realigned_{train,val,test}.csv  # includes PSEUDOGLOSS col
│   └── {train,val,test}/frontal/*.npy
└── experiments/
    ├── models/
    │   ├── exp1_phoenix_sign2gloss/
    │   │   ├── best_model.pt
    │   │   ├── checkpoint_epoch_*.pt
    │   │   └── final_*.pt
    │   └── ...
    └── results/
        ├── exp1_phoenix_sign2gloss/
        │   ├── metrics_*.json
        │   ├── predictions_*.csv
        │   └── args_*.json
        └── ...
```

---

## Training

### PHOENIX-2014-T

```bash
# Exp 1: Sign2Gloss (CTC only)
python train.py --epochs 150 --decode beam --beam_width 10

# Exp 2: Sign2Gloss2Text (joint CTC + BART)
python train.py --epochs 150 --decode beam --beam_width 10 \
    --use_bart --ctc_weight 0.5 --freeze_bart_epochs 15

# Exp 3: Glossless Sign2Text (BART only)
python train.py --epochs 150 --decode beam --beam_width 10 \
    --use_bart --ctc_weight 0.0
```

### How2Sign

```bash
# Generate pseudoglosses first (one-time)
python generate_pseudoglosses_how2sign.py --root_dir /data/how2sign_hf

# Exp 4: Glossless baseline (BART only, no pseudoglosses)
python train.py --dataset how2sign \
    --root_dir /data/how2sign_hf \
    --use_bart --ctc_weight 0.0 \
    --max_frames 300 --epochs 150 --decode beam --beam_width 10

# Exp 5: POS pseudogloss + BART (Option A)
python train.py --dataset how2sign \
    --root_dir /data/how2sign_hf \
    --use_bart --ctc_weight 0.5 --freeze_bart_epochs 15 \
    --max_frames 300 --epochs 150 --decode beam --beam_width 10
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `phoenix` | `phoenix` or `how2sign` |
| `--root_dir` | auto | Dataset root path |
| `--output_dir` | `..` | Base dir for `results/` and `models/` |
| `--epochs` | 100 | Training epochs |
| `--dim` | 256 | Model hidden dimension |
| `--batch_size` | 20 | Batch size |
| `--decode` | `greedy` | `greedy` or `beam` |
| `--beam_width` | 10 | Beam search width |
| `--use_bart` | off | Enable BART translation head |
| `--ctc_weight` | 0.3 | CTC loss weight (0=BART only, 1=CTC only) |
| `--freeze_bart_epochs` | 5 | Epochs to freeze BART before joint training |
| `--exp_name` | auto | Custom experiment name |
| `--resume` | — | Path to checkpoint to resume from |
| `--pretrained_path` | — | Path to pretrained checkpoint for transfer learning |
| `--freeze_strategy` | `none` | `full`, `convblocks`, or `none` — encoder freeze for transfer |
| `--subset_pct` | `100` | % of training data to use (for low-data experiments) |
| `--lr` | `1e-3` | Base learning rate (use `5e-5` for full fine-tune transfer) |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **WER** | Word Error Rate on gloss sequences (lower is better) |
| **BLEU-1/4** | n-gram precision on gloss predictions |
| **Trans BLEU-1/4** | Translation quality (BART mode only) |
| **Exact Match** | Fraction of perfectly predicted gloss sequences |

All metrics are logged to [Weights & Biases](https://wandb.ai) (`project: slt`) per epoch. Final predictions are logged as W&B Tables with target/prediction pairs.

---

## CI/CD

Push to `main` triggers `.github/workflows/docker.yaml`:

1. Builds the Docker image
2. Pushes `ghcr.io/kagozi/slt:latest` + `sha-<commit>` tag
3. All NRP jobs pull from `ghcr.io/kagozi/slt:latest`

Make the package public in GitHub → Packages → slt → Package settings → Change visibility → Public, so NRP can pull without an image pull secret.
