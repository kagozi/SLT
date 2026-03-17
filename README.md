# Sign Language Translation (SLT)

End-to-end sign language recognition and translation using body keypoints extracted with MediaPipe. Trains on PHOENIX-2014-T (German Sign Language) and How2Sign (American Sign Language) datasets.

**Thesis project — University of South Dakota**

---

## Table of Contents

- [Architecture](#architecture)
- [Experiments](#experiments)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [NRP / Nautilus Cluster](#nrp--nautilus-cluster)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)

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

| # | Dataset | Mode | Key Flags |
|---|---|---|---|
| 1 | PHOENIX-2014-T | Sign2Gloss | *(default)* |
| 2 | PHOENIX-2014-T | Sign2Gloss2Text | `--use_bart --ctc_weight 0.5 --freeze_bart_epochs 15` |
| 3 | PHOENIX-2014-T | Glossless Sign2Text | `--use_bart --ctc_weight 0.0` |
| 4 | How2Sign | Glossless Sign2Text | `--dataset how2sign --use_bart --ctc_weight 0.0` |
| 5 | How2Sign | Pseudo-gloss | `--dataset how2sign --use_bart --ctc_weight 0.5` |

---

## Datasets

### PHOENIX-2014-T

German Weather Forecast Sign Language dataset. Contains ~8,000 video sequences with gloss annotations and German text translations.

- **Source:** [RWTH-PHOENIX-Weather 2014-T](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
- **PVC path (NRP):** `/data/phoenix2014/PHOENIX-2014-T-release-v3/`
- **Keypoints cached at:** `.../features/keypoints/{train,dev,test}/`

### How2Sign

Large-scale American Sign Language dataset (~80 hours of instructional video).

- **Source:** [Kaggle — psewmuthu/how2sign-holistic](https://www.kaggle.com/datasets/psewmuthu/how2sign-holistic)
- **PVC path (NRP):** `/data/how2sign/`

---

## Project Structure

```
slt/
├── train.py                   # Main training + evaluation entry point
├── models.py                  # SignLanguageTransformer, BARTTranslationHead
├── dataset.py                 # PhoenixSignDataset
├── dataset_how2sign.py        # How2SignDataset
├── preprocessing.py           # PhoenixKeypointExtractor (MediaPipe)
├── preextract_keypoints.py    # Offline keypoint extraction script
├── evaluate.py                # Standalone evaluation
├── utils.py                   # GlossTokenizer, Trainer, collate_fn, CTC decoding
├── Dockerfile                 # Image: ghcr.io/kagozi/slt:latest
├── environment.yaml           # Conda environment spec
├── requirements.txt
│
├── .github/
│   └── workflows/
│       └── docker.yaml        # CI/CD: build + push on push to main/feature/*
│
└── nautilius/                 # NRP / Kubernetes manifests
    ├── slt-data-pvc.yaml          # 500Gi CephFS PVC
    ├── data-uploader.yaml         # Pod for manual data uploads
    ├── pvc-inspector.yaml         # Pod for browsing PVC contents
    ├── extract-phoenix.yaml       # Extracts phoenix-2014-T.v3.tar.gz on PVC
    ├── create-kaggle-secret.sh    # Creates K8s secret from .env
    ├── download-how2sign-job.yaml # Job: download How2Sign from Kaggle → PVC
    └── preextract-phoenix-job.yaml# Job: run MediaPipe keypoint extraction
```

---

## Setup

### Local

```bash
# Clone
git clone https://github.com/kagozi/slt.git
cd slt

# Create environment
conda env create -f environment.yaml
conda activate slt-multistream

# Extract keypoints for PHOENIX (run once before training)
python preextract_keypoints.py \
  --root_dir /path/to/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \
  --max_frames 250
```

### Environment variables

Copy `.env.example` to `.env` and fill in your keys:

```
WANDB_API_KEY=...
KAGGLE_USER_NAME=...
KAGGE_API_KEY=...
```

---

## NRP / Nautilus Cluster

All jobs target namespace `gai-lina-group`. Run steps in order.

### 1. Create the PVC

```bash
kubectl apply -f nautilius/slt-data-pvc.yaml
```

> Already done if the Phoenix data is present on the cluster.

### 2. Register Kaggle credentials

```bash
chmod +x nautilius/create-kaggle-secret.sh
./nautilius/create-kaggle-secret.sh
```

### 3. Download How2Sign (~70 GB)

```bash
kubectl apply -f nautilius/download-how2sign-job.yaml
kubectl logs -f job/download-how2sign -n gai-lina-group
# Saves to /data/how2sign/
```

### 4. Extract Phoenix keypoints

The Docker image is built and pushed to `ghcr.io/kagozi/slt:latest` automatically by CI on every push to `main`.

```bash
kubectl apply -f nautilius/preextract-phoenix-job.yaml
kubectl logs -f job/preextract-phoenix-keypoints -n gai-lina-group
# Writes .npy files to /data/phoenix2014/.../features/keypoints/{train,dev,test}/
```

### Inspect PVC contents at any time

```bash
kubectl apply -f nautilius/pvc-inspector.yaml
kubectl exec -it pvc-inspector -n gai-lina-group -- ls /data
kubectl delete pod pvc-inspector -n gai-lina-group
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
# Exp 4: Glossless Sign2Text
python train.py --dataset how2sign \
    --root_dir /data/how2sign \
    --use_bart --ctc_weight 0.0 \
    --max_frames 300 --epochs 150 --decode beam --beam_width 10

# Exp 5: Pseudo-gloss
python train.py --dataset how2sign \
    --root_dir /data/how2sign \
    --use_bart --ctc_weight 0.5 --freeze_bart_epochs 15 \
    --max_frames 300 --epochs 150 --decode beam --beam_width 10
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `phoenix` | `phoenix` or `how2sign` |
| `--epochs` | 100 | Training epochs |
| `--dim` | 192 | Model hidden dimension |
| `--batch_size` | 20 | Batch size |
| `--decode` | `greedy` | `greedy` or `beam` |
| `--beam_width` | 10 | Beam search width |
| `--use_bart` | off | Enable BART translation head |
| `--ctc_weight` | 0.3 | CTC loss weight (0=BART only, 1=CTC only) |
| `--freeze_bart_epochs` | 5 | Epochs to freeze BART before joint training |
| `--exp_name` | auto | Custom experiment name |
| `--resume` | — | Path to checkpoint to resume from |

Results and checkpoints are saved to `../results/<exp_name>/` and `../models/<exp_name>/`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **WER** | Word Error Rate on gloss sequences (lower is better) |
| **BLEU-1/4** | n-gram precision on gloss predictions |
| **Trans BLEU-1/4** | Translation quality (German text, BART mode only) |
| **Exact Match** | Fraction of perfectly predicted gloss sequences |

---
