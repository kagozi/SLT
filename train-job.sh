#!/bin/bash
#
# GPU job submission script for SLURM
# Stage 1: RGB-only Video-to-Gloss Training
#

### Job Configuration #########################################################

#SBATCH --job-name=slt-stage1-train
#SBATCH --output=slt-stage1-train-%j.out
#SBATCH --error=slt-stage1-train-%j.err

#SBATCH --get-user-env

# Resources
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Time limit
#SBATCH --time=48:00:00

# Email notifications
#SBATCH --mail-user=alex.kagozi@coyotes.usd.edu
#SBATCH --mail-type=BEGIN,END,FAIL

##############################################################################

echo "=========================================="
echo "SLT Stage-1 Training Job Started"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

nvidia-smi

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------

eval "$(mamba shell hook --shell bash)"
mamba activate slt-multistream

python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_CACHE="../data_cache"
RUN_DIR="../runs/video_stage1"

mkdir -p "$RUN_DIR"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

echo "=========================================="
echo "Starting Stage-1 RGB-only Training"
echo "=========================================="

python scripts.train_video_model \
  --data_cache "$DATA_CACHE" \
  --output_dir "$RUN_DIR" \
  --epochs 50 \
  --batch_size 8 \
  --encoder_backbone resnet18 \
  --encoder_layers 4 \
  --decoder_layers 6 \
  --nhead 8 \
  --d_model 512 \
  --dropout 0.1 \
  --lr_factor 1.0 \
  --warmup_steps 4000 \
  --grad_clip 1.0 \
  --label_smoothing 0.1 \
  --num_workers 4 \
  --seed 42 \
  --device cuda


TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
  echo "❌ Training failed with exit code $TRAIN_EXIT_CODE"
  exit $TRAIN_EXIT_CODE
fi

# ---------------------------------------------------------------------------
# Optional: Evaluation on test set
# ---------------------------------------------------------------------------

echo "=========================================="
echo "Running evaluation on test split"
echo "=========================================="

BEST_MODEL="$RUN_DIR/best_model.pt"

if [ -f "$BEST_MODEL" ]; then
  python scripts.test_video_model \
    --checkpoint "$BEST_MODEL" \
    --data_cache "$DATA_CACHE" \
    --split test \
    --batch_size 16 \
    --num_workers 4 \
    --output "$RUN_DIR/test_results.json"
else
  echo "⚠️ Best model not found, skipping evaluation"
fi

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

nvidia-smi
