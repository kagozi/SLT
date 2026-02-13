#!/bin/bash
#
# GPU job submission script for SLURM
# Stage 1: RGB-only Video-to-Gloss Training
#

### Job Configuration #########################################################

#SBATCH --job-name=slt-stage1-train
#SBATCH --output=slt-stage1-train-%j.out

#SBATCH --get-user-env

# Resources
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

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
export NLTK_DATA=$PWD/.nltk_data
python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA'); nltk.download('omw-1.4', download_dir='$NLTK_DATA')"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_CACHE="../data_cache"

mkdir -p "$RUN_DIR"

python train.py --data_cache "$DATA_CACHE" --dataset phoenix2014t --data_root "$DATA_CACHE"

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="

nvidia-smi
