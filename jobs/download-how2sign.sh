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

nvidia-smi

# ---------------------------------------------------------------------------
# Activate environment
# ---------------------------------------------------------------------------

eval "$(mamba shell hook --shell bash)"
mamba activate slt-multistream

#python -c "import kagglehub; path = kagglehub.dataset_download('psewmuthu/how2sign-holistic'); print('Path to dataset files:', path)"
# ---------------------------------------------------------------------------
# Download dataset into current submission folder
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Use lab shared directory for BOTH data and cache
# ---------------------------------------------------------------------------

DATA_DIR=/home/usd.local/alex.kagozi/santosh_lab/shared/KagoziA/gsl/HowToSign
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Redirect all cache away from ~/.cache
cd /home/usd.local/alex.kagozi/santosh_lab/shared/KagoziA/gsl/HowToSign

export KAGGLE_CONFIG_DIR=/home/usd.local/alex.kagozi/.kaggle

kaggle datasets download -d psewmuthu/how2sign-holistic --unzip


echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="