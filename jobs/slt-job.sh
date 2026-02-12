#!/bin/bash
#
# GPU job submission script for SLURM
#

### Job Configuration #########################################################

#SBATCH --job-name=sl-preprocessing
#SBATCH --output=sl-preprocessing-%j.out  # Log file with Job ID in name

# Export environment variables
#SBATCH --get-user-env

# Resources
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=gpu              # <--- Required to use GPUs
#SBATCH --gres=gpu:1          # Request one Pascal GPU
#SBATCH --mem=32G                    # Request memory (optional but recommended)

# Time limit (job will be killed if exceeded)
#SBATCH --time=48:00:00              # 10 hours

# Email notifications
#SBATCH --mail-user=alex.kagozi@coyotes.usd.edu
#SBATCH --mail-type=BEGIN,END,FAIL

### Commands to run your program #############################################

echo "Job started on $(date)"
pwd
nvidia-smi

# Activate conda/venv if needed
# source ~/.bashrc
# conda activate myenv
eval "$(mamba shell hook --shell bash)"
mamba activate slt-multistream

DATA_ROOT="../phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T"
OUT_DIR="../data_cache"

for SPLIT in train dev test; do
  echo "===== PREPROCESSING $SPLIT ====="
  python -m scripts.preprocess_rgb \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_DIR" \
    --split "$SPLIT"

  python -m scripts.preprocess_kpts \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_DIR" \
    --split "$SPLIT"

  python -m scripts.preprocess_hands \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUT_DIR" \
    --split "$SPLIT"
done

echo "Job ended on $(date)"
nvidia-smi

