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

# python -c "import torch; print('Torch CUDA available:', torch.cuda.is_available())"
# export NLTK_DATA=$PWD/.nltk_data
# python -c "import nltk; nltk.download('wordnet', download_dir='$NLTK_DATA'); nltk.download('omw-1.4', download_dir='$NLTK_DATA')"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# python preextract_keypoints.py --root_dir ../phoenix2014/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/
# Run 1: CTC-only, greedy, 100 epochs (baseline)
python train.py --epochs 100 --decode greedy

# Run 2: CTC-only, beam search, 150 epochs
python train.py --epochs 150 --decode beam --beam_width 10

# Run 3: CTC + BART translation, 150 epochs
python train.py --epochs 150 --decode beam --beam_width 10 --use_bart --bart_model facebook/bart-base

# Run 4: CTC + BART, more aggressive joint training
python train.py --epochs 150 --use_bart --ctc_weight 0.5 --freeze_bart_epochs 10

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
