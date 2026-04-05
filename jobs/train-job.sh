#!/bin/bash
#
# GPU job submission script for SLURM
# Stage 1: RGB-only Video-to-Gloss Training
#

### Job Configuration #########################################################

#SBATCH --job-name=mediapipe-asl-train
#SBATCH --output=mediapipe-asl-train-%j.out

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
# Exp 1: Sign2Gloss (CTC-only baseline, fastest)
# python train.py --epochs 150 --decode beam --beam_width 10

# Exp 2: Sign2Gloss2Text (joint CTC + BART, 3-stage training)
# python train.py --epochs 150 --decode beam --beam_width 10 \
#     --use_bart --ctc_weight 0.5 --freeze_bart_epochs 15

# Exp 3: Glossless Sign2Text (BART-only, no CTC)
# python train.py --epochs 150 --decode beam --beam_width 10 \
#     --use_bart --ctc_weight 0.0 --freeze_bart_epochs 0


## American Sign Language (ASL)
python train.py \
    --dataset how2sign \
    --root_dir /data/how2sign_hf \
    --use_bart \
    --ctc_weight 0.0 \
    --freeze_bart_epochs 0 \
    --max_frames 300 \
    --epochs 150 \
    --decode beam \
    --beam_width 10

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
