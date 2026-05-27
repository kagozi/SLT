FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    git \
    unrar \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
      "numpy<2" scipy pandas tqdm pyyaml rich click \
      opencv-python pillow matplotlib \
      mediapipe==0.10.14 \
      ultralytics \
      einops sentencepiece regex scikit-learn \
      omegaconf hydra-core tensorboard sacrebleu nltk rouge_score \
      wandb kaggle rarfile sacrebleu yt-dlp \
      "transformers==4.44.2" \
      spacy && \
    python -m spacy download en_core_web_sm

# Copy code
COPY *.py .

ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
