FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Python deps
COPY environment.yaml .
RUN pip install --upgrade pip && \
    pip install \
      numpy scipy pandas tqdm pyyaml rich click \
      opencv-python pillow matplotlib \
      torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117 \
      mediapipe==0.10.14 \
      ultralytics \
      einops sentencepiece regex scikit-learn \
      omegaconf hydra-core tensorboard sacrebleu nltk \
      wandb kaggle \
      transformers \
      spacy && \
    python -m spacy download en_core_web_sm

# Copy code
COPY *.py .

ENV PYTHONPATH=/workspace