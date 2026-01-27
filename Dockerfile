FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 \
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
      torch torchvision torchaudio \
      mediapipe==0.10.14 \
      ultralytics \
      einops sentencepiece regex scikit-learn \
      omegaconf hydra-core tensorboard sacrebleu nltk

# Copy code
COPY preprocessing preprocessing
COPY scripts scripts

ENV PYTHONPATH=/workspace