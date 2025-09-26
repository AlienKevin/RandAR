#!/bin/bash

# RandAR Setup Script

# setup uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync
source .venv/bin/activate

# Downloads required model weights and reference data

echo "Starting RandAR setup..."

mkdir -p temp

# Download LlamaGen VQ model weights
echo "Downloading LlamaGen VQ model weights..."
wget https://huggingface.co/FoundationVision/LlamaGen/resolve/main/vq_ds16_c2i.pt -O temp/vq_ds16_c2i.pt

# Download RandAR model weights
echo "Downloading RandAR model weights..."
wget https://huggingface.co/ziqipang/RandAR/resolve/main/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors -O temp/randar_0.3b_llamagen_360k_bs_1024_lr_0.0004.safetensors

# Download ImageNet reference batch
echo "Downloading ImageNet reference batch..."
wget https://huggingface.co/AlienKevin/adm_evals/resolve/main/VIRTUAL_imagenet256_labeled.npz -O temp/VIRTUAL_imagenet256_labeled.npz

# Log into gcloud for storage
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud auth login
gcloud config set project flowmo
gcloud auth application-default login

echo "Setup complete!"
