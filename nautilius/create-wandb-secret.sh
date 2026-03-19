#!/bin/bash
# Creates a Kubernetes Secret with the W&B API key from the project's .env file.
# Run once before submitting any training job.
#
# Usage:
#   chmod +x create-wandb-secret.sh
#   ./create-wandb-secret.sh

set -e

ENV_FILE="$(dirname "$0")/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env file not found at $ENV_FILE"
  exit 1
fi

WANDB_API_KEY=$(grep '^WANDB_API_KEY=' "$ENV_FILE" | cut -d= -f2 | tr -d '[:space:]')

if [[ -z "$WANDB_API_KEY" ]]; then
  echo "ERROR: Could not parse WANDB_API_KEY from .env"
  exit 1
fi

kubectl delete secret wandb-credentials -n gai-lina-group --ignore-not-found

kubectl create secret generic wandb-credentials \
  -n gai-lina-group \
  --from-literal=api-key="$WANDB_API_KEY"

echo "Secret 'wandb-credentials' created successfully."
