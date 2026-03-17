#!/bin/bash
# Creates a Kubernetes Secret with Kaggle credentials from the project's .env file.
# Run once before submitting the download-how2sign-job.
#
# Usage:
#   chmod +x create-kaggle-secret.sh
#   ./create-kaggle-secret.sh

set -e

ENV_FILE="$(dirname "$0")/../.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env file not found at $ENV_FILE"
  exit 1
fi

# Parse values from .env
KAGGLE_USERNAME=$(grep '^KAGGLE_USER_NAME=' "$ENV_FILE" | cut -d= -f2 | tr -d '[:space:]')
KAGGLE_KEY=$(grep '^KAGGE_API_KEY=' "$ENV_FILE" | cut -d= -f2 | tr -d '[:space:]')

if [[ -z "$KAGGLE_USERNAME" || -z "$KAGGLE_KEY" ]]; then
  echo "ERROR: Could not parse KAGGLE_USER_NAME or KAGGE_API_KEY from .env"
  exit 1
fi

kubectl delete secret kaggle-credentials -n gai-lina-group --ignore-not-found

kubectl create secret generic kaggle-credentials \
  -n gai-lina-group \
  --from-literal=username="$KAGGLE_USERNAME" \
  --from-literal=api-key="$KAGGLE_KEY"

echo "Secret 'kaggle-credentials' created successfully."
