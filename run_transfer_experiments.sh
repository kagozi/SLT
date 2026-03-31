#!/usr/bin/env bash
# ============================================================
# run_transfer_experiments.sh
# Launch all transfer learning experiments (Exps 7–14) on NRP.
#
# PREREQUISITE: Exp 1 must be complete and its checkpoint must
# exist at:
#   /data/experiments/exp1_phoenix_sign2gloss/models/best_model.pt
#
# Usage:
#   bash run_transfer_experiments.sh            # launch all
#   bash run_transfer_experiments.sh --freeze   # freeze exps only (7-10)
#   bash run_transfer_experiments.sh --lowdata  # low-data exps only (12-14)
#   bash run_transfer_experiments.sh --joint    # joint training only (11)
#   bash run_transfer_experiments.sh --features # feature extraction only
# ============================================================

set -euo pipefail

NS="gai-lina-group"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/nautilius"

MODE="${1:-all}"

run_freeze_exps() {
    echo "==> Launching freeze-strategy experiments (7–10)..."
    kubectl apply -f "${DIR}/train-exp7-transfer-full-freeze.yaml"
    kubectl apply -f "${DIR}/train-exp8-transfer-conv-freeze.yaml"
    kubectl apply -f "${DIR}/train-exp9-transfer-full-finetune.yaml"
    kubectl apply -f "${DIR}/train-exp10-transfer-scratch.yaml"
    echo "    Exps 7-10 submitted."
}

run_joint_exp() {
    echo "==> Launching joint training experiment (11)..."
    kubectl apply -f "${DIR}/train-exp11-joint-training.yaml"
    echo "    Exp 11 submitted."
}

run_lowdata_exps() {
    echo "==> Launching low-data transfer experiments (12–14)..."
    kubectl apply -f "${DIR}/train-exp12-lowdata-10pct.yaml"
    kubectl apply -f "${DIR}/train-exp13-lowdata-5pct.yaml"
    kubectl apply -f "${DIR}/train-exp14-lowdata-1pct.yaml"
    echo "    Exps 12-14 submitted."
}

run_feature_extraction() {
    echo "==> Launching feature extraction job..."
    kubectl apply -f "${DIR}/extract-features-job.yaml"
    echo "    Feature extraction job submitted."
}

case "${MODE}" in
    --freeze)
        run_freeze_exps
        ;;
    --joint)
        run_joint_exp
        ;;
    --lowdata)
        run_lowdata_exps
        ;;
    --features)
        run_feature_extraction
        ;;
    all)
        run_freeze_exps
        run_joint_exp
        run_lowdata_exps
        echo ""
        echo "All transfer experiments submitted."
        echo "Run feature extraction AFTER all jobs complete:"
        echo "  bash run_transfer_experiments.sh --features"
        ;;
    *)
        echo "Unknown option: ${MODE}"
        echo "Usage: $0 [--freeze|--joint|--lowdata|--features|all]"
        exit 1
        ;;
esac

echo ""
echo "Monitor with:"
echo "  kubectl get pods -n ${NS}"
echo "  kubectl logs -f job/<job-name> -n ${NS}"
echo ""
echo "Track metrics at: https://wandb.ai → project: slt"
