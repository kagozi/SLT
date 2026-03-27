#!/bin/bash
# DEPRECATED — Kaggle Holistic download script.
#
# How2Sign data is now sourced from the official RGB front clips,
# manually uploaded to the NRP PVC at /data/how2sign_rgb/.
#
# Current data pipeline (Nautilus jobs):
#   kubectl apply -f nautilius/extract-how2sign-rgb-job.yaml
#   kubectl apply -f nautilius/preextract-how2sign-keypoints-job.yaml
#   kubectl apply -f nautilius/generate-pseudoglosses-job.yaml   # exp5 only
echo "This script is deprecated. See nautilius/ for the current How2Sign pipeline."
