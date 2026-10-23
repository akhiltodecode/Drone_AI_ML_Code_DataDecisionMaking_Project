#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONHASHSEED=0
study_workers="${DRONE_STUDY_WORKERS:-4}"

python runpod_runner.py \
  --mode extended-paper \
  --output results/extended \
  --replications 30 \
  --cv-folds 3 \
  --search-candidates 12 \
  --workers "$study_workers" \
  --bootstrap-resamples 5000 \
  --resume
