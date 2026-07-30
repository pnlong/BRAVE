#!/usr/bin/env bash
# Resume water_uncond_run from epoch-7343 ckpt and verify phase-2 (step 1M) fix.
#
# Usage (from cluster login node):
#   bash scripts/submit_water_phase2_resume.sh
#   bash scripts/submit_water_phase2_resume.sh --gres=gpu:1

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

export DB_PATH="/data/scratch-fast/p1long/BRAVE/fsd50k_brave/water/preprocessed"
export RUN_NAME="water_uncond_run"
export CONFIG="configs/brave.gin"
export OUT_PATH="/data/scratch-fast/p1long/BRAVE/fsd50k_brave/water/runs"
export CKPT="${OUT_PATH}/water_uncond_run_8e1e614287/epoch-epoch=7343.ckpt"
export BATCH=8
export WORKERS=0
export WANDB_RUN_ID="ibgpu2ag"
export CUDNN_BENCHMARK=0
export CUDA_LAUNCH_BLOCKING=1

mkdir -p logs

echo "Submitting water phase-2 resume (model.py no_grad fix on disc steps):"
echo "  CKPT=${CKPT}"
echo "  WORKERS=${WORKERS}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the cluster login node." >&2
  exit 1
fi

job_id="$(
  sbatch --parsable \
    --chdir="${BRAVE_ROOT}" \
    --job-name=water-phase2 \
    --output="${BRAVE_ROOT}/logs/train-%j.log" \
    --error="${BRAVE_ROOT}/logs/train-%j.log" \
    "$@" \
    scripts/train.sbatch
)" || {
  echo "ERROR: sbatch failed." >&2
  exit 1
}

echo ""
echo "Submitted job ${job_id}"
echo "Log: ${BRAVE_ROOT}/logs/train-${job_id}.log"
echo "Watch: tail -f ${BRAVE_ROOT}/logs/train-${job_id}.log"
echo ""
echo "Success = log passes Epoch 7346 batch 124+ without CUDA illegal memory access."
