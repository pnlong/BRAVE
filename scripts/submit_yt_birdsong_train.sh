#!/usr/bin/env bash
# Submit unconditional BRAVE training on yt_birdsong (same LMDB as the Fader run).
#
# Usage (from repo root):
#   bash scripts/submit_yt_birdsong_train.sh
#   bash scripts/submit_yt_birdsong_train.sh --gres=gpu:4 --cpus-per-task=32
#
# Do NOT use multiline sbatch --export=ALL,\ — bash splits the list and SLURM
# never sees DB_PATH / RUN_NAME (the job inherits stale shell vars instead).

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

export DB_PATH="/data/scratch-fast/p1long/BRAVE/yt_birdsong/preprocessed_pcen"
export RUN_NAME="yt_birdsong_run"
export CONFIG="configs/brave_birdsong.gin"
export OUT_PATH="/data/scratch-fast/p1long/BRAVE/yt_birdsong/runs"

# Avoid accidentally resuming a different experiment left in the shell.
unset CKPT WANDB_RUN_ID GPUS GPU

mkdir -p logs

echo "Submitting yt_birdsong unconditional BRAVE:"
echo "  DB_PATH=${DB_PATH}"
echo "  RUN_NAME=${RUN_NAME}"
echo "  CONFIG=${CONFIG}"
echo "  OUT_PATH=${OUT_PATH}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the cluster login node, not a local/Cursor shell." >&2
  exit 1
fi

# --chdir + absolute log paths so output lands in BRAVE/logs even if sbatch cwd differs.
job_id="$(
  sbatch --parsable \
    --chdir="${BRAVE_ROOT}" \
    --job-name=birdsong-brave \
    --output="${BRAVE_ROOT}/logs/train-%j.log" \
    --error="${BRAVE_ROOT}/logs/train-%j.log" \
    "$@" \
    scripts/train.sbatch
)" || {
  echo "ERROR: sbatch failed (no job submitted — check message above)." >&2
  exit 1
}

echo ""
echo "Submitted job ${job_id}"
echo "Log file: ${BRAVE_ROOT}/logs/train-${job_id}.log"
echo "Watch:    tail -f ${BRAVE_ROOT}/logs/train-${job_id}.log"
echo ""
echo "First lines should show DB_PATH=.../yt_birdsong/preprocessed_pcen and RUN_NAME=yt_birdsong_run"
