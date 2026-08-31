#!/usr/bin/env bash
# Submit joint LMDB preprocess (domain X + domain Y → one output).
#
# Default: tap contact mic (audio_subset_clean) + yt_birdsong, normalize on.
#
# Usage (from repo root, on the login node):
#   bash scripts/submit_joint_preprocess.sh
#   NORMALIZE=1 PCEN=0 bash scripts/submit_joint_preprocess.sh
#
# Inline env also works:
#   INPUT_PATH_X=... INPUT_PATH_Y=... OUTPUT_PATH=... \
#     sbatch --chdir=$PWD scripts/preprocess_joint.sbatch
#
# Do NOT use multiline sbatch --export=ALL,\ — bash splits the list and SLURM
# never sees the vars.

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
cd "${BRAVE_ROOT}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

export INPUT_PATH_X="${INPUT_PATH_X:-${BRAVE_STORAGE}/tap_samples/audio_subset_clean}"
export INPUT_PATH_Y="${INPUT_PATH_Y:-${BRAVE_STORAGE}/yt_birdsong/audio_subset}"
export OUTPUT_PATH="${OUTPUT_PATH:-${BRAVE_STORAGE}/joint_tap_contact_birdsong/preprocessed_normalized}"
export NORMALIZE="${NORMALIZE:-1}"
export PCEN="${PCEN:-0}"
export CHANNELS="${CHANNELS:-1}"

mkdir -p logs

echo "Submitting joint preprocess:"
echo "  INPUT_PATH_X=${INPUT_PATH_X}"
echo "  INPUT_PATH_Y=${INPUT_PATH_Y}"
echo "  OUTPUT_PATH=${OUTPUT_PATH}"
echo "  NORMALIZE=${NORMALIZE} PCEN=${PCEN}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "ERROR: sbatch not found. Run this on the cluster login node." >&2
  exit 1
fi

job_id="$(
  sbatch --parsable \
    --chdir="${BRAVE_ROOT}" \
    --job-name=joint-prep \
    --output="${BRAVE_ROOT}/logs/preprocess-%j.log" \
    --error="${BRAVE_ROOT}/logs/preprocess-%j.log" \
    "$@" \
    scripts/preprocess_joint.sbatch
)" || {
  echo "ERROR: sbatch failed (no job submitted — check message above)." >&2
  exit 1
}

echo ""
echo "Submitted job ${job_id}"
echo "Log file: ${BRAVE_ROOT}/logs/preprocess-${job_id}.log"
echo "Watch:    tail -f ${BRAVE_ROOT}/logs/preprocess-${job_id}.log"
echo ""
echo "When done, train with:"
echo "  DB_PATH=${OUTPUT_PATH} \\"
echo "  RUN_NAME=joint_tap_contact_birdsong_uncond \\"
echo "  CONFIG=configs/brave_birdsong.gin \\"
echo "  OUT_PATH=${BRAVE_STORAGE}/joint_tap_contact_birdsong/runs \\"
echo "  sbatch --chdir=${BRAVE_ROOT} --gres=gpu:4 --cpus-per-task=32 --mem=128G scripts/train.sbatch"
