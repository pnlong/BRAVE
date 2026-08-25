#!/usr/bin/env bash
#SBATCH --job-name=pres-demos
#SBATCH --output=logs/presentation-demos-%j.log
#SBATCH --error=logs/presentation-demos-%j.log
#SBATCH --account=hai-res
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Presentation phase-1/2/3 demos + Y-latent PCA for all yt_playlists.
#
#   sbatch --chdir=/data/hai-res/p1long/BRAVE scripts/presentation_demos.sbatch
#
# Optional env overrides:
#   DOMAINS="rain_sounds waves_crashing"
#   OUT_DIR=$BRAVE_STORAGE/presentation_demos
#   MAX_SEC=15 SEGMENT_SEC=3
#   CYCLEGAN_HASH=48072c314a

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/slurm_env.sh"

cd "${BRAVE_ROOT}"
mkdir -p logs

OUT_DIR="${OUT_DIR:-${BRAVE_STORAGE}/presentation_demos}"
TAP_DIR="${TAP_DIR:-${BRAVE_STORAGE}/tap_samples/audio_subset}"
PLAYLISTS_ROOT="${PLAYLISTS_ROOT:-${BRAVE_STORAGE}/yt_playlists}"
TAP_BACKBONE="${TAP_BACKBONE:-${BRAVE_STORAGE}/tap_samples/runs/tap_uncond_run_8e1e614287/epoch_1500000.ckpt}"
CYCLEGAN_HASH="${CYCLEGAN_HASH:-48072c314a}"
MAX_SEC="${MAX_SEC:-15}"
SEGMENT_SEC="${SEGMENT_SEC:-3}"
SILENCE_SEC="${SILENCE_SEC:-0}"

DOMAIN_ARGS=()
if [[ -n "${DOMAINS:-}" ]]; then
  # shellcheck disable=SC2206
  DOMAIN_ARGS=(--domains ${DOMAINS})
fi

echo "OUT_DIR=${OUT_DIR}"
echo "TAP_DIR=${TAP_DIR}"
echo "PLAYLISTS_ROOT=${PLAYLISTS_ROOT}"
echo "CYCLEGAN_HASH=${CYCLEGAN_HASH}"
echo "MAX_SEC=${MAX_SEC} SEGMENT_SEC=${SEGMENT_SEC}"
echo "DOMAINS=${DOMAINS:-"(all default)"}"

python scripts/presentation_phase_demos.py \
  --gpu \
  --out-dir "${OUT_DIR}" \
  --tap-dir "${TAP_DIR}" \
  --playlists-root "${PLAYLISTS_ROOT}" \
  --tap-backbone "${TAP_BACKBONE}" \
  --cyclegan-hash "${CYCLEGAN_HASH}" \
  --max-sec "${MAX_SEC}" \
  --segment-sec "${SEGMENT_SEC}" \
  --silence-sec "${SILENCE_SEC}" \
  "${DOMAIN_ARGS[@]}"

echo "Done. Outputs: ${OUT_DIR}"
