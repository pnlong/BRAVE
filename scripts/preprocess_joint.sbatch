#!/usr/bin/env bash
#SBATCH --job-name=brave-prep-joint
#SBATCH --output=logs/preprocess-%j.log
#SBATCH --error=logs/preprocess-%j.log
#SBATCH --account=hai-res
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --export=ALL

# Joint LMDB preprocess: two (or more) audio roots → one output LMDB.
# Prefer scripts/submit_joint_preprocess.sh so env vars reach the job.
#
# Required:
#   INPUT_PATH_X, INPUT_PATH_Y, OUTPUT_PATH
# Optional extra roots (space-separated):
#   INPUT_PATH_EXTRA="/path/a /path/b"
# Flags: NORMALIZE=1 (default), PCEN=0|1, CHANNELS, SAMPLING_RATE, WORKERS, …

set -euo pipefail

: "${INPUT_PATH_X:?Set INPUT_PATH_X to first domain audio root}"
: "${INPUT_PATH_Y:?Set INPUT_PATH_Y to second domain audio root}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to joint LMDB directory}"

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

_fix_empty_storage_path() {
  local p="$1"
  if [[ "${p}" == /tap_samples/* || "${p}" == /yt_birdsong/* ]]; then
    echo "${BRAVE_STORAGE}${p}"
  else
    echo "${p}"
  fi
}

INPUT_PATH_X="$(_fix_empty_storage_path "${INPUT_PATH_X}")"
INPUT_PATH_Y="$(_fix_empty_storage_path "${INPUT_PATH_Y}")"
OUTPUT_PATH="$(_fix_empty_storage_path "${OUTPUT_PATH}")"
unset -f _fix_empty_storage_path

CHANNELS="${CHANNELS:-1}"
SAMPLING_RATE="${SAMPLING_RATE:-44100}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
SUBSET_SEED="${SUBSET_SEED:-42}"
CONCAT_SEED="${CONCAT_SEED:-42}"
# Match prior contact + birdsong LMDBs (both used normalize).
NORMALIZE="${NORMALIZE:-1}"
PCEN="${PCEN:-0}"

PREPROCESS_ARGS=(
  --input_path="${INPUT_PATH_X}"
  --input_path="${INPUT_PATH_Y}"
  --output_path="${OUTPUT_PATH}"
  --channels="${CHANNELS}"
  --sampling_rate="${SAMPLING_RATE}"
  --workers="${WORKERS}"
  --subset_seed="${SUBSET_SEED}"
  --concat_seed="${CONCAT_SEED}"
)

if [[ -n "${INPUT_PATH_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  _extra=( ${INPUT_PATH_EXTRA} )
  for _p in "${_extra[@]}"; do
    PREPROCESS_ARGS+=(--input_path="${_p}")
  done
fi

if [[ -n "${MAX_STORED_SEC:-}" ]]; then
  PREPROCESS_ARGS+=(--max_stored_sec="${MAX_STORED_SEC}")
fi
if [[ "${PCEN}" == "1" ]]; then
  PREPROCESS_ARGS+=(--pcen)
fi
if [[ "${NORMALIZE}" == "1" ]]; then
  PREPROCESS_ARGS+=(--normalize)
fi
if [[ -n "${NORMALIZE_MAX_GAIN_DB:-}" ]]; then
  PREPROCESS_ARGS+=(--normalize_max_gain_db="${NORMALIZE_MAX_GAIN_DB}")
fi

mkdir -p logs "$(dirname "${OUTPUT_PATH}")"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

cd "${BRAVE_ROOT}"

echo "========== BRAVE preprocess_joint.sbatch (job ${SLURM_JOB_ID:-local}) =========="
echo "INPUT_PATH_X=${INPUT_PATH_X}"
echo "INPUT_PATH_Y=${INPUT_PATH_Y}"
if [[ -n "${INPUT_PATH_EXTRA:-}" ]]; then
  echo "INPUT_PATH_EXTRA=${INPUT_PATH_EXTRA}"
fi
echo "OUTPUT_PATH=${OUTPUT_PATH}"
echo "NORMALIZE=${NORMALIZE} PCEN=${PCEN}"

python RAVE/scripts/preprocess.py "${PREPROCESS_ARGS[@]}"

python scripts/lmdb_duration.py "${OUTPUT_PATH}"
