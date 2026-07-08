#!/usr/bin/env bash
#SBATCH --job-name=brave-preprocess
#SBATCH --output=logs/preprocess-%j.log
#SBATCH --error=logs/preprocess-%j.log
#SBATCH --account=hai-res
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --export=ALL

set -euo pipefail

: "${INPUT_PATH:?Set INPUT_PATH to raw audio root (directory searched recursively)}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH to LMDB output directory}"

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"

CHANNELS="${CHANNELS:-1}"
SAMPLING_RATE="${SAMPLING_RATE:-44100}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
SUBSET_SEED="${SUBSET_SEED:-42}"
CONCAT_SEED="${CONCAT_SEED:-42}"

PREPROCESS_ARGS=(
  --input_path="${INPUT_PATH}"
  --output_path="${OUTPUT_PATH}"
  --channels="${CHANNELS}"
  --sampling_rate="${SAMPLING_RATE}"
  --workers="${WORKERS}"
  --subset_seed="${SUBSET_SEED}"
  --concat_seed="${CONCAT_SEED}"
)

if [[ -n "${MAX_STORED_SEC:-}" ]]; then
  PREPROCESS_ARGS+=(--max_stored_sec="${MAX_STORED_SEC}")
fi
if [[ "${PCEN:-0}" == "1" ]]; then
  PREPROCESS_ARGS+=(--pcen)
fi
if [[ "${NORMALIZE:-0}" == "1" ]]; then
  PREPROCESS_ARGS+=(--normalize)
fi
if [[ -n "${NORMALIZE_MAX_GAIN_DB:-}" ]]; then
  PREPROCESS_ARGS+=(--normalize_max_gain_db="${NORMALIZE_MAX_GAIN_DB}")
fi

mkdir -p logs "$(dirname "${OUTPUT_PATH}")"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

cd "${BRAVE_ROOT}"

echo "INPUT_PATH=${INPUT_PATH}"
echo "OUTPUT_PATH=${OUTPUT_PATH}"
if [[ -n "${MAX_STORED_SEC:-}" ]]; then
  echo "MAX_STORED_SEC=${MAX_STORED_SEC}"
fi
if [[ "${PCEN:-0}" == "1" ]]; then
  echo "PCEN=1"
fi
if [[ "${NORMALIZE:-0}" == "1" ]]; then
  echo "NORMALIZE=1"
fi

python RAVE/scripts/preprocess.py "${PREPROCESS_ARGS[@]}"

python scripts/lmdb_duration.py "${OUTPUT_PATH}"
