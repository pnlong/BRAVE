#!/usr/bin/env bash
#SBATCH --job-name=brave-preprocess-fader
#SBATCH --output=logs/preprocess-fader-%j.log
#SBATCH --error=logs/preprocess-fader-%j.log
#SBATCH --account=hai-res
#SBATCH --partition=hai-res-l40s
#SBATCH --qos=hai-res-main
#SBATCH --time=23:59:59
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --export=ALL

set -euo pipefail

: "${INPUT_PATH:?Set INPUT_PATH to raw audio root (unless SKIP_PREPROCESS=1)}"
: "${DB_PATH:?Set DB_PATH to LMDB output directory}"

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"

CONFIG="${CONFIG:-configs/brave_fader_birdsong.gin}"
CHANNELS="${CHANNELS:-1}"
SAMPLING_RATE="${SAMPLING_RATE:-44100}"
NUM_SIGNAL="${NUM_SIGNAL:-131072}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
CONCAT_SEED="${CONCAT_SEED:-42}"
SUBSET_SEED="${SUBSET_SEED:-42}"

FADER_ARGS=(
  --input_path="${INPUT_PATH}"
  --db_path="${DB_PATH}"
  --config="${CONFIG}"
  --channels="${CHANNELS}"
  --sampling_rate="${SAMPLING_RATE}"
  --num_signal="${NUM_SIGNAL}"
  --workers="${WORKERS}"
  --concat_seed="${CONCAT_SEED}"
  --subset_seed="${SUBSET_SEED}"
)

if [[ -n "${MAX_STORED_SEC:-}" ]]; then
  FADER_ARGS+=(--max_stored_sec="${MAX_STORED_SEC}")
fi
if [[ "${PCEN:-0}" == "1" ]]; then
  FADER_ARGS+=(--pcen)
fi
if [[ "${NORMALIZE:-0}" == "1" ]]; then
  FADER_ARGS+=(--normalize)
fi
if [[ "${DENOISE:-0}" == "1" ]]; then
  FADER_ARGS+=(--denoise)
fi
if [[ "${SKIP_PREPROCESS:-0}" == "1" ]]; then
  FADER_ARGS+=(--skip_preprocess)
fi
if [[ "${SKIP_MANIFEST:-0}" == "1" ]]; then
  FADER_ARGS+=(--skip_manifest)
fi
if [[ "${SKIP_PRECOMPUTE:-0}" == "1" ]]; then
  FADER_ARGS+=(--skip_precompute)
fi
if [[ -n "${SIDECAR_SCHEME:-}" ]]; then
  FADER_ARGS+=(--sidecar_scheme="${SIDECAR_SCHEME}")
fi

mkdir -p logs "$(dirname "${DB_PATH}")"

source "${BRAVE_ROOT}/scripts/slurm_env.sh"

cd "${BRAVE_ROOT}"

echo "INPUT_PATH=${INPUT_PATH}"
echo "DB_PATH=${DB_PATH}"
echo "CONFIG=${CONFIG}"
if [[ -n "${MAX_STORED_SEC:-}" ]]; then
  echo "MAX_STORED_SEC=${MAX_STORED_SEC}"
fi

python RAVE/scripts/preprocess_fader.py "${FADER_ARGS[@]}"

python scripts/lmdb_duration.py "${DB_PATH}"
test -f "${DB_PATH}/attribute_stats.yaml" && echo "attribute_stats.yaml: ok"
