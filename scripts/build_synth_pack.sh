#!/usr/bin/env bash
# Build foundation-Y synth pack = FSD50K tonal/electronic tags + generated abstract audio.
#
# Layout (under $BRAVE_STORAGE/synth_abstract/):
#   fsd50k_subset/     staged FSD50K WAVs (via stage_named_subset OUT_ROOT)
#   generated/         generate_pack.py output
#   audio_subset/      flat merge (symlinks) for preprocess
#   preprocessed/      LMDB (if SUBMIT_PREPROCESS=1)
#
# Usage (login node, from repo root):
#   bash scripts/build_synth_pack.sh
#   HOURS=6 SUBMIT_PREPROCESS=1 bash scripts/build_synth_pack.sh
#
# Env:
#   HOURS              generated hours (default 4)
#   SEED               generation seed (default 42)
#   METHOD             FSD50K staging symlink|copy (default symlink)
#   WORKERS            FSD staging workers (default 8)
#   OVERWRITE          1 to restage / regenerate / re-merge
#   SKIP_FSD           1 skip FSD50K staging
#   SKIP_GENERATE      1 skip generation
#   SUBMIT_PREPROCESS  1 sbatch preprocess after merge

set -euo pipefail

BRAVE_ROOT="${BRAVE_ROOT:-/data/hai-res/p1long/BRAVE}"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/load_env.sh"
# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/env.sh"

HOURS="${HOURS:-4}"
SEED="${SEED:-42}"
METHOD="${METHOD:-symlink}"
WORKERS="${WORKERS:-8}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_FSD="${SKIP_FSD:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SUBMIT_PREPROCESS="${SUBMIT_PREPROCESS:-0}"

PACK_ROOT="${BRAVE_STORAGE}/synth_abstract"
FSD_ROOT="${PACK_ROOT}/fsd50k_subset"
FSD_AUDIO="${FSD_ROOT}/audio_subset"
GENERATED="${PACK_ROOT}/generated"
AUDIO_SUBSET="${PACK_ROOT}/audio_subset"
PREPROCESSED="${PACK_ROOT}/preprocessed"
GEN_PY="${BRAVE_ROOT}/dataset_exploration/synth_abstract/generate_pack.py"

mkdir -p "${PACK_ROOT}" "${AUDIO_SUBSET}"

if [[ "${SKIP_FSD}" != "1" ]]; then
  echo "=== [1/3] stage FSD50K synth_tonal → ${FSD_AUDIO} ==="
  NAME=synth_tonal \
    OUT_ROOT="${FSD_ROOT}" \
    METHOD="${METHOD}" \
    WORKERS="${WORKERS}" \
    OVERWRITE="${OVERWRITE}" \
    PARTITION="${PARTITION:-train,valid}" \
    SUBMIT_PREPROCESS=0 \
    bash "${BRAVE_ROOT}/dataset_exploration/fsd50k/stage_named_subset.sh" synth_tonal
else
  echo "=== [1/3] skip FSD50K staging ==="
fi

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  echo "=== [2/3] generate abstract pack → ${GENERATED} (${HOURS} h) ==="
  if [[ "${OVERWRITE}" == "1" && -d "${GENERATED}" ]]; then
    find "${GENERATED}" -maxdepth 1 -name 'gen_*.wav' -delete
  fi
  python3 "${GEN_PY}" --output-dir "${GENERATED}" --hours "${HOURS}" --seed "${SEED}"
else
  echo "=== [2/3] skip generation ==="
fi

echo "=== [3/3] merge into ${AUDIO_SUBSET} ==="
if [[ "${OVERWRITE}" == "1" ]]; then
  find "${AUDIO_SUBSET}" -maxdepth 1 \( -type l -o -name '*.wav' \) -delete 2>/dev/null || true
fi
shopt -s nullglob
for wav in "${FSD_AUDIO}"/*.wav; do
  base="$(basename "${wav}")"
  dst="${AUDIO_SUBSET}/fsd_${base}"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    [[ "${OVERWRITE}" == "1" ]] || continue
    rm -f "${dst}"
  fi
  ln -s "$(readlink -f "${wav}")" "${dst}"
done
for wav in "${GENERATED}"/*.wav; do
  base="$(basename "${wav}")"
  dst="${AUDIO_SUBSET}/${base}"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    [[ "${OVERWRITE}" == "1" ]] || continue
    rm -f "${dst}"
  fi
  ln -s "$(readlink -f "${wav}")" "${dst}"
done
shopt -u nullglob

n_wav="$(find "${AUDIO_SUBSET}" -maxdepth 1 -name '*.wav' | wc -l | tr -d ' ')"
echo "Merged ${n_wav} wav entries under ${AUDIO_SUBSET}"
echo ""
echo "Preprocess:"
echo "  INPUT_PATH=${AUDIO_SUBSET} OUTPUT_PATH=${PREPROCESSED} \\"
echo "    sbatch ${BRAVE_ROOT}/scripts/preprocess.sbatch"

if [[ "${SUBMIT_PREPROCESS}" == "1" ]]; then
  mkdir -p "${BRAVE_ROOT}/logs"
  job_id="$(
    INPUT_PATH="${AUDIO_SUBSET}" OUTPUT_PATH="${PREPROCESSED}" \
      sbatch --parsable "${BRAVE_ROOT}/scripts/preprocess.sbatch"
  )"
  echo "Submitted preprocess job ${job_id}"
fi
