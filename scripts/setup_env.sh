#!/usr/bin/env bash
# Create or update the BRAVE micromamba environment.
#
# Usage (from BRAVE repo root):
#   ./scripts/setup_env.sh
#   ./scripts/setup_env.sh --cuda 12.4
#   ./scripts/setup_env.sh --cuda auto
#   ./scripts/setup_env.sh --cpu
#   ./scripts/setup_env.sh --update
#   ./scripts/setup_env.sh --eval-deps
#
# After setup:
#   micromamba activate brave
#   source scripts/env.sh
#   wandb login   # optional, for training logs

set -euo pipefail

BRAVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="brave"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/data/hai-res/${USER}/micromamba}"
CUDA_TARGET="auto"
DO_UPDATE=0
FORCE_RECREATE=0
SKIP_TORCH=0
INSTALL_EVAL=0

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Options:"
  echo "  --env-name NAME     Env name (default: brave)"
  echo "  --prefix PATH       Micromamba root (default: /data/hai-res/\$USER/micromamba)"
  echo "  --cuda auto|cpu|11.8|12.1|12.4   PyTorch CUDA wheels (default: auto)"
  echo "  --cpu               Same as --cuda cpu"
  echo "  --update            micromamba env update instead of create"
  echo "  --force-recreate    Remove existing env and create fresh"
  echo "  --skip-torch        Do not reinstall torch/torchaudio"
  echo "  --eval-deps         Install FAD + neural-latency-eval (paper eval only)"
  echo "  -h, --help          Show this help"
}

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

detect_cuda() {
  if [[ "${CUDA_TARGET}" != "auto" ]]; then
    echo "${CUDA_TARGET}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "no nvidia-smi; using CPU PyTorch"
    echo cpu
    return
  fi
  local ver
  ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1)"
  case "${ver}" in
    12.4|12.5|12.6|12.8|12.9) echo 12.4 ;;
    12.1|12.2|12.3) echo 12.1 ;;
    11.8|11.7|11.6) echo 11.8 ;;
    "")
      log "nvidia-smi present but CUDA version not parsed; defaulting to cu121"
      echo 12.1
      ;;
    *)
      log "driver reports CUDA ${ver}; defaulting pip wheels to cu124"
      echo 12.4
      ;;
  esac
}

pytorch_index() {
  case "$1" in
    cpu) echo "https://download.pytorch.org/whl/cpu" ;;
    11.8|cu118) echo "https://download.pytorch.org/whl/cu118" ;;
    12.1|cu121) echo "https://download.pytorch.org/whl/cu121" ;;
    12.4|cu124) echo "https://download.pytorch.org/whl/cu124" ;;
    *) die "unsupported --cuda value: $1 (use auto, cpu, 11.8, 12.1, or 12.4)" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --prefix) MAMBA_ROOT_PREFIX="$2"; shift 2 ;;
    --cuda) CUDA_TARGET="$2"; shift 2 ;;
    --cpu) CUDA_TARGET="cpu"; shift ;;
    --update) DO_UPDATE=1; shift ;;
    --force-recreate) FORCE_RECREATE=1; shift ;;
    --skip-torch) SKIP_TORCH=1; shift ;;
    --eval-deps) INSTALL_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

export MAMBA_ROOT_PREFIX

# shellcheck disable=SC1091
source "${BRAVE_ROOT}/scripts/micromamba_env.sh"

log "using micromamba (${MAMBA_EXE}, root: ${MAMBA_ROOT_PREFIX}, env: ${ENV_NAME})"
log "BRAVE_ROOT=${BRAVE_ROOT}"

[[ -f "${BRAVE_ROOT}/environment.yaml" ]] || die "environment.yaml not found under ${BRAVE_ROOT}"
[[ -f "${BRAVE_ROOT}/RAVE/requirements.txt" ]] || die "RAVE/requirements.txt not found"

ENV_PATH="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"

_is_valid_conda_prefix() {
  [[ -f "$1/conda-meta/history" && -x "$1/bin/python" ]]
}

_bootstrap_micromamba_base() {
  if _is_valid_conda_prefix "${MAMBA_ROOT_PREFIX}"; then
    return 0
  fi
  log "bootstrapping micromamba base at ${MAMBA_ROOT_PREFIX} (shell init alone is not enough)"
  micromamba install -n base -c conda-forge python -y
}

_remove_env_path() {
  local path="$1"
  if _is_valid_conda_prefix "${path}"; then
    micromamba env remove -p "${path}" -y
    return 0
  fi
  log "env broken or missing python; deleting ${path} directly"
  rm -rf "${path}"
  local envs_file
  for envs_file in \
    "/data/hai-res/${USER}/.conda/environments.txt" \
    "${HOME}/.conda/environments.txt"; do
    [[ -f "${envs_file}" ]] || continue
    grep -vxF "${path}" "${envs_file}" > "${envs_file}.tmp" && mv "${envs_file}.tmp" "${envs_file}"
  done
}

_bootstrap_micromamba_base

if [[ "${FORCE_RECREATE}" -eq 1 && -d "${ENV_PATH}" ]]; then
  log "removing existing env at ${ENV_PATH}"
  _remove_env_path "${ENV_PATH}"
fi

if [[ "${DO_UPDATE}" -eq 1 ]]; then
  if [[ ! -d "${ENV_PATH}" ]]; then
    die "env '${ENV_NAME}' not found at ${ENV_PATH}; run without --update"
  fi
  if ! _is_valid_conda_prefix "${ENV_PATH}"; then
    die "env at ${ENV_PATH} is broken (no bin/python); run: ./scripts/setup_env.sh --force-recreate"
  fi
  log "updating env at ${ENV_PATH}"
  (cd "${BRAVE_ROOT}" && micromamba env update -p "${ENV_PATH}" -c conda-forge --override-channels -f environment.yaml -y)
elif [[ -d "${ENV_PATH}" ]]; then
  if _is_valid_conda_prefix "${ENV_PATH}"; then
    die "env '${ENV_NAME}' already exists at ${ENV_PATH}; use --update or --force-recreate"
  fi
  log "removing broken/incomplete env at ${ENV_PATH}"
  _remove_env_path "${ENV_PATH}"
  log "creating env at ${ENV_PATH}"
  (cd "${BRAVE_ROOT}" && micromamba create -p "${ENV_PATH}" -c conda-forge --override-channels -f environment.yaml -y)
else
  log "creating env at ${ENV_PATH}"
  (cd "${BRAVE_ROOT}" && micromamba create -p "${ENV_PATH}" -c conda-forge --override-channels -f environment.yaml -y)
fi

micromamba activate "${ENV_PATH}"

RESOLVED_CUDA="$(detect_cuda)"
INDEX="$(pytorch_index "${RESOLVED_CUDA}")"

if [[ "${SKIP_TORCH}" -eq 0 ]]; then
  log "installing torch==2.5.0 + torchaudio==2.5.0 from ${INDEX}"
  python -m pip install --upgrade pip
  python -m pip install --force-reinstall \
    "torch==2.5.0" "torchaudio==2.5.0" --index-url "${INDEX}"
else
  log "skipping torch/torchaudio reinstall (--skip-torch)"
fi

if [[ "${INSTALL_EVAL}" -eq 1 ]]; then
  log "installing optional evaluation dependencies"
  python -m pip install frechet_audio_distance
  python -m pip install "git+https://github.com/jorshi/neural-latency-eval"
fi

log "verifying imports"
PYTHONPATH="${BRAVE_ROOT}/RAVE${PYTHONPATH:+:${PYTHONPATH}}" python - <<'PY'
import sys
mods = [
    "torch",
    "torchaudio",
    "pytorch_lightning",
    "gin",
    "librosa",
    "cached_conv",
    "wandb",
    "rave",
]
failed = []
for name in mods:
    try:
        __import__(name)
    except Exception as exc:
        failed.append(f"{name}: {exc}")
if failed:
    print("import check FAILED:", file=sys.stderr)
    for line in failed:
        print(f"  - {line}", file=sys.stderr)
    sys.exit(1)
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
PY

log "setup complete"
cat <<EOF

Next steps:
  micromamba activate ${ENV_NAME}
  source ${BRAVE_ROOT}/scripts/env.sh

SLURM jobs source scripts/slurm_env.sh automatically (same micromamba root).

Optional:
  wandb login

Quick smoke test:
  python -m pytest RAVE/tests/test_preprocess_pcen.py -q

Training example:
  python RAVE/scripts/train.py --config=configs/brave.gin --name=my_run --db_path=/path/to/lmdb

EOF
