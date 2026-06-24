#!/usr/bin/env bash
# Create or update the BRAVE conda/mamba environment on a new machine.
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
#   conda activate brave
#   source scripts/env.sh
#   wandb login   # optional, for training logs

set -euo pipefail

BRAVE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="brave"
CUDA_TARGET="auto"
DO_UPDATE=0
SKIP_TORCH=0
INSTALL_EVAL=0

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "Options:"
  echo "  --env-name NAME     Conda env name (default: brave)"
  echo "  --cuda auto|cpu|11.8|12.1|12.4   PyTorch CUDA wheels (default: auto)"
  echo "  --cpu               Same as --cuda cpu"
  echo "  --update            conda env update instead of create"
  echo "  --skip-torch        Do not reinstall torch/torchaudio"
  echo "  --eval-deps         Install FAD + neural-latency-eval (paper eval only)"
  echo "  -h, --help          Show this help"
}

log() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --cuda) CUDA_TARGET="$2"; shift 2 ;;
    --cpu) CUDA_TARGET="cpu"; shift ;;
    --update) DO_UPDATE=1; shift ;;
    --skip-torch) SKIP_TORCH=1; shift ;;
    --eval-deps) INSTALL_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

pick_conda() {
  if command -v mamba >/dev/null 2>&1; then
    echo mamba
  elif command -v conda >/dev/null 2>&1; then
    echo conda
  else
    die "neither mamba nor conda found on PATH"
  fi
}

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

CONDA="$(pick_conda)"
log "using ${CONDA} (env: ${ENV_NAME})"
log "BRAVE_ROOT=${BRAVE_ROOT}"

[[ -f "${BRAVE_ROOT}/environment.yaml" ]] || die "environment.yaml not found under ${BRAVE_ROOT}"
[[ -f "${BRAVE_ROOT}/RAVE/requirements.txt" ]] || die "RAVE/requirements.txt not found"

if [[ "${DO_UPDATE}" -eq 1 ]]; then
  log "updating conda env from environment.yaml"
  "${CONDA}" env update -n "${ENV_NAME}" -f "${BRAVE_ROOT}/environment.yaml" --prune -y
else
  if "${CONDA}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    die "env '${ENV_NAME}' already exists; use --update or remove it first"
  fi
  log "creating conda env from environment.yaml"
  "${CONDA}" env create -f "${BRAVE_ROOT}/environment.yaml" -n "${ENV_NAME}" -y
fi

# shellcheck disable=SC1091
source "$("${CONDA}" info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

RESOLVED_CUDA="$(detect_cuda)"
INDEX="$(pytorch_index "${RESOLVED_CUDA}")"

if [[ "${SKIP_TORCH}" -eq 0 ]]; then
  log "installing torch + torchaudio from ${INDEX}"
  python -m pip install --upgrade pip
  python -m pip install --upgrade torch torchaudio --index-url "${INDEX}"
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
  conda activate ${ENV_NAME}
  source ${BRAVE_ROOT}/scripts/env.sh

Optional:
  wandb login

Quick smoke test (from repo root, with env active + env.sh sourced):
  python -m pytest RAVE/tests/test_preprocess_pcen.py -q

Training example:
  export PYTHONPATH="\${PWD}/RAVE:\${PYTHONPATH}"
  python RAVE/scripts/train.py --config=configs/brave.gin --name=my_run --db_path=/path/to/lmdb

EOF
