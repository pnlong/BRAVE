# Source after: micromamba activate brave
#   source /path/to/BRAVE/scripts/env.sh
#
# Adds vendored RAVE to PYTHONPATH for train/preprocess scripts.

if [[ -n "${BASH_SOURCE[0]:-}" ]] && [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  _BRAVE_ENV_SH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  _BRAVE_ENV_SH_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

export BRAVE_ROOT="${BRAVE_ROOT:-${_BRAVE_ENV_SH_DIR}}"
export PYTHONPATH="${BRAVE_ROOT}/RAVE${PYTHONPATH:+:${PYTHONPATH}}"

# Optional: split datasets vs checkpoints across data drives (set in ~/.bashrc or before sourcing).
#   export BRAVE_DATA=/mnt/datasets/brave      # LMDBs, raw audio, preprocess output
#   export BRAVE_RUNS=/mnt/checkpoints/brave   # training runs / .ckpt trees
# Training still needs --db_path and --out_path on the CLI; BRAVE_RUNS is a convenience default.
if [[ -n "${BRAVE_DATA:-}" ]]; then
  export BRAVE_DATA
  export BRAVE_STORAGE="${BRAVE_STORAGE:-$BRAVE_DATA}"
elif [[ -n "${BRAVE_STORAGE:-}" ]]; then
  export BRAVE_STORAGE
  export BRAVE_DATA="$BRAVE_STORAGE"
fi
export BRAVE_RUNS="${BRAVE_RUNS:-${BRAVE_ROOT}/runs}"

unset _BRAVE_ENV_SH_DIR
