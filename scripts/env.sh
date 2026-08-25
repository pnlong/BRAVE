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

# Bulk data root: LMDBs, raw audio, preprocess output, training runs on shared storage.
# Override in ~/.bashrc or BRAVE_ROOT/.env before sourcing.
#   export BRAVE_STORAGE=/data/hai-res/$USER/BRAVE-data
# Optional split for checkpoints only:
#   export BRAVE_RUNS=/path/to/checkpoints
# Training still needs --db_path and --out_path on the CLI; BRAVE_RUNS is a convenience default.
_BRAVE_STORAGE_DEFAULT="/data/hai-res/${USER}/BRAVE-data"
if [[ -n "${BRAVE_DATA:-}" ]]; then
  export BRAVE_DATA
  export BRAVE_STORAGE="${BRAVE_STORAGE:-$BRAVE_DATA}"
elif [[ -n "${BRAVE_STORAGE:-}" ]]; then
  export BRAVE_STORAGE
  export BRAVE_DATA="$BRAVE_STORAGE"
else
  export BRAVE_STORAGE="${_BRAVE_STORAGE_DEFAULT}"
  export BRAVE_DATA="${BRAVE_STORAGE}"
fi
unset _BRAVE_STORAGE_DEFAULT
export BRAVE_RUNS="${BRAVE_RUNS:-${BRAVE_ROOT}/runs}"
# Legacy alias (scripts migrated to BRAVE_STORAGE).
export SCRATCH="${SCRATCH:-$BRAVE_STORAGE}"

unset _BRAVE_ENV_SH_DIR
