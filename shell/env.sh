# Shared environment setup for cluster jobs. Sourced by the job scripts in this folder.
#
# Optional settings (set in your shell or pass with sbatch --export):
#   MODULES   space-separated environment modules to load, e.g. "python/3.11 cuda/12.1"
#   VENV_DIR  path to a virtualenv to activate
# If neither is set, the python on PATH is used.

if [ -n "${MODULES:-}" ] && command -v module >/dev/null 2>&1; then
    for mod in $MODULES; do
        module load "$mod"
    done
fi

if [ -n "${VENV_DIR:-}" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
