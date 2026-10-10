#!/bin/bash
#SBATCH --job-name=gapo_long
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#
# Long training run with a larger iteration budget. Submit from the repository root:
#   sbatch shell/train_long.sh
# Override the defaults below with environment variables, for example:
#   CONFIG=configs/other.json ITERATIONS=100000 sbatch shell/train_long.sh
# Scheduler options can be overridden at submission, for example:
#   sbatch --time=48:00:00 shell/train_long.sh

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/ilc_pilot_v1_care_robotics.json}"
ITERATIONS="${ITERATIONS:-50000}"
LR="${LR:-0.0003}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/gapo_outputs}"

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
# shellcheck disable=SC1091
source shell/env.sh

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

echo "Job ${SLURM_JOB_ID:-local} on ${HOSTNAME}, started $(date)"
echo "Config: $CONFIG | iterations: $ITERATIONS | lr: $LR | hidden: $HIDDEN_DIM"

python run_training.py \
    --config "$CONFIG" \
    --iterations "$ITERATIONS" \
    --lr "$LR" \
    --hidden-dim "$HIDDEN_DIM" \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --exp-name "long_job_${SLURM_JOB_ID:-local}" \
    --save-interval "${SAVE_INTERVAL:-1000}" \
    --log-interval "${LOG_INTERVAL:-100}" \
    --seed "$SEED"
EXIT_CODE=$?

echo "Finished $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
