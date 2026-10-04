#!/bin/bash
#SBATCH --job-name=gapo_train
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# Single training run. Submit from the repository root:
#   sbatch shell/train.sh
# Override the defaults below with environment variables, for example:
#   CONFIG=configs/other.json ITERATIONS=20000 sbatch shell/train.sh
# Scheduler options can be overridden at submission, for example:
#   sbatch --partition=<name> shell/train.sh

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/hospital_3nodes_consumables.json}"
ITERATIONS="${ITERATIONS:-10000}"
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
echo "Config: $CONFIG | iterations: $ITERATIONS | seed: $SEED"
echo "Output: $OUTPUT_DIR/job_${SLURM_JOB_ID:-local}"

python run_training.py \
    --config "$CONFIG" \
    --iterations "$ITERATIONS" \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --exp-name "job_${SLURM_JOB_ID:-local}" \
    --save-interval "${SAVE_INTERVAL:-500}" \
    --log-interval "${LOG_INTERVAL:-50}" \
    --seed "$SEED"
EXIT_CODE=$?

echo "Finished $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
