#!/bin/bash
#SBATCH --job-name=gapo_sweep
#SBATCH --output=slurm_sweep_%A_%a.out
#SBATCH --error=slurm_sweep_%A_%a.err
#SBATCH --array=0-8
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#
# Hyperparameter sweep: one job per (learning rate, hidden dim) pair.
# The array size must equal len(LRS) * len(HIDDEN_DIMS). Submit from the repository root:
#   sbatch shell/sweep.sh
# Override the defaults below with environment variables, for example:
#   CONFIG=configs/other.json ITERATIONS=20000 sbatch shell/sweep.sh

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONFIG="${CONFIG:-configs/ilc_pilot_v1_care_robotics.json}"
ITERATIONS="${ITERATIONS:-10000}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/gapo_sweep}"

LRS=(0.0001 0.0003 0.001)
HIDDEN_DIMS=(32 64 128)

N_LR=${#LRS[@]}
LR=${LRS[$((SLURM_ARRAY_TASK_ID % N_LR))]}
HIDDEN=${HIDDEN_DIMS[$((SLURM_ARRAY_TASK_ID / N_LR))]}

cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 1; }
# shellcheck disable=SC1091
source shell/env.sh

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

echo "Sweep task ${SLURM_ARRAY_TASK_ID:-0} on ${HOSTNAME}, started $(date)"
echo "lr: $LR | hidden: $HIDDEN | iterations: $ITERATIONS"

python run_training.py \
    --config "$CONFIG" \
    --iterations "$ITERATIONS" \
    --lr "$LR" \
    --hidden-dim "$HIDDEN" \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --exp-name "sweep_lr${LR}_hd${HIDDEN}" \
    --save-interval "${SAVE_INTERVAL:-500}" \
    --log-interval "${LOG_INTERVAL:-50}" \
    --seed "$SEED"
EXIT_CODE=$?

echo "Finished $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE
