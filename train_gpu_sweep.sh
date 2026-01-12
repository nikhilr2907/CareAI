#!/bin/bash
#SBATCH --job-name=gapo_sweep
#SBATCH --output=slurm_sweep_%A_%a.out
#SBATCH --error=slurm_sweep_%A_%a.err
#SBATCH --array=0-8                # 9 jobs (3 LRs × 3 hidden dims)
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

echo "=========================================="
echo "GAPO Hyperparameter Sweep"
echo "=========================================="
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module load python/3.9
module load cuda/11.8

# Activate environment
source ~/venv/gapo_env/bin/activate

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Hyperparameter arrays
LRS=(0.0001 0.0003 0.001)
HIDDEN_DIMS=(32 64 128)

# Calculate which hyperparameters to use
LR_IDX=$((SLURM_ARRAY_TASK_ID % 3))
HIDDEN_IDX=$((SLURM_ARRAY_TASK_ID / 3))

LR=${LRS[$LR_IDX]}
HIDDEN=${HIDDEN_DIMS[$HIDDEN_IDX]}

echo "Hyperparameters for this job:"
echo "  Learning Rate: $LR"
echo "  Hidden Dim: $HIDDEN"
echo "=========================================="

# Paths
PROJECT_DIR="$HOME/CareRobotics/project_code"
OUTPUT_DIR="$HOME/gapo_sweep"

cd $PROJECT_DIR || exit 1

# Run training with specific hyperparameters
python run_training.py \
  --config configs/hospital_3nodes_consumables.json \
  --iterations 10000 \
  --lr $LR \
  --hidden-dim $HIDDEN \
  --device cuda \
  --output-dir $OUTPUT_DIR \
  --exp-name sweep_lr${LR}_hd${HIDDEN} \
  --save-interval 500 \
  --log-interval 50 \
  --seed 42

EXIT_CODE=$?

echo "=========================================="
echo "Training completed"
echo "Exit code: $EXIT_CODE"
echo "Results: $OUTPUT_DIR/sweep_lr${LR}_hd${HIDDEN}"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
