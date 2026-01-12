#!/bin/bash
#SBATCH --job-name=gapo_long
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=72:00:00           # 72 hours for long training
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G                  # More memory for larger configs
#SBATCH --cpus-per-task=8          # More CPUs for data loading

# Optional: Request specific GPU type
# #SBATCH --constraint=a100        # Request A100 GPUs
# #SBATCH --gres=gpu:a100:1       # Request 1 A100

# Optional: Email notifications
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@university.edu

echo "=========================================="
echo "GAPO Long Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "=========================================="

# Load modules (adjust to your cluster)
module load python/3.9
module load cuda/11.8

# Activate virtual environment
source ~/venv/gapo_env/bin/activate

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print GPU info
nvidia-smi

# Paths
PROJECT_DIR="$HOME/CareRobotics/project_code"
OUTPUT_DIR="$HOME/gapo_outputs"
CONFIG_FILE="configs/hospital_10nodes.json"  # Larger config

cd $PROJECT_DIR || exit 1

# Run training with more iterations
python run_training.py \
  --config $CONFIG_FILE \
  --iterations 50000 \
  --lr 0.0003 \
  --hidden-dim 128 \
  --device cuda \
  --output-dir $OUTPUT_DIR \
  --exp-name long_job_${SLURM_JOB_ID} \
  --save-interval 1000 \
  --log-interval 100 \
  --seed 42

EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
