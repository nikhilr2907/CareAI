#!/bin/bash
#SBATCH --job-name=gapo_train
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Optional: Email notifications (uncomment and add your email)
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@university.edu

echo "=========================================="
echo "GAPO Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Load modules (adjust to your cluster)
echo "Loading modules..."
module load python/3.9
module load cuda/11.8
module list

# Activate virtual environment
echo "Activating virtual environment..."
source /vol/bitbucket/nr125/robotics/bin/activate
which python
python --version

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1  # Ensures real-time logging
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print GPU info
echo "=========================================="
echo "GPU Information:"
nvidia-smi
echo "=========================================="

# Define paths
PROJECT_DIR="/vol/bitbucket/nr125/project_code"
OUTPUT_DIR="/vol/bitbucket/nr125/gapo_outputs"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Go to project directory
cd $PROJECT_DIR || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }
echo "Current directory: $(pwd)"
echo "Project files:"
ls -lh run_training.py configs/

# Check if config file exists
CONFIG_FILE="configs/hospital_3nodes_consumables.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "Starting training..."
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR/job_${SLURM_JOB_ID}"
echo "=========================================="

# Run training
python run_training.py \
  --config $CONFIG_FILE \
  --iterations 10000 \
  --device cuda \
  --output-dir $OUTPUT_DIR \
  --exp-name job_${SLURM_JOB_ID} \
  --save-interval 500 \
  --log-interval 50 \
  --seed 42

# Capture exit code
EXIT_CODE=$?

echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=========================================="

# Print output location
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS! Results saved to:"
    echo "  Output dir: $OUTPUT_DIR/job_${SLURM_JOB_ID}"
    echo "  Log file: $OUTPUT_DIR/job_${SLURM_JOB_ID}/logs/training.log"
    echo "  Checkpoints: $OUTPUT_DIR/job_${SLURM_JOB_ID}/checkpoints/"

    # Show final checkpoint
    echo ""
    echo "Saved checkpoints:"
    ls -lh $OUTPUT_DIR/job_${SLURM_JOB_ID}/checkpoints/ 2>/dev/null || echo "  (No checkpoints found)"

    # Show last 20 lines of log
    echo ""
    echo "Last 20 lines of training log:"
    tail -n 20 $OUTPUT_DIR/job_${SLURM_JOB_ID}/logs/training.log 2>/dev/null || echo "  (No log file found)"
else
    echo "FAILED with exit code $EXIT_CODE"
    echo "Check errors in slurm_${SLURM_JOB_ID}.err"
fi

# Print resource usage
echo ""
echo "Job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State 2>/dev/null || echo "  (sacct not available)"

exit $EXIT_CODE
