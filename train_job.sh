#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${USER}
#SBATCH --output=logs/training_%j.out
#SBATCH --job-name=robotics_training

# Load CUDA
source /vol/cuda/12.0.0/setup.sh

# Activate virtual environment
source /vol/bitbucket/nr125/robotics/bin/activate

# Navigate to project directory
cd /vol/bitbucket/nr125/project_code

# Run training script
python run_training.py