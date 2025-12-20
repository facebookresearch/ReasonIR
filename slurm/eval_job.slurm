#!/bin/bash

#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=InstallEnvironment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=logs_eval/slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

source .venv/bin/activate

bash evaluation/bright/script.sh
