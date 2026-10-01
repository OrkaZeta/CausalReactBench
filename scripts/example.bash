#!/usr/bin/env bash
#SBATCH --partition=H100,RTX6000PRO,audible,A100,L40S,A40,A30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=JOB_NAME
#SBATCH --output=logs/JOB_NAME/%x-%j.out
#SBATCH --error=logs/JOB_NAME/%x-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=OrkaZeta@outlook.com
set -euxo pipefail

LOG_DIR="logs/JOB_NAME"
# Define log function


# ==================================================
# Detect Slurm vs Direct Bash Execution
# ==================================================
set MODE="bash" 