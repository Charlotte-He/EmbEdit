#!/usr/bin/env bash
#SBATCH --job-name=embedit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail

# Activate your installed environment BEFORE calling sbatch.
# Run from the repository root; sbatch forwards the current directory.
# Select GPU type / QOS / memory according to your cluster and model.
cd "${SLURM_SUBMIT_DIR:?Submit this script through sbatch}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -m embedit.cli "$@"
