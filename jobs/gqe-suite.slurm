#!/bin/bash
#SBATCH --job-name=gqe-suite
#SBATCH --output=/scratch/kcwp264/Mitsubushi_GQE/results/gqe-suite-%j.out
#SBATCH --error=/scratch/kcwp264/Mitsubushi_GQE/results/gqe-suite-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/Mitsubushi_GQE"
PYTHON_BIN="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"

echo "Starting experiment suite on $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"

bash "${PROJECT_ROOT}/scripts/run_full_benchmark.sh"

echo "Finished on $(date)"

