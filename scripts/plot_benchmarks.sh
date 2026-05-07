#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}"

"${PYTHON_BIN}" "${ROOT}/src/gqe/eval/plot_benchmark_results.py" "$@"

