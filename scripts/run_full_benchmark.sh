#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python}"

mkdir -p "${ROOT}/results/data" "${ROOT}/results/baselines" "${ROOT}/results/train" "${ROOT}/results/tables" "${ROOT}/results/plots"

echo "[1/5] Generating Hamiltonian dataset..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/data/generate_hamiltonians.py" \
  --config "${ROOT}/configs/experiment.yaml" \
  --out "${ROOT}/results/data"

echo "[2/5] Running ADAPT-VQE baseline..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/baselines/run_adapt_vqe.py" \
  --out "${ROOT}/results/baselines/adapt_vqe.json"

echo "[3/5] Running CUDA-Q VQE baseline..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/baselines/run_cudaq_vqe.py" \
  --out "${ROOT}/results/baselines/cudaq_vqe.json"

echo "[4/5] Training strict supervised model..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/models/train_supervised.py" \
  --config "${ROOT}/configs/experiment.yaml" \
  --out "${ROOT}/results/train/supervised_train.done"

echo "[5/6] Aggregating metrics..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/eval/aggregate_metrics.py" \
  --ham "${ROOT}/results/data/hamiltonians.json" \
  --baseline "${ROOT}/results/baselines/adapt_vqe.json" \
  --cudaq-baseline "${ROOT}/results/baselines/cudaq_vqe.json" \
  --train "${ROOT}/results/train/train_metrics.json" \
  --out "${ROOT}/results/tables/benchmark_summary.csv"

echo "[6/6] Generating benchmark plots..."
"${PYTHON_BIN}" "${ROOT}/src/gqe/eval/plot_benchmark_results.py" \
  --summary-csv "${ROOT}/results/tables/benchmark_summary.csv" \
  --train-json "${ROOT}/results/train/train_metrics.json" \
  --out-dir "${ROOT}/results/plots" \
  --manifest "${ROOT}/results/plots/benchmark_plot_manifest.json"

echo "Experiment suite complete. See ${ROOT}/results."

