# Conditional-GQE Materials: End-to-End Experiment Suite

This repository contains the full benchmark stack used across this project:
- Hamiltonian generation from molecular inputs
- Quantum baselines (Qiskit-compatible and CUDA-Q)
- Supervised model training (strict CE scaffold)
- Aggregation and visualization
- Optional CUDA-Q GQE loop

## Repository Layout

```text
Conditional-GQE_materials/
├── configs/
│   └── experiment.yaml
├── hamitonian_pipeline/
│   └── test.py
├── jobs/
│   └── gqe-suite.slurm
├── proposals/
│   ├── deep-research-report.md
│   └── phase-1/2 PDFs
├── results/
│   ├── baselines/
│   ├── data/
│   ├── plots/
│   ├── tables/
│   └── train/
├── scripts/
│   ├── plot_benchmarks.sh
│   └── run_full_benchmark.sh
├── src/gqe/
│   ├── baselines/
│   │   ├── run_adapt_vqe.py
│   │   ├── run_cudaq_vqe.py
│   │   ├── run_cudaq_gqe.py
│   ├── data/
│   │   └── generate_hamiltonians.py
│   ├── eval/
│   │   ├── aggregate_metrics.py
│   │   ├── plot_benchmark_results.py
│   ├── models/
│   │   ├── train_supervised.py
│   └── __init__.py
├── README.md
├── requirements.txt
└── PROJECT_DETAILS.md
```

## Environment

Reference environment path used by this project:

- `/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python`

Core dependency file:

- `requirements.txt`

Installed runtime currently includes (and expects) CUDA-Q + CUDA-Q Solvers where applicable.

## Run Configurations

### 1) Train-only (default: CPU-safe)

```bash
python src/gqe/models/train_supervised.py \
  --config configs/experiment.yaml \
  --out results/train/supervised_train.done
```

GPU-enabled training is explicitly requested:

```bash
python src/gqe/models/train_supervised.py \
  --config configs/experiment.yaml \
  --out results/train/supervised_train.done \
  --use-cuda
```

Hyperparameter overrides:
- `--seq-samples`
- `--model-hidden`
- `--model-layers`
- `--model-vocab`

### 2) Full suite (recommended)

```bash
bash scripts/run_full_benchmark.sh
```

This performs, in order:
1. Hamiltonian generation
2. ADAPT-VQE baseline
3. CUDA-Q VQE baseline
4. CUDA-Q GQE baseline (optional)
5. Strict-supervised training
6. Metrics aggregation
7. Plot generation

Optional GQE in full suite:

```bash
RUN_CUDAQ_GQE=1 bash scripts/run_full_benchmark.sh
```

### 3) GQE baseline only

```bash
python src/gqe/baselines/run_cudaq_gqe.py \
  --out results/baselines/cudaq_gqe.json
```

### 4) Baseline-only/aggregation-only

```bash
python src/gqe/eval/aggregate_metrics.py \
  --ham results/data/hamiltonians.json \
  --baseline results/baselines/adapt_vqe.json \
  --cudaq-baseline results/baselines/cudaq_vqe.json \
  --gqe-baseline results/baselines/cudaq_gqe.json \
  --train results/train/train_metrics.json \
  --out results/tables/benchmark_summary.csv

python src/gqe/eval/plot_benchmark_results.py \
  --summary-csv results/tables/benchmark_summary.csv \
  --train-json results/train/train_metrics.json \
  --out-dir results/plots \
  --manifest results/plots/benchmark_plot_manifest.json
```

## Config schema notes

`configs/experiment.yaml` includes:
- `dataset`: molecule list and basis
- `training`: `epochs`, `batch_size`, `learning_rate`, `max_seq_len`, `synthetic_samples`
- `model`: `vocab_size`, `hidden_size`, `num_layers`, `dropout`
- `baselines`: baseline control params

## Output artifacts

Latest defaults write into:

- `results/data/hamiltonians.json`
- `results/baselines/adapt_vqe.json`
- `results/baselines/cudaq_vqe.json`
- `results/baselines/cudaq_gqe.json`
- `results/train/train_metrics.json`
- `results/tables/benchmark_summary.csv`
- `results/plots/*.png`
- `results/plots/benchmark_plot_manifest.json`

## Current training model behavior

`train_supervised.py` defaults to CPU-only execution unless `--use-cuda` is passed.
It also records:
- epoch and batch losses
- model architecture (`model` section)
- synthetic sample count and sequence length

A targeted warning filter is included to reduce noisy CUDA driver-version warnings from optional PyTorch paths.

## CUDA-Q GQE notes

- `run_cudaq_gqe.py` follows NVIDIA docs-oriented operator-pool and `solvers.gqe(...)` flow for H2.
- Uses `cudaq.pauli_word` typed kernel argument to match CUDA-Q kernel typing constraints.
- If `cudaq` / `cudaq-solvers[gqe]` are not installed, the script emits a clear error with install guidance.

## Push workflow

After changes are committed:

```bash
git add -A
git commit -m "Update experiment suite and docs"
git push -u origin main
```

If push fails due authentication/network policy, authenticate with the same account used for:
`https://github.com/Quantum-Buddies/Conditional-GQE_materials`.
