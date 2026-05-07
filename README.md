# Mitsubushi_GQE Experiment Suite

This repository implements a reproducible experiment suite for the Phase 2
`H(x) -> U` conditional generative quantum eigensolver plan.

## What this suite includes

- Hamiltonian dataset generation with `PySCF + OpenFermion`
- Baseline ADAPT-VQE benchmark with `qiskit-nature + qiskit-algorithms`
- Baseline CUDA-Q VQE benchmark with `cudaq + cudaq-solvers`
- Strictly supervised sequence model training scaffold (Option 1)
- Result aggregation into CSV tables and PNG visualizations
- One-command orchestration script and SLURM launcher

## Quick start

Use your existing environment:

`/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python`

Run end-to-end:

```bash
bash scripts/run_full_benchmark.sh
```

Outputs are written under `results/`.
Plots are written to `results/plots` as PNG images and listed in
`results/plots/benchmark_plot_manifest.json`.

Re-run plots only (no full suite):

```bash
bash scripts/plot_benchmarks.sh \
  --summary-csv "results/tables/benchmark_summary.csv" \
  --train-json "results/train/train_metrics.json" \
  --out-dir "results/plots" \
  --manifest "results/plots/benchmark_plot_manifest.json"
```

Train the model (CPU-safe default, no legacy CUDA warning noise):

```bash
python src/gqe/models/train_supervised.py \
  --config configs/experiment.yaml \
  --out results/train/supervised_train.done
```

Enable CUDA explicitly if you have a compatible driver:

```bash
python src/gqe/models/train_supervised.py \
  --config configs/experiment.yaml \
  --out results/train/supervised_train.done \
  --use-cuda
```

## Main entrypoints

- `src/gqe/data/generate_hamiltonians.py`
- `src/gqe/baselines/run_adapt_vqe.py`
- `src/gqe/models/train_supervised.py`
- `src/gqe/eval/aggregate_metrics.py`
- `src/gqe/eval/plot_benchmark_results.py`
- `src/gqe/data/generate_hamiltonians.py` (progress bars enabled for dataset generation)

### TQDM progress bars

Scripts that run iterative work now include `tqdm` progress bars:
- `generate_hamiltonians.py`
- `run_adapt_vqe.py`
- `train_supervised.py`

