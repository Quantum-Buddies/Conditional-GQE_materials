# Mitsubushi_GQE Experiment Suite

This repository implements a reproducible experiment suite for the Phase 2
`H(x) -> U` conditional generative quantum eigensolver plan.

## What this suite includes

**Note on "GQE" Terminology:** 
NVIDIA recently released a built-in Generative Quantum Eigensolver in `cudaq-solvers` (`solvers.gqe`). 
*Our codebase's Conditional-GQE (cGQE) is a separate research approach*—it uses an autoregressive sequence model (Transformer/GRU scaffold) to conditionally map Hamiltonians to unitaries `H(x) -> U`. It does not use the `solvers.gqe` API. However, our CUDA-Q baseline perfectly aligns with NVIDIA's official VQE patterns using `solvers.vqe` and the UCCSD ansatz.

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

### CUDA-Q GQE baseline (optional)

This repository now includes an optional baseline that exercises NVIDIA's
Generative Quantum Eigensolver via `cudaq-solvers`:

- `src/gqe/baselines/run_cudaq_gqe.py` (H\ :sub:`2` example using `solvers.gqe`)

**Requirements:**

- CUDA-Q and CUDA-Q Solvers installed in the same environment as above.
- GQE extras installed:

```bash
pip install cudaq-solvers[gqe]
```

If the `cudaq-solvers[gqe]` extras (and their PyTorch dependency) are not
available, the script will raise a clear `RuntimeError` with installation
instructions.

**PyTorch vs cluster driver:** GQE pulls in PyTorch. If the wheel’s CUDA
version is newer than what the node driver supports, PyTorch may warn about an
“old” driver. `run_cudaq_gqe.py` filters that noise so logs stay readable; CUDA-Q
still uses its own stack. For a fully clean PyTorch+GPU setup, align the driver
and PyTorch CUDA build per [PyTorch get-started](https://pytorch.org/get-started/locally/).

**Run GQE baseline only:**

```bash
python src/gqe/baselines/run_cudaq_gqe.py \
  --out results/baselines/cudaq_gqe.json
```

**Run full benchmark suite with GQE enabled:**

```bash
RUN_CUDAQ_GQE=1 bash scripts/run_full_benchmark.sh
```

### TQDM progress bars

Scripts that run iterative work now include `tqdm` progress bars:
- `generate_hamiltonians.py`
- `run_adapt_vqe.py`
- `train_supervised.py`

## Detailed workflow reference

- For a full reproducibility walkthrough and command summary, see:
  - `PROJECT_DETAILS.md`

