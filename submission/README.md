# Quantum-Buddies | H-cGQE: Hierarchical Conditional Generative Quantum Eigensolver

**Team Name:** Quantum-Buddies  
**Project Title:** H-cGQE: Hierarchical Conditional Generative Quantum Eigensolver for Electronic Structure Calculations  
**Challenge Track:** Mitsubishi Chemical & AIST — GIC 2026 Phase 3  

---

## Launch on qBraid

[![Launch on qBraid](https://qbraid-static.s3.us-east-2.amazonaws.com/launch-on-qbraid.svg)](https://account.qbraid.com?gitHubUrl=https://github.com/Quantum-Buddies/Conditional_GQE.git)

Click the button above to launch this project on qBraid's quantum computing platform.

---

## Overview

H-cGQE is a two-stage pipeline for quantum chemistry on NISQ hardware:

1. **Autoregressive Circuit Synthesis:** A GPT-2 style transformer generates Pauli operator sequences conditioned on molecular Hamiltonian features.
2. **Classical Coefficient Optimization:** L-BFGS-B optimizes rotation angles (thetas) for the generated operators using CUDA-Q statevector simulation on NVIDIA GPUs.

Training uses supervised fine-tuning followed by DAPO reinforcement learning with a MAP-Elites quality-diversity archive.

### Key Results

| Experiment | Metric | Value |
|---|---|---|
| H-cGQE on CH3I (8q, GPU) | Error vs FCI | **0.629 mHa** (chemical accuracy) |
| H-cGQE on H2@0.74 (4q, GPU) | Error vs FCI | **0.15 mHa** |
| Cepheus QPU SQD | Molecules | **12 molecules** (8–28q) on Rigetti Cepheus-1-108Q |
| Cepheus Best SQD | Error vs FCI | **13.9 mHa** (methyl iodide, 12q) |
| Cepheus EUV Photoresist | Molecules | 8 (methyl iodide, iodobenzene, phenol, o-cresol, anisole, toluene, benzene, imeph) |
| QSCI Scaling | Max qubits | **40 qubits** (benzene CAS(20e,20o)) |
| IQM Emerald QPU | State fidelity | **87.5%** |
| FMO2 Iodobenzene | Solver error | 26.25 mHa |
| Classical: HF vs H-cGQE (H2) | Error reduction | 20.5 mHa → 0.15 mHa |
| VQE: HEA-VQE vs H-cGQE (CH3I) | Error reduction | 987.8 mHa → 0.63 mHa |
| VQE: ADAPT-VQE vs H-cGQE (H2) | Comparable | 0.0002 mHa vs 0.15 mHa |

---

## Setup Instructions

### Environment Dependencies

```
Python >= 3.11
CUDA-Q >= 0.10
PyTorch >= 2.7 (CUDA 12.6+)
openfermion
openfermionpyscf
pyscf
qiskit
qbraid-sdk
huggingface_hub
matplotlib
fpdf
tqdm
pyyaml
```

Or install directly from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Installation on qBraid

```bash
# Clone the repository
git clone https://github.com/Quantum-Buddies/Conditional_GQE.git
cd Conditional_GQE

# Install dependencies (qBraid provides CUDA-Q and PyTorch pre-installed)
pip install -r requirements.txt

# For QPU submissions via qBraid
pip install qbraid-sdk
```

### Download Model Checkpoints

All trained model checkpoints are hosted on Hugging Face:

[https://huggingface.co/Quantum-Buddies/Conditional-GQE-models](https://huggingface.co/Quantum-Buddies/Conditional-GQE-models)

```bash
# Download all models (~450 MB)
python scripts/download_models.py

# Download only essential models (~70 MB)
python scripts/download_models.py --only essential

# Check which models are missing locally
python scripts/download_models.py --check
```

**Key checkpoints:**

| File | Size | Description |
|---|---|---|
| `h_cgqe_model_b200_sft.pt` | 31 MB | SFT warm-start checkpoint |
| `h_cgqe_model_rl_qd_scratch.pt` | 40 MB | RL trained model (main checkpoint) |
| `h_cgqe_model.pt` | 6.1 MB | Base H-cGQE model |
| `chemistry_encoder.pt` | 114 KB | Chemistry GNN encoder |
| `gqe_supervised_dataset.pt` | 15 MB | Supervised training dataset |

Scripts that load checkpoints will auto-download from Hugging Face if the local file is missing. No manual download required unless you prefer to pre-fetch.



## Step-by-Step Instructions to Run on qBraid

### 1. Generate Molecular Hamiltonians

```bash
python src/gqe/data/generate_hamiltonians.py \
    --config configs/gic2026_molecules.yaml \
    --out results/data/hamiltonians_gic2026/
```

**Input:** YAML config with molecular geometries, basis sets, active spaces.  
**Output:** `hamiltonians.json` with Pauli term lists, qubit counts, FCI/HF reference energies.

### 2. Train H-cGQE Transformer (Supervised + RL)

```bash
# Supervised fine-tuning
python src/gqe/models/train_supervised.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/train/h_cgqe_model_sft.pt

# DAPO reinforcement learning
python src/gqe/models/train_rl_dapo.py \
    --checkpoint results/train/h_cgqe_model_sft.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/train/h_cgqe_model_qbraid_rl.pt \
    --n-epochs 50 \
    --target nvidia \
    --max-qubits 24
```

**Input:** Hamiltonian JSON + SFT checkpoint.  
**Output:** RL-trained model checkpoint + MAP-Elites archive + training metrics.

### 3. Evaluate and Optimize Circuits

```bash
# Generate circuits from trained model
python src/gqe/eval/evaluate_h_cgqe.py \
    --checkpoint results/train/h_cgqe_model_qbraid_rl.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --out results/eval/h_cgqe_evaluation.json

# L-BFGS-B coefficient optimization
python src/gqe/eval/optimize_h_cgqe_coefficients.py \
    --evaluation results/eval/h_cgqe_evaluation.json \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --target nvidia-mqpu \
    --out results/eval/h_cgqe_optimized.json
```

**Input:** Trained checkpoint + Hamiltonian data.  
**Output:** Optimized energies, operator sequences, theta values per molecule.

### 4. RL Optimize and Submit to QPU

```bash
# L-BFGS-B on RL checkpoint circuits + export SQD manifests
python scripts/rl_optimize_and_submit.py \
    --checkpoint results/train/h_cgqe_model_qbraid_rl.pt \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --molecules h2 lih beh2 \
    --export-only

# Submit to Rigetti Cepheus
python scripts/submit_sqd_to_cepheus.py \
    --manifests results/qpu/sqd_manifests/ \
    --device aws:rigetti:qpu:cepheus-1-108q \
    --shots 4096

# Retrieve results and run SQD post-processing
python scripts/retrieve_and_sqd.py \
    --ledger results/qpu/qpu_ledger.sqlite \
    --out results/qpu/cepheus_rl_sqd_results.json
```

**Input:** SQD manifests with circuit QASM + measurement groupings.  
**Output:** QPU bitstring counts + SQD-recovered energies with error analysis.

### 5. QSCI Scaling

```bash
python src/gqe/eval/qsci.py \
    --hamiltonians results/data/hamiltonians_gic2026/hamiltonians.json \
    --backend nvidia \
    --max-qubits 40 \
    --out results/phase3_final/qsci_scaling.json
```

**Input:** Hamiltonian records for molecules 4–40 qubits.  
**Output:** QSCI energies, bitstring counts, runtime per molecule.

### 6. FMO2 Fragmentation

```bash
# Generate fragment + dimer Hamiltonians
python scripts/generate_fmo2_fragments.py

# Run FMO2 exact and H-cGQE
python src/gqe/eval/run_fmo2.py \
    --fragments results/data/fragments/monomers.json \
    --dimers results/data/fragments/dimers.json \
    --method exact \
    --out results/phase3_final/fmo/fmo2_exact.json

python src/gqe/eval/run_fmo2.py \
    --fragments results/data/fragments/monomers.json \
    --dimers results/data/fragments/dimers.json \
    --method hcgqe \
    --checkpoint results/train/h_cgqe_model_qbraid_rl.pt \
    --out results/phase3_final/fmo/fmo2_hcgqe.json
```

**Input:** Monomer + dimer Hamiltonian files.  
**Output:** FMO2-reconstructed energies with error decomposition.



## Expected Inputs/Outputs

| Step | Input | Output |
|---|---|---|
| Hamiltonian generation | `configs/gic2026_molecules.yaml` | `results/data/hamiltonians_gic2026/hamiltonians.json` |
| Supervised training | Hamiltonian JSON | `results/train/h_cgqe_model_sft.pt` |
| RL training | SFT checkpoint + Hamiltonians | `results/train/h_cgqe_model_qbraid_rl.pt` + MAP-Elites archive |
| Evaluation | RL checkpoint + Hamiltonians | `results/eval/h_cgqe_evaluation.json` |
| Coefficient optimization | Evaluation JSON | `results/eval/h_cgqe_optimized.json` |
| QPU submission | SQD manifests | `results/qpu/cepheus_rl_sqd_results.json` |
| QSCI scaling | Hamiltonian JSON | `results/phase3_final/qsci_scaling.json` |
| FMO2 | Fragment/dimer Hamiltonians | `results/phase3_final/fmo/fmo2_exact.json` |
| Report generation | All result JSONs | `submission/Write-Up.pdf` |

---

## Source Code Organization

```
Conditional-GQE_materials/
├── src/gqe/
│   ├── models/          # H-cGQE transformer, RL training, chemistry encoder
│   ├── data/            # Hamiltonian generation, fragmentation, graph dataset
│   ├── eval/            # Evaluation, optimization, FMO2, QSCI, error mitigation
│   ├── baselines/       # CUDA-Q GQE, VQE baselines
│   └── common/          # Operator pool, utilities
├── scripts/             # Pipeline scripts (submission, QPU, reporting)
├── configs/             # Experiment configurations
├── results/             # Result JSONs, figures, checkpoints
└── jobs/                # Slurm job scripts
```

---

## Known Limitations and Assumptions

1. **NISQ Noise:** QPU results on Rigetti Cepheus show SQD errors ranging from 13.9 mHa (methyl iodide, 12q) to 130.0 mHa (N2, 20q), correlating with circuit depth and qubit count. Particle number preservation varies from 8% to 71% due to NISQ noise on symmetry-conserving measurements.

2. **Statevector Limit:** Exact statevector simulation is capped at 24 qubits on L40S GPUs due to cuStateVec distributed mode segfaults on PCIe-only systems. MPS backend used for >24q.

3. **FMO2 Fragmentation:** Current FMO2 demonstration uses 2 fragments, where the dimer equals the parent molecule. Code for 3+ fragment scaling is implemented but not yet executed at scale.

4. **QSCI in HF Regime:** QSCI results at 28–40 qubits recover Hartree-Fock energy rather than correlated ground state. Deeper entangling circuits are needed for post-HF accuracy.

5. **Transfer Learning:** Evaluation is limited to embedding similarity analysis. Full end-to-end transfer experiments on unseen molecules are planned.

6. **CUDA-Q + PyTorch:** Importing CUDA-Q before PyTorch's `torch.compile` causes LLVM symbol conflicts. Always import torch first, then cudaq.

7. **qBraid Credits:** ~1,925 credits remaining. 12 QPU submissions completed (8192 shots each, ~4,538 credits spent) on Rigetti Cepheus-1-108Q via AWS Braket (30 credits/task + 0.0425 credits/shot). Each new molecule at 8192 shots costs ~378 credits; at 4096 shots costs ~204 credits.

8. **Classical/VQE Baselines:** Hartree-Fock (HF), hardware-efficient VQE (HEA-VQE), ADAPT-VQE, and CCSD/CCSD(T) results are included in `results/phase3_final/classical_baseline_comparison.json` and `results/baselines/`. See `REPRODUCIBILITY.md` for the full comparison table.

---

## Reproducibility

All results can be reproduced by running the scripts in the order listed above. Key result files included in the submission:

- `results/phase3_final/consolidated_results_gic2026.json` — Consolidated benchmark + QPU results
- `results/eval/h_cgqe_evaluation_gic2026.json` — H-cGQE unoptimized evaluation (17 molecules)
- `results/eval/h_cgqe_optimized_gic2026.json` — H-cGQE L-BFGS-B optimized energies
- `results/eval/h_cgqe_rl_optimized.json` — RL-optimized circuit energies
- `results/qpu/cepheus_sqd_energies.json` — Corrected Cepheus QPU SQD energies (12 molecules, bit-ordering fix applied)
- `results/phase3_final/consolidated_results_gic2026.json` — Consolidated benchmark + QPU results with EUV photoresist metadata
- `results/phase3_final/classical_baseline_comparison.json` — HF, HEA-VQE, ADAPT-VQE, GQE, H-cGQE comparison
- `results/baselines/cudaq_vqe.json` — HEA-VQE baseline on 5 molecules
- `results/baselines/adapt_vqe_h2.json` — ADAPT-VQE baseline on H2/LiH
- `results/baselines/exact_diagonalization.json` — FCI reference energies
- `results/data/fragments/ccsd_refs.json` — CCSD/CCSD(T) for FMO2 fragments
- `results/qpu/cepheus_*_counts.json` — Raw QPU bitstring counts per molecule
- `results/phase3_final/fmo/` — FMO2 exact and H-cGQE results
- `results/phase3_final/qpu/qpu_validation_consolidated.json` — QPU validation summary
- `results/phase3_final/figures/` — All report figures as PNG
- `proposals/Ryoushi_Quantum_Buddies_Phase3_Report.docx` — Editable Word report

Platform: GB200 + qBraid QPU access (Rigetti Cepheus, IQM Emerald).
