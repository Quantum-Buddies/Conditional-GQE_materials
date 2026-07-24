#!/bin/bash
#SBATCH --job-name=gqe-eval
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/results/logs/gqe-eval-%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/results/logs/gqe-eval-%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --gres=gpu:l40s:2

# =============================================================================
# Post-Training Evaluation Pipeline — runs after RL training completes
#
# 1. H-cGQE inference + L-BFGS-B optimization on all molecules
# 2. FMO2 reconstruction (exact + GQE)
# 3. QPU manifest generation (H2, LiH for Rigetti Cepheus)
# 4. QSCI for 32-40q systems (if available)
#
# Prerequisites:
#   - Trained RL checkpoint (h_cgqe_model_rl_offline.pt or similar)
#   - Fragment Hamiltonians (results/data/fragments/fmo_hamiltonians.json)
#   - CUDA-Q installed (cudaq-env)
#
# Submit:  sbatch jobs/eval_pipeline.slurm
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON_BIN="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
export LD_LIBRARY_PATH="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:${LD_LIBRARY_PATH:-}"

# CUDA-Q target setup
export CUDAQ_MQPU_NGPUS=2

# Use the best available checkpoint
CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_model_rl_offline.pt"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="${PROJECT_ROOT}/results/train/h_cgqe_model_b200_sft.pt"
    echo "WARNING: RL checkpoint not found, using SFT checkpoint: ${CHECKPOINT}"
fi

HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_merged.json"
FRAGMENTS="${PROJECT_ROOT}/results/data/fragments/fmo_hamiltonians.json"
ARCHIVE_DIR="${PROJECT_ROOT}/results/train/h_cgqe_model_rl_offline_map_elites"
OUT_DIR="${PROJECT_ROOT}/results/phase3_final"
mkdir -p "$OUT_DIR/fmo" "$OUT_DIR/eval" "$OUT_DIR/qpu"

echo "=== GQE Evaluation Pipeline ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "Checkpoint: ${CHECKPOINT}"
echo ""

cd "${PROJECT_ROOT}"

# --- Step 1: H-cGQE Inference + Optimization ---
echo "[1/4] H-cGQE inference + L-BFGS-B optimization..."
"${PYTHON_BIN}" -u src/gqe/eval/evaluate_h_cgqe.py \
    --checkpoint "${CHECKPOINT}" \
    --hamiltonians "${HAMILTONIANS}" \
    --target nvidia \
    --target-option mqpu \
    --n-samples 200 \
    --max-qubits 24 \
    --out "${OUT_DIR}/eval/h_cgqe_evaluation_final.json" \
    2>&1 || echo "  WARNING: eval step had issues, continuing..."

# --- Step 2: FMO2 Reconstruction ---
echo ""
echo "[2/4] FMO2 reconstruction..."
if [ -f "$FRAGMENTS" ]; then
    # Exact baseline
    "${PYTHON_BIN}" -u src/gqe/eval/run_fmo2.py \
        --fragments "${FRAGMENTS}" \
        --method exact \
        --out "${OUT_DIR}/fmo/fmo2_exact_final.json" \
        2>&1

    # H-cGQE with archive circuits
    "${PYTHON_BIN}" -u src/gqe/eval/run_fmo2.py \
        --fragments "${FRAGMENTS}" \
        --method hcgqe \
        --checkpoint "${CHECKPOINT}" \
        --archive-dir "${ARCHIVE_DIR}" \
        --target nvidia --target-option mqpu \
        --n-samples 100 \
        --out "${OUT_DIR}/fmo/fmo2_gqe_final.json" \
        2>&1 || echo "  WARNING: FMO2 GQE step had issues, continuing..."
else
    echo "  SKIP: Fragment Hamiltonians not found at ${FRAGMENTS}"
fi

# --- Step 3: QPU Manifest Generation ---
echo ""
echo "[3/4] QPU manifest generation..."
OPTIMIZED="${PROJECT_ROOT}/results/eval/h_cgqe_uccsd_optimized.json"
if [ -f "$HAMILTONIANS" ] && [ -f "$OPTIMIZED" ]; then
    "${PYTHON_BIN}" -u scripts/phase3/generate_qpu_manifests.py \
        --molecules h2_0.74 lih_1.6_full \
        --hamiltonians "${HAMILTONIANS}" \
        --optimized "${OPTIMIZED}" \
        --out-dir "${OUT_DIR}/qpu/manifests" \
        --shots 4096 \
        2>&1 || echo "  WARNING: QPU manifest step had issues, continuing..."
else
    echo "  SKIP: Need hamiltonians + optimized results"
fi

# --- Step 4: QSCI (32-40q) ---
echo ""
echo "[4/4] QSCI for large systems..."
QSCI_SCRIPT="${PROJECT_ROOT}/src/gqe/eval/qsci.py"
if [ -f "$QSCI_SCRIPT" ]; then
    "${PYTHON_BIN}" -u "${QSCI_SCRIPT}" \
        --checkpoint "${CHECKPOINT}" \
        --hamiltonians "${PROJECT_ROOT}/results/data/hamiltonians_40plus/hamiltonians.json" \
        --target nvidia \
        --out "${OUT_DIR}/eval/qsci_results.json" \
        2>&1 || echo "  WARNING: QSCI step had issues, continuing..."
else
    echo "  SKIP: QSCI script not found"
fi

echo ""
echo "=== Evaluation Pipeline Complete ==="
echo "End: $(date)"
echo "Results: ${OUT_DIR}/"
