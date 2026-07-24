#!/bin/bash
#SBATCH --job-name=rl-qd-1g
#SBATCH --output=results/logs/rl-qd-1gpu-%j.out
#SBATCH --error=results/logs/rl-qd-1gpu-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:1

# =============================================================================
# QD-GRPO RL from Scratch — 1× L40S (Ablation)
# =============================================================================
# Free parallel ablation while main training runs on qBraid.
# 4 core molecules only (h2, lih, beh2, n2) for comparison.
#
# Submit:  sbatch jobs/rl_qd_1gpu_aire.slurm
# Monitor: tail -f results/logs/rl-qd-1gpu-*.out
# =============================================================================

set -euo pipefail

# --- Portable path resolution ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"

# --- Conda environment activation (portable) ---
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate cudaq-env 2>/dev/null || true
fi

# Fallback: if conda activate didn't work, prepend to PATH
if ! python -c "import cudaq" 2>/dev/null; then
    export PATH="$PROJECT_ROOT/../.conda_envs/cudaq-env/bin:$PATH"
    export LD_LIBRARY_PATH="$PROJECT_ROOT/../.conda_envs/cudaq-env/lib:${LD_LIBRARY_PATH:-}"
fi

PYTHON_BIN="python"

# --- L40S PCIe-only — prevent NCCL issues ---
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- CUDA-Q performance tuning ---
export CUDAQ_MGPU_FUSE=6
export CUDAQ_ALLOW_FP32_EMULATED=1
export CUDAQ_MEMPOOL=1
export CUDAQ_FUSE=6

# --- Paths (relative to PROJECT_ROOT) ---
HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_gic2026/hamiltonians.json"
ENERGY_CACHE="${PROJECT_ROOT}/results/train/rl_energy_cache.sqlite"
PRETRAIN_DATA="${PROJECT_ROOT}/results/train/rl_pretrain_from_cache.json"
OUTPUT="${PROJECT_ROOT}/results/train/h_cgqe_model_rl_qd_1gpu_scratch.pt"
ARCHIVE_DIR="${PROJECT_ROOT}/results/train/rl_qd_map_elites_1gpu"

echo "=== QD-GRPO RL from Scratch (1× L40S Ablation) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Project: ${PROJECT_ROOT}"
echo ""

cd "${PROJECT_ROOT}"

# 4 core molecules for ablation comparison
"${PYTHON_BIN}" -u src/gqe/models/train_rl_dapo.py \
    --from-scratch \
    --d-model 256 \
    --nhead 8 \
    --encoder-layers 4 \
    --decoder-layers 6 \
    --dim-feedforward 1024 \
    --dropout 0.1 \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 \
    --out "${OUTPUT}" \
    --epochs 150 \
    --n-samples 16 \
    --n-iters 4 \
    --reuse-iters 3 \
    --lr 1e-5 \
    --temperature 1.0 \
    --top-p 0.9 \
    --target-entropy 1.5 \
    --explore-eps 0.3 \
    --adaptive-eps \
    --force-entanglement \
    --max-repeat 4 \
    --max-qubits 24 \
    --max-seq-len 64 \
    --max-terms 128 \
    --max-pauli-len 24 \
    --use-cuda \
    --single-gpu \
    --use-bf16 \
    --torch-compile \
    --compile-mode reduce-overhead \
    --fused-optimizer \
    --no-dynamic-sampling \
    --token-level-loss \
    --entropy-coef 0.01 \
    --clip-low 0.2 \
    --clip-high 0.28 \
    --w-energy 1.0 \
    --w-entangle 0.1 \
    --w-depth 0.05 \
    --w-commute 0.05 \
    --w-diversity 0.2 \
    --target-len 10 \
    --gate-auxiliary-rewards \
    --energy-improvement-threshold 0.0 \
    --freq-penalty 1.0 \
    --buffer-size 2000 \
    --buffer-batch-size 128 \
    --curriculum \
    --curriculum-warmup 20 \
    --curriculum-steps 3 \
    --pretrain-data "${PRETRAIN_DATA}" \
    --pretrain-fraction 0.8 \
    --pretrain-decay-epochs 100 \
    --adaptive-theta \
    --adaptive-theta-iters 10 \
    --qd-mode \
    --qd-novelty-weight 1.0 \
    --qd-lambda-final 0.1 \
    --qd-coverage-threshold 0.5 \
    --qd-n-bins-entanglement 10 \
    --qd-n-bins-depth 10 \
    --qd-lbfgs-iters 3 \
    --qd-archive-path "${ARCHIVE_DIR}" \
    --energy-cache "${ENERGY_CACHE}" \
    --theta 0.01 \
    --seed 42 \
    --target nvidia \
    --target-option fp32 \
    2>&1

echo ""
echo "=== Ablation Training Complete ==="
echo "End: $(date)"
echo "Output: ${OUTPUT}"
echo "Metrics: ${OUTPUT%.pt}_rl_metrics.json"
