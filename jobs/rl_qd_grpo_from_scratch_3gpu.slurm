#!/bin/bash
#SBATCH --job-name=rl-qd-3g
#SBATCH --output=/scratch/kcwp264/Conditional-GQE_materials/results/logs/rl-qd-3gpu-%j.out
#SBATCH --error=/scratch/kcwp264/Conditional-GQE_materials/results/logs/rl-qd-3gpu-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:l40s:3

# =============================================================================
# QD-GRPO RL Training from Scratch — 3× L40S
#
# Architecture: MAP-Elites × GRPO (Quality-Diversity RL)
#   - MAP-Elites archive: 2D grid (entanglement_density × circuit_depth)
#   - Novelty bonus: intrinsic reward for exploring unoccupied niches
#   - Truncated L-BFGS-B surrogate (3 iters) for energy evaluation
#   - DAPO loss: asymmetric clipping, token-level, dynamic sampling
#
# Bootstrapping: B200 cache → pretrain JSON (24K circuits with real energies)
#   - 50% of replay buffer pre-filled with cached circuits
#   - Linear decay to 0% over 100 epochs
#   - Cache also used in write-through mode for online RL
#
# Online RL: CUDA-Q nvidia-mqpu for real energy evaluation
#   - Max 24q (L40S cuStateVec limit, ethylene 28q excluded from online)
#   - Write-through to SQLite cache (grows over time)
#
# Submit:  sbatch jobs/rl_qd_grpo_from_scratch_3gpu.slurm
# Monitor: tail -f results/logs/rl-qd-3gpu-*.out
# =============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/kcwp264/Conditional-GQE_materials"
PYTHON_BIN="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/bin/python"
export LD_LIBRARY_PATH="/mnt/scratch/kcwp264/.conda_envs/cudaq-env/lib:${LD_LIBRARY_PATH:-}"

# L40S PCIe-only — prevent NCCL issues
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CUDA-Q performance tuning (gate fusion + memory pool)
# Gate fusion level 6: merges more consecutive gates for fewer GPU kernels
# NVIDIA blog shows up to 10x speedup tuning this for specific circuits
export CUDAQ_MGPU_FUSE=6
# Allow FP32 emulated BF16 tensor-core kernels (faster observe on Ada/Hopper)
export CUDAQ_ALLOW_FP32_EMULATED=1
# CUDA-Q memory pool: reuse GPU allocations across observe() calls
export CUDAQ_MEMPOOL=1
# cuStateVec: use gate fusion level 6 for single-GPU nvidia target too
export CUDAQ_FUSE=6

HAMILTONIANS="${PROJECT_ROOT}/results/data/hamiltonians_gic2026/hamiltonians.json"
ENERGY_CACHE="${PROJECT_ROOT}/results/train/rl_energy_cache.sqlite"
PRETRAIN_DATA="${PROJECT_ROOT}/results/train/rl_pretrain_from_cache.json"
OUTPUT="${PROJECT_ROOT}/results/train/h_cgqe_model_rl_qd_scratch.pt"
ARCHIVE_DIR="${PROJECT_ROOT}/results/train/rl_qd_map_elites"

echo "=== QD-GRPO RL from Scratch (3× L40S) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Pretrain data: ${PRETRAIN_DATA}"
echo "Energy cache: ${ENERGY_CACHE}"
echo ""

cd "${PROJECT_ROOT}"

# All 34 GIC molecules ≤24q (ethylene 28q excluded from online RL)
# Ethylene still gets pretrain data bootstrapping via the cache
"${PYTHON_BIN}" -u src/gqe/models/train_rl_dapo.py \
    --from-scratch \
    --d-model 256 \
    --nhead 8 \
    --encoder-layers 4 \
    --decoder-layers 6 \
    --dim-feedforward 1024 \
    --dropout 0.1 \
    --hamiltonians "${HAMILTONIANS}" \
    --molecules h2 lih beh2 n2 h2o nh3 ch4 formaldehyde acetylene hf co \
                imeph_cas12 iodobenzene_cas12 methyl_iodide_cas12 phenol_cas12 \
                benzene_cas12 toluene_cas12 anisole_cas12 ocresol_cas12 \
                diarylethene_frag_cas12 \
                h2_0.5 h2_1.0 h2_1.5 h2_2.0 lih_1.2 lih_2.0 lih_3.0 \
                n2_1.8 n2_2.5 beh2_1.0 beh2_1.6 lih_1.6_631g \
                n2_1.1_631g_cas8 h2o_1.0_631g_cas8 \
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
    --multi-gpu \
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
    2>&1

echo ""
echo "=== Training Complete ==="
echo "End: $(date)"
echo "Output: ${OUTPUT}"
echo "MAP-Elites archives: ${ARCHIVE_DIR}"
echo "Metrics: ${OUTPUT%.pt}_rl_metrics.json"
