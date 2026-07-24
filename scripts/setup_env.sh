#!/usr/bin/env bash
# =============================================================================
# Portable One-Shot Setup Script for qBraid GPU Instances
# =============================================================================
# Run this immediately after `git clone` on a new qBraid instance.
#
# Usage:
#   bash scripts/setup_env.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo "  Conditional-GQE: Portable Environment Setup"
echo "  Root : $ROOT"
echo "  Host : $(hostname)"
echo "  Date : $(date)"
echo "============================================================"
echo ""

# ------------------------------------------------------------------
# 1. Git LFS — install if missing, then pull all LFS-tracked assets
# ------------------------------------------------------------------
echo ">>> [1/5] Git LFS..."
if ! command -v git-lfs &>/dev/null; then
    echo "    git-lfs not found — attempting install..."
    if command -v apt-get &>/dev/null; then
        apt-get update -qq && apt-get install -y -qq git-lfs 2>/dev/null || true
    elif command -v conda &>/dev/null; then
        conda install -y -c conda-forge git-lfs 2>/dev/null || true
    fi
fi

if command -v git-lfs &>/dev/null; then
    echo "    Pulling LFS assets..."
    cd "$ROOT"
    git lfs install --skip-repo 2>/dev/null || true
    git lfs pull 2>/dev/null || echo "    WARNING: git lfs pull failed — check network or LFS quota."
else
    echo "    WARNING: git-lfs could not be installed automatically."
    echo "             Manually install it and run 'git lfs pull' to fetch"
    echo "             checkpoints, energy cache, and pretrain data."
fi

# ------------------------------------------------------------------
# 2. Python Dependencies
# ------------------------------------------------------------------
echo ""
echo ">>> [2/5] Python Dependencies..."
cd "$ROOT"

if [ -f "requirements-qbraid.txt" ]; then
    pip install --quiet -r requirements-qbraid.txt
elif [ -f "requirements.txt" ]; then
    pip install --quiet -r requirements.txt
else
    pip install --quiet pyyaml numpy scipy matplotlib tqdm \
        openfermion openfermionpyscf pyscf \
        qiskit qiskit-algorithms qiskit-nature \
        cudaq cudaq-solvers fpdf2
fi

# ------------------------------------------------------------------
# 3. PyTorch with CUDA
# ------------------------------------------------------------------
echo ""
echo ">>> [3/5] PyTorch CUDA..."
python3 -c "
import torch, sys, subprocess
if not torch.cuda.is_available():
    print('    CUDA not available — reinstalling PyTorch (cu126)...')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet',
                    'torch', 'torchvision',
                    '--index-url', 'https://download.pytorch.org/whl/cu126',
                    '--force-reinstall'])
else:
    print(f'    PyTorch {torch.__version__} with CUDA ready.')
"

# ------------------------------------------------------------------
# 4. GPU & CUDA-Q Verification
# ------------------------------------------------------------------
echo ""
echo ">>> [4/5] GPU & CUDA-Q Verification..."
python3 -c "
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'    GPU  : {name}')
    print(f'    CC   : sm_{cap[0]}{cap[1]}')
    print(f'    VRAM : {vram:.1f} GB')
else:
    print('    WARNING: No GPU detected by PyTorch.')
    sys.exit(1)

try:
    import cudaq
    cudaq.set_target('nvidia')
    print(f'    CUDA-Q: {cudaq.__version__} (target=nvidia OK)')
except Exception as e:
    print(f'    WARNING: CUDA-Q test failed: {e}')
"

# ------------------------------------------------------------------
# 5. Asset Audit
# ------------------------------------------------------------------
echo ""
echo ">>> [5/5] Asset Audit..."

check_asset() {
    if [ -f "$1" ]; then
        local sz; sz=$(du -h "$1" | cut -f1)
        echo "    [OK]      $2 ($sz)"
    else
        echo "    [MISSING] $2"
    fi
}

check_asset "$ROOT/results/train/h_cgqe_model_b200_sft.pt"       "SFT Warm-Start Checkpoint"
check_asset "$ROOT/results/train/rl_energy_cache.sqlite"          "SQLite Energy Cache (25K entries)"
check_asset "$ROOT/results/train/rl_pretrain_from_cache.json"     "Pretrain Bootstrap JSON (24K circuits)"
check_asset "$ROOT/results/data/hamiltonians_gic2026/hamiltonians.json" "GIC 2026 Hamiltonians (35 molecules)"
check_asset "$ROOT/results/baselines/cudaq_gqe_baseline.json"     "GQE Baseline Results"

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo ""
echo "  Next Steps:"
echo "    Smoke test : bash scripts/train_rl.sh smoke"
echo "    Full RL    : bash scripts/train_rl.sh full"
echo "    Evaluation : bash scripts/evaluate_rl.sh all"
echo "============================================================"
