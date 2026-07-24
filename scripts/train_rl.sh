#!/usr/bin/env bash
# =============================================================================
# Portable RL Training Launcher for qBraid GPU Instances
# =============================================================================
# Two-phase cache-boosted DAPO RL training:
#   Phase A (cache-warmup): 30 epochs, --cache-only, no CUDA-Q calls (~45 min)
#   Phase B (online-rl):    50 epochs, write-through cache, CUDA-Q for misses (~1.5-2h)
#
# Usage:
#   bash scripts/train_rl.sh [smoke|cache-warmup|online-rl|full]
#   Default: full (cache-warmup then online-rl)
#
# All paths are relative to the repo root (auto-detected).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODE="${1:-full}"

# --- Source GPU-specific environment ---
# shellcheck source=/dev/null
source "$SCRIPT_DIR/env_gpu.sh"

# --- Paths (all relative to ROOT) ---
SFT_CKPT="$ROOT/results/train/h_cgqe_model_b200_sft.pt"
HAMILTONIANS="$ROOT/results/data/hamiltonians_gic2026/hamiltonians.json"
ENERGY_CACHE="$ROOT/results/train/rl_energy_cache.sqlite"
PRETRAIN_DATA="$ROOT/results/train/rl_pretrain_from_cache.json"
OUTPUT_CKPT="$ROOT/results/train/h_cgqe_model_qbraid_rl.pt"
OUTPUT_CKPT_WARMUP="$ROOT/results/train/h_cgqe_model_qbraid_rl_warmup.pt"
LOG_DIR="$ROOT/results/train"

mkdir -p "$LOG_DIR"

# --- Auto-detect GPU-specific limits ---
# H200 141GB → 30q SV, H100 80GB → 26q SV, B200 180GB → 32q SV, A100 80GB → 26q
MAX_QUBITS=28
MPS_THRESHOLD=24
case "$GPU_CC" in
    9.0*)  [ "$GPU_VRAM_GB" -ge 140 ] 2>/dev/null && MAX_QUBITS=30 || MAX_QUBITS=26 ;;
    10.0*) MAX_QUBITS=32; MPS_THRESHOLD=28 ;;
    8.0*)  MAX_QUBITS=26; MPS_THRESHOLD=22 ;;
    *)     MAX_QUBITS=24; MPS_THRESHOLD=20 ;;
esac

# --- Auto-generate molecule list from hamiltonians JSON ---
MOLECULES="$(python3 -c "
import json
d = json.load(open('$HAMILTONIANS'))
recs = sorted(
    [r for r in d['records'] if r.get('n_qubits', 99) <= $MAX_QUBITS],
    key=lambda r: (r.get('n_qubits', 99), r.get('name', '')),
)
print(' '.join(r['name'] for r in recs))
" 2>/dev/null || echo 'h2 lih beh2 n2')"

# --- Credit rate lookup for cost tracking ---
get_credit_rate() {
    case "$GPU_NAME" in
        *H200*)  echo 9.15 ;;
        *H100*)  echo 8.95 ;;
        *B200*)  echo 14.57 ;;
        *A100*)  echo 4.15 ;;
        *L40S*)  echo 3.80 ;;
        *L4*)    echo 0.82 ;;
        *RTX*5090*) echo 2.07 ;;
        *RTX*4090*) echo 1.45 ;;
        *A10*)   echo 2.70 ;;
        *RTX*6000*) echo 1.53 ;;
        *)       echo 0.0 ;;
    esac
}

# --- Prerequisite Checks ---
check_required() {
    local missing=0
    for f in "$SFT_CKPT" "$HAMILTONIANS" "$ENERGY_CACHE" "$PRETRAIN_DATA"; do
        if [ ! -f "$f" ]; then
            echo "  [MISSING] $f"
            missing=1
        fi
    done
    if [ "$missing" -eq 1 ]; then
        echo ""
        echo "ERROR: Required files missing. Run 'bash scripts/setup_env.sh' first."
        exit 1
    fi
}

# --- Common RL Arguments (shared across all modes) ---
COMMON_ARGS=(
    --checkpoint "$SFT_CKPT"
    --hamiltonians "$HAMILTONIANS"
    --molecules $MOLECULES
    --d-model 256
    --nhead 8
    --encoder-layers 4
    --decoder-layers 6
    --dim-feedforward 1024
    --dropout 0.1
    --lr 1e-5
    --temperature 1.0
    --top-p 0.9
    --target-entropy 1.5
    --explore-eps 0.3
    --adaptive-eps
    --force-entanglement
    --max-repeat 4
    --max-seq-len 64
    --max-terms 128
    --max-pauli-len 24
    --use-cuda
    --use-bf16
    --torch-compile
    --compile-mode reduce-overhead
    --fused-optimizer
    --token-level-loss
    --entropy-coef 0.01
    --clip-low 0.2
    --clip-high 0.28
    --w-energy 1.0
    --w-entangle 0.1
    --w-depth 0.05
    --w-commute 0.05
    --w-diversity 0.2
    --target-len 10
    --gate-auxiliary-rewards
    --energy-improvement-threshold 0.0
    --freq-penalty 1.0
    --buffer-size 2000
    --buffer-batch-size 128
    --curriculum
    --curriculum-warmup 10
    --curriculum-steps 3
    --pretrain-data "$PRETRAIN_DATA"
    --pretrain-fraction 0.8
    --adaptive-theta
    --adaptive-theta-iters 10
    --qd-mode
    --qd-novelty-weight 1.0
    --qd-lambda-final 0.1
    --qd-coverage-threshold 0.5
    --qd-n-bins-entanglement 10
    --qd-n-bins-depth 10
    --qd-lbfgs-iters 3
    --energy-cache "$ENERGY_CACHE"
    --theta 0.01
    --seed 42
    --target nvidia
    --target-option fp32
    --max-qubits "$MAX_QUBITS"
    --mps-threshold "$MPS_THRESHOLD"
)

# =====================================================================
# Mode: smoke — 2 epochs, 2 molecules, sanity check (~2 min)
# =====================================================================
run_smoke() {
    echo "============================================================"
    echo "  SMOKE TEST — 2 epochs, 2 molecules (~2 min)"
    echo "============================================================"
    check_required

    local SMOKE_OUT="$ROOT/results/train/h_cgqe_model_qbraid_smoke.pt"
    python3 -u "$ROOT/src/gqe/models/train_rl_dapo.py" \
        "${COMMON_ARGS[@]}" \
        --molecules h2 lih \
        --out "$SMOKE_OUT" \
        --epochs 2 \
        --n-samples 16 \
        --n-iters 4 \
        --reuse-iters 3 \
        --cache-only \
        --pretrain-decay-epochs 2 \
        --no-curriculum \
        2>&1 | tee "$LOG_DIR/rl_smoke.log"

    echo "Smoke test complete → $SMOKE_OUT"
}

# =====================================================================
# Mode: cache-warmup — 30 epochs, --cache-only, no CUDA-Q (~45 min)
# =====================================================================
run_cache_warmup() {
    echo "============================================================"
    echo "  PHASE A: Cache-Only Warmup — 30 epochs (~45 min)"
    echo "  GPU     : $GPU_NAME (${GPU_VRAM_GB}GB, CC $GPU_CC)"
    echo "  Max q   : $MAX_QUBITS"
    echo "  Mols    : $(echo $MOLECULES | wc -w) molecules"
    echo "  Cache   : $ENERGY_CACHE"
    echo "============================================================"
    check_required

    local START=$(date +%s)

    python3 -u "$ROOT/src/gqe/models/train_rl_dapo.py" \
        "${COMMON_ARGS[@]}" \
        --out "$OUTPUT_CKPT_WARMUP" \
        --epochs 30 \
        --n-samples 16 \
        --n-iters 4 \
        --reuse-iters 3 \
        --cache-only \
        --pretrain-decay-epochs 30 \
        2>&1 | tee "$LOG_DIR/rl_cache_warmup.log"

    local ELAPSED=$(( ($(date +%s) - START) / 60 ))
    local RATE; RATE=$(get_credit_rate)
    local COST; COST=$(python3 -c "print(f'{$ELAPSED * $RATE:.1f}')" 2>/dev/null || echo "N/A")
    echo ""
    echo "  Phase A complete: ${ELAPSED} min (~${COST} credits)"
    echo "  Output: $OUTPUT_CKPT_WARMUP"
}

# =====================================================================
# Mode: online-rl — 50 epochs, write-through cache, CUDA-Q misses (~1.5-2h)
# =====================================================================
run_online_rl() {
    echo "============================================================"
    echo "  PHASE B: Online RL — 50 epochs (~1.5-2h)"
    echo "  GPU     : $GPU_NAME (${GPU_VRAM_GB}GB, CC $GPU_CC)"
    echo "  Max q   : $MAX_QUBITS"
    echo "  Cache   : $ENERGY_CACHE (write-through)"
    echo "============================================================"

    # Load warmup checkpoint if available, else fall back to SFT
    local CKPT
    if [ -f "$OUTPUT_CKPT_WARMUP" ]; then
        CKPT="$OUTPUT_CKPT_WARMUP"
        echo "  Loading warmup checkpoint: $CKPT"
    else
        CKPT="$SFT_CKPT"
        echo "  No warmup checkpoint found — loading SFT: $CKPT"
    fi

    local START=$(date +%s)

    python3 -u "$ROOT/src/gqe/models/train_rl_dapo.py" \
        "${COMMON_ARGS[@]}" \
        --checkpoint "$CKPT" \
        --out "$OUTPUT_CKPT" \
        --epochs 50 \
        --n-samples 16 \
        --n-iters 4 \
        --reuse-iters 3 \
        --pretrain-decay-epochs 50 \
        --eval-async \
        --eval-async-chunk 24 \
        2>&1 | tee "$LOG_DIR/rl_online.log"

    local ELAPSED=$(( ($(date +%s) - START) / 60 ))
    local RATE; RATE=$(get_credit_rate)
    local COST; COST=$(python3 -c "print(f'{$ELAPSED * $RATE:.1f}')" 2>/dev/null || echo "N/A")
    echo ""
    echo "  Phase B complete: ${ELAPSED} min (~${COST} credits)"
    echo "  Output: $OUTPUT_CKPT"
}

# =====================================================================
# Mode: full — cache-warmup then online-rl sequentially (~3h total)
# =====================================================================
run_full() {
    local TOTAL_START=$(date +%s)

    run_cache_warmup
    echo ""
    run_online_rl

    local TOTAL=$(( ($(date +%s) - TOTAL_START) / 60 ))
    local RATE; RATE=$(get_credit_rate)
    local TOTAL_COST; TOTAL_COST=$(python3 -c "print(f'{$TOTAL * $RATE:.1f}')" 2>/dev/null || echo "N/A")
    echo ""
    echo "============================================================"
    echo "  FULL TRAINING COMPLETE"
    echo "  Total time  : ${TOTAL} min"
    echo "  Est. cost   : ${TOTAL_COST} credits"
    echo "  Output      : $OUTPUT_CKPT"
    echo "============================================================"
}

# =====================================================================
# Dispatch
# =====================================================================
cd "$ROOT"

case "$MODE" in
    smoke)        run_smoke ;;
    cache-warmup) run_cache_warmup ;;
    online-rl)    run_online_rl ;;
    full)         run_full ;;
    *)
        echo "Usage: bash scripts/train_rl.sh [smoke|cache-warmup|online-rl|full]"
        echo "  smoke        : 2 epochs, 2 molecules, sanity check (~2 min)"
        echo "  cache-warmup : 30 epochs, cache-only, no CUDA-Q (~45 min)"
        echo "  online-rl    : 50 epochs, write-through cache, CUDA-Q misses (~1.5-2h)"
        echo "  full         : cache-warmup then online-rl (~3h total)"
        exit 1
        ;;
esac
