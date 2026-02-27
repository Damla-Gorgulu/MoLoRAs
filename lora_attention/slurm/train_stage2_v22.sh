#!/bin/bash
#
# v2.2 Stage 2: LDM loss + Switch load-balancing loss + temperature annealing
#
# Key fixes over v2.1:
#   - Replaces per-sample entropy loss (gradient=0 once collapsed) with
#     Switch-Transformer-style EMA load-balancing loss (always non-zero gradient)
#   - Temperature annealing: τ starts at 5.0 (near-uniform) → 1.0 over 2000 steps
#     to prevent snap-collapse in the first 50 steps
#   - lambda_end raised from 0.01 → 0.05 (keep pressure on routing balance)
#
# SLURM output goes to stage2_v22/ (fresh run — do NOT resume from v2.1)
#
# Usage:
#   sbatch slurm/train_stage2_v22.sh
#
# Override examples:
#   MAX_STEPS=10000 sbatch slurm/train_stage2_v22.sh
#   RESUME=/scratch/eyavuz21/lora_attention/stage2_v22/latest.pt sbatch slurm/train_stage2_v22.sh
#

#SBATCH --job-name=MoELoRA-S2v22
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=24:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================"

# ── Modules ────────────────────────────────────────────────
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

# ── Conda environment ───────────────────────────────────────
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
export PYTHONUNBUFFERED=1

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage2_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
WIKIART_DIR="/home/eyavuz21/datasets/wikiart"
LABEL_MAP_PATH="$CACHE_DIR/wikiart_label_map.json"
OUTPUT_DIR="/scratch/eyavuz21/lora_attention/stage2_v22"
STAGE1_CKPT="${STAGE1_CKPT:-/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt}"
LOG_DIR="$REPO_ROOT/lora_attention/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Hyperparameters (override via env vars) ─────────────────
MAX_STEPS="${MAX_STEPS:-8000}"
LR="${LR:-5e-5}"
POOL_MIN="${POOL_MIN:-5}"
POOL_MAX="${POOL_MAX:-20}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"
LAMBDA_START="${LAMBDA_START:-0.1}"
LAMBDA_END="${LAMBDA_END:-0.05}"       # kept high — don't relax pressure
TAU_START="${TAU_START:-5.0}"          # hot start → near-uniform routing
TAU_END="${TAU_END:-1.0}"
TAU_WARMUP="${TAU_WARMUP:-2000}"       # 25% of training to cool down
EMA_BETA="${EMA_BETA:-0.99}"           # ~100-step EMA window for expert usage
MAX_IMAGES="${MAX_IMAGES:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-25}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "stage:        Stage 2 v2.2 (load-balance fix)"
echo "STAGE1_CKPT:  $STAGE1_CKPT"
echo "MAX_STEPS:    $MAX_STEPS"
echo "LR:           $LR"
echo "POOL:         [$POOL_MIN, $POOL_MAX]"
echo "LORA_ALPHA:   $LORA_ALPHA"
echo "λ_lb:         $LAMBDA_START → $LAMBDA_END"
echo "τ:            $TAU_START → $TAU_END (over $TAU_WARMUP steps)"
echo "EMA_BETA:     $EMA_BETA"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "----------------------------------------"

# Verify prerequisites
if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "ERROR: Stage 1 v2.1 checkpoint not found: $STAGE1_CKPT"
    echo "Run: sbatch slurm/train_stage1_v21.sh"
    exit 1
fi
if [[ ! -f "$LABEL_MAP_PATH" ]]; then
    echo "ERROR: WikiArt label map not found: $LABEL_MAP_PATH"
    echo "Run: sbatch slurm/precompute_v2.sh"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --zoo_dir            "$ZOO_DIR" \
    --cache_dir          "$CACHE_DIR" \
    --output_dir         "$OUTPUT_DIR" \
    --wikiart_dir        "$WIKIART_DIR" \
    --label_map_path     "$LABEL_MAP_PATH" \
    --stage1_ckpt        "$STAGE1_CKPT" \
    --max_steps          "$MAX_STEPS" \
    --lr                 "$LR" \
    --min_pool_size      "$POOL_MIN" \
    --max_pool_size      "$POOL_MAX" \
    --lora_alpha         "$LORA_ALPHA" \
    --lambda_start       "$LAMBDA_START" \
    --lambda_end         "$LAMBDA_END" \
    --tau_start          "$TAU_START" \
    --tau_end            "$TAU_END" \
    --tau_warmup_steps   "$TAU_WARMUP" \
    --ema_beta           "$EMA_BETA" \
    --max_images_per_style "$MAX_IMAGES" \
    --save_every         "$SAVE_EVERY" \
    --log_every          "$LOG_EVERY" \
    --mixed_precision    fp16 \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
