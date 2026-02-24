#!/bin/bash
#
# v2.1 Stage 2: LDM loss + entropy regularisation with WikiArt
# Uses Stage 1 v2.1 encoder (CE routing, product-space synthesis default).
# Requires full SDXL in fp16 — V100 32GB recommended.
#
# Prerequisites:
#   1. slurm/precompute_v2.sh completed (label map; similarity matrix NOT needed)
#   2. slurm/train_stage1_v21.sh completed (stage1_v21/latest.pt)
#
# Usage:
#   sbatch slurm/train_stage2_v21.sh
#
# Override examples:
#   MAX_STEPS=10000 sbatch slurm/train_stage2_v21.sh
#   STAGE1_CKPT=/path/to/other.pt sbatch slurm/train_stage2_v21.sh
#   RESUME=/scratch/eyavuz21/lora_attention/stage2_v21/latest.pt sbatch slurm/train_stage2_v21.sh
#

#SBATCH --job-name=MoELoRA-S2v21
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
OUTPUT_DIR="/scratch/eyavuz21/lora_attention/stage2_v21"
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
LAMBDA_END="${LAMBDA_END:-0.01}"
MAX_IMAGES="${MAX_IMAGES:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-25}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "stage:        Stage 2 v2.1"
echo "STAGE1_CKPT:  $STAGE1_CKPT"
echo "MAX_STEPS:    $MAX_STEPS"
echo "LR:           $LR"
echo "POOL:         [$POOL_MIN, $POOL_MAX]"
echo "LORA_ALPHA:   $LORA_ALPHA"
echo "λ_entropy:    $LAMBDA_START → $LAMBDA_END"
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
    --max_images_per_style "$MAX_IMAGES" \
    --save_every         "$SAVE_EVERY" \
    --log_every          "$LOG_EVERY" \
    --mixed_precision    fp16 \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
