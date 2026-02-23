#!/bin/bash
#
# v2.0 Stage 1: Soft-target per-tensor routing with WikiArt + LoRARankEncoder
# Lightweight — no SDXL needed. V100 16GB is sufficient.
#
# Prerequisites:
#   1. clip_similarity.pt + wikiart_label_map.json must exist
#      (run slurm/precompute_v2.sh first)
#
# Usage:
#   sbatch slurm/train_stage1_v2.sh
#
# Override examples:
#   MAX_STEPS=20000 sbatch slurm/train_stage1_v2.sh
#   TAU_LABEL=0.5 sbatch slurm/train_stage1_v2.sh
#

#SBATCH --job-name=MoELoRA-S1v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=12:00:00
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
# Use explicit path to avoid conda init issues on compute nodes
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
export PYTHONUNBUFFERED=1

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage1_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
WIKIART_DIR="/home/eyavuz21/datasets/wikiart"
SIMILARITY_PATH="$CACHE_DIR/clip_similarity.pt"
LABEL_MAP_PATH="$CACHE_DIR/wikiart_label_map.json"
OUTPUT_DIR="/scratch/eyavuz21/lora_attention/stage1_v2"
LOG_DIR="$REPO_ROOT/lora_attention/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Hyperparameters (override via env vars) ─────────────────
MAX_STEPS="${MAX_STEPS:-15000}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TAU_LABEL="${TAU_LABEL:-0.3}"
POOL_MIN="${POOL_MIN:-5}"
POOL_MAX="${POOL_MAX:-20}"
MAX_IMAGES="${MAX_IMAGES:-500}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-50}"
NUM_WORKERS="${NUM_WORKERS:-0}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "MAX_STEPS:  $MAX_STEPS"
echo "LR:         $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "TAU_LABEL:  $TAU_LABEL"
echo "POOL:       [$POOL_MIN, $POOL_MAX]"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "----------------------------------------"

# Verify pre-computed files exist
if [[ ! -f "$SIMILARITY_PATH" ]]; then
    echo "ERROR: CLIP similarity matrix not found: $SIMILARITY_PATH"
    echo "Run: sbatch slurm/precompute_v2.sh"
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
    --similarity_path    "$SIMILARITY_PATH" \
    --label_map_path     "$LABEL_MAP_PATH" \
    --max_steps          "$MAX_STEPS" \
    --lr                 "$LR" \
    --batch_size         "$BATCH_SIZE" \
    --tau_label          "$TAU_LABEL" \
    --min_pool_size      "$POOL_MIN" \
    --max_pool_size      "$POOL_MAX" \
    --max_images_per_style "$MAX_IMAGES" \
    --save_every         "$SAVE_EVERY" \
    --log_every          "$LOG_EVERY" \
    --num_workers        "$NUM_WORKERS" \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
