#!/bin/bash
#
# Mini generalization v2: singleton-only Stage 1 canary.
#
# Tightens the previous mini generalization setup:
#   - only singleton styles
#   - smaller pool
#   - sharper Stage 1 training temperature
#   - intended to answer a narrower question:
#       can Stage 1 retrieve the correct in-pool expert reliably?
#
# Usage:
#   sbatch lora_attention/slurm/mini_generalization/train_mini_generalization_v2.sh

#SBATCH --job-name=MoELoRA-MiniGenV2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=06:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "========================================"

module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage1_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="${CACHE_DIR:-/scratch/eyavuz21/lora_attention}"
WIKIART_DIR="/home/eyavuz21/datasets/wikiart"
LABEL_MAP_PATH="${LABEL_MAP_PATH:-$REPO_ROOT/lora_attention/configs/wikiart_label_map_mini_generalization_v2.json}"

ROOT_BASE="${ROOT_BASE:-/scratch/eyavuz21/lora_attention/mini_generalization_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_BASE/stage1_train}"
mkdir -p "$OUTPUT_DIR"

MAX_STEPS="${MAX_STEPS:-1200}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TAU_LABEL="${TAU_LABEL:-0.3}"
POOL_MIN="${POOL_MIN:-3}"
POOL_MAX="${POOL_MAX:-5}"
MAX_IMAGES="${MAX_IMAGES:-96}"
SAVE_EVERY="${SAVE_EVERY:-200}"
LOG_EVERY="${LOG_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-0.5}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "OUTPUT_DIR:        $OUTPUT_DIR"
echo "LABEL_MAP_PATH:    $LABEL_MAP_PATH"
echo "MAX_STEPS:         $MAX_STEPS"
echo "LR:                $LR"
echo "POOL:              [$POOL_MIN, $POOL_MAX]"
echo "MAX_IMAGES:        $MAX_IMAGES"
echo "TRAIN_TEMPERATURE: $TRAIN_TEMPERATURE"
echo "----------------------------------------"

if [[ ! -f "$LABEL_MAP_PATH" ]]; then
    echo "ERROR: label map not found: $LABEL_MAP_PATH"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --zoo_dir              "$ZOO_DIR" \
    --cache_dir            "$CACHE_DIR" \
    --output_dir           "$OUTPUT_DIR" \
    --wikiart_dir          "$WIKIART_DIR" \
    --label_map_path       "$LABEL_MAP_PATH" \
    --target_mode          ce \
    --max_steps            "$MAX_STEPS" \
    --lr                   "$LR" \
    --batch_size           "$BATCH_SIZE" \
    --tau_label            "$TAU_LABEL" \
    --min_pool_size        "$POOL_MIN" \
    --max_pool_size        "$POOL_MAX" \
    --max_images_per_style "$MAX_IMAGES" \
    --train_temperature    "$TRAIN_TEMPERATURE" \
    --save_every           "$SAVE_EVERY" \
    --log_every            "$LOG_EVERY" \
    --num_workers          "$NUM_WORKERS" \
    --no_normalize_keys    \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished: $(date)"
echo "========================================"
