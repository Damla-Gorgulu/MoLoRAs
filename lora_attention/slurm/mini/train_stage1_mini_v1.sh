#!/bin/bash
#
# Mini Stage 1: 4-style routing canary with a tiny train/val split.
#
# This uses a dedicated mini zoo and mini WikiArt split prepared by:
#   python lora_attention/scripts/prepare_mini_experiment.py
#
# Usage:
#   sbatch slurm/mini/train_stage1_mini_v1.sh
#

#SBATCH --job-name=MoELoRA-Mini-S1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=04:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================"

module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage1_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

MINI_ROOT="/scratch/eyavuz21/lora_attention/mini_v1"
ZOO_DIR="$MINI_ROOT/zoo/bloras"
CACHE_DIR="$MINI_ROOT/cache"
WIKIART_DIR="$MINI_ROOT/wikiart_train"
LABEL_MAP_PATH="$MINI_ROOT/wikiart_label_map_mini.json"
OUTPUT_DIR="$MINI_ROOT/outputs/stage1"

mkdir -p "$OUTPUT_DIR"

MAX_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TAU_LABEL="${TAU_LABEL:-0.3}"
POOL_MIN="${POOL_MIN:-4}"
POOL_MAX="${POOL_MAX:-4}"
MAX_IMAGES="${MAX_IMAGES:-32}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "MINI_ROOT:  $MINI_ROOT"
echo "MAX_STEPS:  $MAX_STEPS"
echo "LR:         $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "POOL:       [$POOL_MIN, $POOL_MAX]"
echo "MAX_IMAGES: $MAX_IMAGES"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "----------------------------------------"

if [[ ! -f "$LABEL_MAP_PATH" ]]; then
    echo "ERROR: mini label map not found: $LABEL_MAP_PATH"
    echo "Run: python lora_attention/scripts/prepare_mini_experiment.py"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --zoo_dir "$ZOO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --force_rebuild_cache \
    --output_dir "$OUTPUT_DIR" \
    --wikiart_dir "$WIKIART_DIR" \
    --label_map_path "$LABEL_MAP_PATH" \
    --max_steps "$MAX_STEPS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --tau_label "$TAU_LABEL" \
    --min_pool_size "$POOL_MIN" \
    --max_pool_size "$POOL_MAX" \
    --max_images_per_style "$MAX_IMAGES" \
    --save_every "$SAVE_EVERY" \
    --log_every "$LOG_EVERY" \
    --num_workers "$NUM_WORKERS" \
    --seed "$SEED" \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
