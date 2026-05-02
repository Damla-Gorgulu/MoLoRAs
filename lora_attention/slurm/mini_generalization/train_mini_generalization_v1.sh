#!/bin/bash
#
# Mini generalization training run.
#
# Trains Stage 1 v2.1 on a compact WikiArt subset so we can later replay the
# neutral generalization benchmark against a checkpoint that was actually
# trained for this mini setting.
#
# The goal is not exact-instance retrieval. We want a small but real training
# run that can tell us whether the router learns on a limited style subset and
# whether the resulting checkpoint generalizes under the neutral visibility
# benchmark.
#
# Usage:
#   sbatch lora_attention/slurm/mini_generalization/train_mini_generalization_v1.sh

#SBATCH --job-name=MoELoRA-MiniGenTrain
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
LABEL_MAP_PATH="${LABEL_MAP_PATH:-$REPO_ROOT/lora_attention/configs/wikiart_label_map_mini_generalization_v1.json}"

ROOT_BASE="${ROOT_BASE:-/scratch/eyavuz21/lora_attention/mini_generalization_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_BASE/stage1_train}"
mkdir -p "$OUTPUT_DIR"

MAX_STEPS="${MAX_STEPS:-1500}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
TAU_LABEL="${TAU_LABEL:-0.3}"
POOL_MIN="${POOL_MIN:-5}"
POOL_MAX="${POOL_MAX:-10}"
MAX_IMAGES="${MAX_IMAGES:-64}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "OUTPUT_DIR:       $OUTPUT_DIR"
echo "LABEL_MAP_PATH:   $LABEL_MAP_PATH"
echo "MAX_STEPS:        $MAX_STEPS"
echo "LR:               $LR"
echo "POOL:             [$POOL_MIN, $POOL_MAX]"
echo "MAX_IMAGES:       $MAX_IMAGES"
echo "TRAIN_TEMPERATURE:$TRAIN_TEMPERATURE"
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
