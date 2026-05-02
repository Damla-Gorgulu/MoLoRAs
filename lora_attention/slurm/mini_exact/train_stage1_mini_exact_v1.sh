#!/bin/bash
#
# Mini exact-exemplar Stage 1.
#
# Uses the exact B-LoRA source images as positives and mild style-preserving
# augmentations on the queries.
#

#SBATCH --job-name=MoELoRA-Mini-Exact-S1
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

ROOT="${ROOT:-/scratch/eyavuz21/lora_attention/mini_exact_v1}"
ZOO_DIR="$ROOT/zoo/bloras"
CACHE_DIR="$ROOT/cache"
MANIFEST="$ROOT/manifest.json"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-stage1}"
OUTPUT_DIR="$ROOT/outputs/$OUTPUT_SUBDIR"

mkdir -p "$OUTPUT_DIR"

MAX_STEPS="${MAX_STEPS:-1000}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POOL_MIN="${POOL_MIN:-4}"
POOL_MAX="${POOL_MAX:-4}"
VIEW_COUNT="${VIEW_COUNT:-32}"
SAVE_EVERY="${SAVE_EVERY:-250}"
LOG_EVERY="${LOG_EVERY:-25}"
NUM_WORKERS="${NUM_WORKERS:-0}"
TRAIN_TEMPERATURE="${TRAIN_TEMPERATURE:-1.0}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: exact manifest not found: $MANIFEST"
    echo "Run: python lora_attention/scripts/prepare_mini_exact_experiment.py"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --zoo_dir "$ZOO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --exact_manifest_path "$MANIFEST" \
    --exact_views_per_style "$VIEW_COUNT" \
    --target_mode ce \
    --min_pool_size "$POOL_MIN" \
    --max_pool_size "$POOL_MAX" \
    --train_temperature "$TRAIN_TEMPERATURE" \
    --max_steps "$MAX_STEPS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --save_every "$SAVE_EVERY" \
    --log_every "$LOG_EVERY" \
    --num_workers "$NUM_WORKERS" \
    --force_rebuild_cache

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
