#!/bin/bash
#
# MoELoRA Stage 1 Training: Ground Truth Mapping
# Trains the RoutingMLP with MSE loss on rank-level attention vs one-hot GT.
# No SDXL required — lightweight, runs on T4 or V100.
#
# Usage:
#   sbatch slurm/train_stage1.sh
#
# Override examples:
#   MAX_STEPS=20000 sbatch slurm/train_stage1.sh
#   POOL_MAX=10 sbatch slurm/train_stage1.sh
#

#SBATCH --job-name=MoELoRA-Stage1
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
source activate B-LoRA_2 || conda activate B-LoRA_2

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage1.py"
ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
IMAGE_DIRS="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
OUTPUT_DIR="/scratch/eyavuz21/lora_attention/stage1"
LOG_DIR="$REPO_ROOT/lora_attention/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Hyperparameters (override via env vars) ─────────────────
MAX_STEPS="${MAX_STEPS:-10000}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
POOL_MIN="${POOL_MIN:-3}"
POOL_MAX="${POOL_MAX:-20}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-50}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "MAX_STEPS:  $MAX_STEPS"
echo "LR:         $LR"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "POOL:       [$POOL_MIN, $POOL_MAX]"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "----------------------------------------"

# ── Build cache if not present ──────────────────────────────
echo "Building / verifying LoRA pool cache..."
python "$SCRIPT" \
    --zoo_dir       "$ZOO_DIR" \
    --image_dirs    "$IMAGE_DIRS" \
    --cache_dir     "$CACHE_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --max_steps     "$MAX_STEPS" \
    --lr            "$LR" \
    --batch_size    "$BATCH_SIZE" \
    --min_pool_size "$POOL_MIN" \
    --max_pool_size "$POOL_MAX" \
    --save_every    "$SAVE_EVERY" \
    --log_every     "$LOG_EVERY" \
    --num_workers   "$NUM_WORKERS" \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
