#!/bin/bash
#
# MoELoRA Stage 2 Training: Hold-out Reconstruction via Diffusion Loss
# Requires full SDXL in fp16 — needs V100-32GB or better.
# GT LoRA is excluded from pool; model learns combination routing.
#
# Usage:
#   sbatch slurm/train_stage2.sh
#
# Prerequisites:
#   Stage 1 checkpoint must exist at:
#   /scratch/eyavuz21/lora_attention/stage1/latest.pt
#
# Override examples:
#   MAX_STEPS=3000 sbatch slurm/train_stage2.sh
#   STAGE1_CKPT=/scratch/eyavuz21/lora_attention/stage1/checkpoint-5000/checkpoint.pt \
#     sbatch slurm/train_stage2.sh
#

#SBATCH --job-name=MoELoRA-Stage2
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
source activate B-LoRA_2 || conda activate B-LoRA_2

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage2.py"
ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
IMAGE_DIRS="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
OUTPUT_DIR="/scratch/eyavuz21/lora_attention/stage2"
LOG_DIR="$REPO_ROOT/lora_attention/logs"
STAGE1_CKPT="${STAGE1_CKPT:-/scratch/eyavuz21/lora_attention/stage1/latest.pt}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Hyperparameters (override via env vars) ─────────────────
MAX_STEPS="${MAX_STEPS:-5000}"
LR="${LR:-5e-5}"
POOL_MIN="${POOL_MIN:-3}"
POOL_MAX="${POOL_MAX:-20}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-25}"
RESUME="${RESUME:-}"

echo "----------------------------------------"
echo "STAGE1_CKPT: $STAGE1_CKPT"
echo "MAX_STEPS:   $MAX_STEPS"
echo "LR:          $LR"
echo "POOL:        [$POOL_MIN, $POOL_MAX]"
echo "LORA_ALPHA:  $LORA_ALPHA"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "----------------------------------------"

# Verify Stage 1 checkpoint exists
if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT"
    echo "Run train_stage1.sh first."
    exit 1
fi

python "$SCRIPT" \
    --zoo_dir       "$ZOO_DIR" \
    --image_dirs    "$IMAGE_DIRS" \
    --cache_dir     "$CACHE_DIR" \
    --output_dir    "$OUTPUT_DIR" \
    --stage1_ckpt   "$STAGE1_CKPT" \
    --max_steps     "$MAX_STEPS" \
    --lr            "$LR" \
    --min_pool_size "$POOL_MIN" \
    --max_pool_size "$POOL_MAX" \
    --lora_alpha    "$LORA_ALPHA" \
    --save_every    "$SAVE_EVERY" \
    --log_every     "$LOG_EVERY" \
    --mixed_precision fp16 \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
