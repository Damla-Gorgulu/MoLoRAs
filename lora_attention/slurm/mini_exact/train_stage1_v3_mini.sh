#!/bin/bash
#
# MoELoRA v3 mini Stage 1: learned image-query + LoRA-weight-key routing.
#

#SBATCH --job-name=MoELoRA-V3-Mini-S1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --exclude=ai14
#SBATCH --time=06:00:00
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

source activate B-LoRA_2 || conda activate B-LoRA_2
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage1_v3_mini.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ROOT="${ROOT:-/scratch/eyavuz21/lora_attention/mini_exact_v1}"
ZOO_DIR="${ZOO_DIR:-$ROOT/zoo/bloras}"
CACHE_DIR="${CACHE_DIR:-$ROOT/cache_v3}"
MANIFEST="${MANIFEST:-$ROOT/manifest.json}"
IMAGE_ENCODER="${IMAGE_ENCODER:-clip}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/eyavuz21/lora_attention/mini_v3_${IMAGE_ENCODER}}"

MAX_STEPS="${MAX_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-2e-4}"
RANK_TOKENS="${RANK_TOKENS:-16}"
TENSOR_GROUPS="${TENSOR_GROUPS:-8}"
QUERY_LAYERS="${QUERY_LAYERS:-2}"
KEY_LAYERS="${KEY_LAYERS:-2}"
VIEW_COUNT="${VIEW_COUNT:-32}"
LOG_EVERY="${LOG_EVERY:-25}"
SAVE_EVERY="${SAVE_EVERY:-250}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: exact manifest not found: $MANIFEST"
    echo "Run: python lora_attention/scripts/prepare_mini_exact_experiment.py"
    exit 1
fi

python "$SCRIPT" \
    --zoo_dir "$ZOO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --manifest_path "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --image_encoder "$IMAGE_ENCODER" \
    --rank_tokens "$RANK_TOKENS" \
    --max_tensor_groups "$TENSOR_GROUPS" \
    --query_layers "$QUERY_LAYERS" \
    --key_layers "$KEY_LAYERS" \
    --views_per_style "$VIEW_COUNT" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --log_every "$LOG_EVERY" \
    --save_every "$SAVE_EVERY" \
    --force_rebuild_cache

echo "========================================"
echo "Finished:  $(date)"
echo "Output:    $OUTPUT_DIR"
echo "========================================"
