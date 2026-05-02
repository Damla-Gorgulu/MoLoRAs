#!/bin/bash
#
# Tiny exact-exemplar Stage 2 follow-up.
#
# Intended for the "good Stage 1" case where exact routing has already shown a
# strong signal and we want to test whether that signal survives diffusion loss.
#

#SBATCH --job-name=MoELoRA-Mini-Exact-S2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
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

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage2_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ROOT="${ROOT:-/scratch/eyavuz21/lora_attention/mini_exact_v1}"
ZOO_DIR="$ROOT/zoo/bloras"
CACHE_DIR="$ROOT/cache"
MANIFEST="$ROOT/manifest.json"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/stage2_exact_followup}"
STAGE1_CKPT="${STAGE1_CKPT:-$ROOT/outputs/stage1/latest.pt}"

mkdir -p "$OUTPUT_DIR"

MAX_STEPS="${MAX_STEPS:-250}"
LR="${LR:-2e-5}"
POOL_MIN="${POOL_MIN:-3}"
POOL_MAX="${POOL_MAX:-3}"
VIEW_COUNT="${VIEW_COUNT:-32}"
SAVE_EVERY="${SAVE_EVERY:-125}"
LOG_EVERY="${LOG_EVERY:-10}"
PROMPT_MODE="${PROMPT_MODE:-neutral}"
LAMBDA_START="${LAMBDA_START:-0.03}"
LAMBDA_END="${LAMBDA_END:-0.0}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: exact manifest not found: $MANIFEST"
    exit 1
fi

if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "ERROR: Stage 1 checkpoint not found: $STAGE1_CKPT"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --zoo_dir "$ZOO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --stage1_ckpt "$STAGE1_CKPT" \
    --exact_manifest_path "$MANIFEST" \
    --exact_views_per_style "$VIEW_COUNT" \
    --exact_prompt_mode "$PROMPT_MODE" \
    --min_pool_size "$POOL_MIN" \
    --max_pool_size "$POOL_MAX" \
    --max_steps "$MAX_STEPS" \
    --lr "$LR" \
    --save_every "$SAVE_EVERY" \
    --log_every "$LOG_EVERY" \
    --lambda_start "$LAMBDA_START" \
    --lambda_end "$LAMBDA_END"

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
