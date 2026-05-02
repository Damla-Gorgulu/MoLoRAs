#!/bin/bash
#
# Exact-instance routing validation for the mini Stage 1 run.
#

#SBATCH --job-name=MoELoRA-Mini-Exact-S1-Val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=02:00:00
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
SCRIPT="$REPO_ROOT/lora_attention/validate_stage1_mini_exact.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ROOT="${ROOT:-/scratch/eyavuz21/lora_attention/mini_exact_v1}"
TRAIN_OUTPUT_SUBDIR="${TRAIN_OUTPUT_SUBDIR:-stage1}"
VAL_OUTPUT_SUBDIR="${VAL_OUTPUT_SUBDIR:-stage1_validation}"
CKPT="${STAGE1_CKPT:-$ROOT/outputs/$TRAIN_OUTPUT_SUBDIR/latest.pt}"
MANIFEST="$ROOT/manifest.json"
ZOO_DIR="$ROOT/zoo/bloras"
CACHE_DIR="$ROOT/cache"
OUTPUT_DIR="$ROOT/outputs/$VAL_OUTPUT_SUBDIR"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --checkpoint "$CKPT" \
    --root "$ROOT" \
    --zoo_dir "$ZOO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --manifest_path "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --views_per_style 8 \
    --temperature 1.0

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
