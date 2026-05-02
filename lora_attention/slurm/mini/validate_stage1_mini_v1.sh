#!/bin/bash
#
# Routing-only validation for the mini Stage 1 experiment.
#
# Measures whether the router can retrieve the correct expert on held-out
# seen-style images without loading SDXL.
#
# Usage:
#   sbatch slurm/mini/validate_stage1_mini_v1.sh
#

#SBATCH --job-name=MoELoRA-Mini-S1-Val
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
SCRIPT="$REPO_ROOT/lora_attention/validate_stage1_mini.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

MINI_ROOT="/scratch/eyavuz21/lora_attention/mini_v1"
CKPT="${STAGE1_CKPT:-$MINI_ROOT/outputs/stage1/latest.pt}"
ZOO_DIR="$MINI_ROOT/zoo/bloras"
CACHE_DIR="$MINI_ROOT/cache"
WIKIART_DIR="$MINI_ROOT/wikiart_val"
LABEL_MAP_PATH="$MINI_ROOT/wikiart_label_map_mini.json"
OUTPUT_DIR="$MINI_ROOT/outputs/stage1_validation"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------"
echo "CKPT:      $CKPT"
echo "MINI_ROOT: $MINI_ROOT"
echo "OUTPUT:    $OUTPUT_DIR"
echo "----------------------------------------"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --checkpoint "$CKPT" \
    --zoo_dir "$ZOO_DIR" \
    --cache_dir "$CACHE_DIR" \
    --wikiart_dir "$WIKIART_DIR" \
    --label_map_path "$LABEL_MAP_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --min_pool_size 4 \
    --max_pool_size 4 \
    --max_images_per_style 32 \
    --temperature 1.0

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
