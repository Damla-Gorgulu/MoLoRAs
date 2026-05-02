#!/bin/bash
#
# v2.0 Inference: Per-tensor routing with LoRARankEncoder
# Generates styled images from a query style image.
#
# Usage:
#   CHECKPOINT=/scratch/eyavuz21/lora_attention/stage1_v2/latest.pt \
#   STYLE_IMAGE=/path/to/style.jpg \
#   PROMPT="A cat [v]" \
#   sbatch slurm/inference_v2.sh
#

#SBATCH --job-name=MoELoRA-Inf-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=02:00:00
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
# Use explicit path to avoid conda init issues on compute nodes
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
export PYTHONUNBUFFERED=1

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="/scratch/eyavuz21/lora_attention"

# Required env vars
CHECKPOINT="${CHECKPOINT:?ERROR: Set CHECKPOINT=/path/to/latest.pt}"
STYLE_IMAGE="${STYLE_IMAGE:?ERROR: Set STYLE_IMAGE=/path/to/image.jpg}"
PROMPT="${PROMPT:?ERROR: Set PROMPT='your prompt [v]'}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/eyavuz21/lora_attention/inference_v2}"

# Optional
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_K="${TOP_K:-}"
STYLE_ALPHA="${STYLE_ALPHA:-1.0}"
NUM_IMAGES="${NUM_IMAGES:-4}"
QUERY_LABEL="${QUERY_LABEL:-}"
GT_EXPERT="${GT_EXPERT:-}"
EXCLUDE="${EXCLUDE:-}"
PRODUCT_SYNTH="${PRODUCT_SYNTH:-1}"

mkdir -p "$OUTPUT_DIR"

echo "----------------------------------------"
echo "CHECKPOINT:  $CHECKPOINT"
echo "STYLE_IMAGE: $STYLE_IMAGE"
echo "PROMPT:      $PROMPT"
echo "TEMPERATURE: $TEMPERATURE"
echo "OUTPUT_DIR:  $OUTPUT_DIR"
echo "PRODUCT_SYNTH: $PRODUCT_SYNTH"
echo "----------------------------------------"

SYNTH_ARG="--product_synth"
if [[ "$PRODUCT_SYNTH" == "0" ]]; then
    SYNTH_ARG="--legacy_synth"
fi

"$PYTHON" "$SCRIPT" \
    --checkpoint   "$CHECKPOINT" \
    --style_image  "$STYLE_IMAGE" \
    --prompt       "$PROMPT" \
    --output_dir   "$OUTPUT_DIR" \
    --zoo_dir      "$ZOO_DIR" \
    --cache_dir    "$CACHE_DIR" \
    --temperature  "$TEMPERATURE" \
    --style_alpha  "$STYLE_ALPHA" \
    --num_images   "$NUM_IMAGES" \
    $SYNTH_ARG \
    ${TOP_K:+--top_k "$TOP_K"} \
    ${QUERY_LABEL:+--query_label "$QUERY_LABEL"} \
    ${GT_EXPERT:+--gt_expert "$GT_EXPERT"} \
    ${EXCLUDE:+--exclude_experts $EXCLUDE}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
