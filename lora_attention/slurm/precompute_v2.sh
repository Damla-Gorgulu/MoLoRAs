#!/bin/bash
#
# v2.0 Phase 0: Pre-compute CLIP similarity matrix + WikiArt label map
# Lightweight CPU job (~10 min). Must run before Stage 1 v2.0.
#
# Usage:
#   sbatch slurm/precompute_v2.sh
#

#SBATCH --job-name=MoELoRA-Precompute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --time=01:00:00
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
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
IMAGE_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
WIKIART_DIR="/home/eyavuz21/datasets/wikiart"
OUTPUT_DIR="/scratch/eyavuz21/lora_attention"

mkdir -p "$OUTPUT_DIR"

echo "Running CLIP similarity + WikiArt label map pre-computation..."
"$PYTHON" -m lora_attention.utils.clip_similarity \
    --zoo_dir    "$ZOO_DIR" \
    --image_dir  "$IMAGE_DIR" \
    --wikiart_dir "$WIKIART_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --device     cpu

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
