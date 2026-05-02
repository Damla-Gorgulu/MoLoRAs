#!/bin/bash
#
# Chained post-train validation for the tiny exact-instance Stage 2 follow-up.
#

#SBATCH --job-name=MoELoRA-Mini-Exact-S2-Val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
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
SCRIPT="$REPO_ROOT/lora_attention/validate_stage2_mini_exact.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ROOT="${ROOT:-/scratch/eyavuz21/lora_attention/mini_exact_v1}"
CKPT="${STAGE2_CKPT:-$ROOT/outputs/stage2_exact_followup/latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/stage2_exact_followup_validation}"

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --checkpoint "$CKPT" \
    --root "$ROOT" \
    --manifest_path "$ROOT/manifest.json" \
    --zoo_dir "$ROOT/zoo/bloras" \
    --cache_dir "$ROOT/cache" \
    --output_dir "$OUTPUT_DIR"

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
