#!/bin/bash
#
# Validate a MoELoRA v3 mini Stage-1 checkpoint.
#

#SBATCH --job-name=MoELoRA-V3-Mini-Val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --exclude=ai14
#SBATCH --time=02:00:00
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
SCRIPT="$REPO_ROOT/lora_attention/validate_stage1_v3_mini.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

IMAGE_ENCODER="${IMAGE_ENCODER:-clip}"
ROOT_OUT="${ROOT_OUT:-/scratch/eyavuz21/lora_attention/mini_v3_${IMAGE_ENCODER}}"
CHECKPOINT="${CHECKPOINT:-$ROOT_OUT/latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_OUT/validation}"
TEMPERATURE="${TEMPERATURE:-1.0}"
VIEW_COUNT="${VIEW_COUNT:-8}"

python "$SCRIPT" \
    --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --views_per_style "$VIEW_COUNT"

echo "========================================"
echo "Finished:  $(date)"
echo "Output:    $OUTPUT_DIR"
echo "========================================"
