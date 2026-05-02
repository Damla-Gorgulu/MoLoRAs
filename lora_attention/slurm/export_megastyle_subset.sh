#!/bin/bash
#SBATCH --job-name=MegaStyle-Export
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --time=03:00:00
#SBATCH --exclude=ai14
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
module load conda3/latest
source activate B-LoRA_2 || conda activate B-LoRA_2
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/scripts/export_megastyle_subset.py"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-/scratch/eyavuz21/datasets/MegaStyle-1.4M}"
SELECTION_PATH="${SELECTION_PATH:?SELECTION_PATH required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"

python "$SCRIPT" \
  --dataset_root "$DATASET_ROOT" \
  --selection_path "$SELECTION_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --image_size "$IMAGE_SIZE"
