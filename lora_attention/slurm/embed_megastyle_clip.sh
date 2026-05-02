#!/bin/bash
#SBATCH --job-name=MegaStyle-EmbedCLIP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --qos=ai
#SBATCH --time=12:00:00
#SBATCH --exclude=ai14
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest
source activate B-LoRA_2 || conda activate B-LoRA_2
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/scripts/embed_megastyle_clip.py"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-/scratch/eyavuz21/datasets/MegaStyle-1.4M}"
MANIFEST_PATH="${MANIFEST_PATH:?MANIFEST_PATH required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
BATCH_SIZE="${BATCH_SIZE:-64}"

python "$SCRIPT" \
  --dataset_root "$DATASET_ROOT" \
  --manifest_path "$MANIFEST_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE"
