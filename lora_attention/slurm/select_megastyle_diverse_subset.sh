#!/bin/bash
#SBATCH --job-name=MegaStyle-Select200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --qos=ai
#SBATCH --time=02:00:00
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
SCRIPT="$REPO_ROOT/lora_attention/scripts/select_megastyle_diverse_subset.py"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

EMBEDDINGS_PATH="${EMBEDDINGS_PATH:?EMBEDDINGS_PATH required}"
METADATA_PATH="${METADATA_PATH:?METADATA_PATH required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
K="${K:-200}"

python "$SCRIPT" \
  --embeddings_path "$EMBEDDINGS_PATH" \
  --metadata_path "$METADATA_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --k "$K"
