#!/bin/bash
#SBATCH --job-name=MoELoRA-V3-Synth-Inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_t4:1
#SBATCH --qos=ai
#SBATCH --time=02:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest
source activate B-LoRA_2 || conda activate B-LoRA_2
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

IMAGE_ENCODER="${IMAGE_ENCODER:-clip}"
ROOT_OUT="${ROOT_OUT:-/scratch/eyavuz21/lora_attention/mini_v3_synth_${IMAGE_ENCODER}}"
CHECKPOINT="${CHECKPOINT:-$ROOT_OUT/latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_OUT/inference}"

python "$REPO_ROOT/lora_attention/inference_synthesis_v3_mini.py" \
  --checkpoint "$CHECKPOINT" \
  --output_dir "$OUTPUT_DIR" \
  --prompt "${PROMPT:-A dog}" \
  --style_alpha "${STYLE_ALPHA:-2.0}" \
  --steps "${STEPS:-20}" \
  --guidance "${GUIDANCE:-7.5}"
