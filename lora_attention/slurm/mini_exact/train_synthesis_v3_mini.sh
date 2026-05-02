#!/bin/bash
#SBATCH --job-name=MoELoRA-V3-Synth
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
ROUTING_CKPT="${ROUTING_CKPT:-/scratch/eyavuz21/lora_attention/mini_v3_clip_strong/latest.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/eyavuz21/lora_attention/mini_v3_synth_${IMAGE_ENCODER}}"

python "$REPO_ROOT/lora_attention/train_synthesis_v3_mini.py" \
  --image_encoder "$IMAGE_ENCODER" \
  --routing_checkpoint "$ROUTING_CKPT" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "${MAX_STEPS:-300}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --views_per_style "${VIEW_COUNT:-32}" \
  --rank_tokens "${RANK_TOKENS:-32}" \
  --max_tensor_groups "${TENSOR_GROUPS:-16}" \
  --query_layers "${QUERY_LAYERS:-3}" \
  --key_layers "${KEY_LAYERS:-3}" \
  --save_every "${SAVE_EVERY:-100}" \
  --log_every "${LOG_EVERY:-25}"
