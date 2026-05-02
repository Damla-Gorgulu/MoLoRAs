#!/bin/bash
#
# Oracle-copy ablation diagnostic (no training).
# Compares:
#   vanilla
#   direct reference style-block
#   oracle copy through synth path
#   MoE forced GT one-hot synthesis

#SBATCH --job-name=MoELoRA-OracleCopy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --exclude=ai14
#SBATCH --time=03:00:00
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

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
SCRIPT="$REPO_ROOT/lora_attention/diagnostics/oracle_copy_ablation.py"
RUN_TAG="${RUN_TAG:-oracle_copy_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/scratch/eyavuz21/lora_attention/diagnostics/$RUN_TAG}"

STYLE_NAME="${STYLE_NAME:-Baroque}"
PROMPT="${PROMPT:-A dog}"
STYLE_ALPHA="${STYLE_ALPHA:-2.0}"
SEED="${SEED:-42}"
STEPS="${STEPS:-30}"
GUIDANCE="${GUIDANCE:-7.5}"
CHECKPOINT="${CHECKPOINT:-/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt}"
ZOO_DIR="${ZOO_DIR:-/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras}"
STYLE_IMAGES_DIR="${STYLE_IMAGES_DIR:-/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images}"
CACHE_DIR="${CACHE_DIR:-/scratch/eyavuz21/lora_attention}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

echo "OUT_DIR:   $OUT_DIR"
echo "STYLE:     $STYLE_NAME"
echo "PROMPT:    $PROMPT"
echo "ALPHA:     $STYLE_ALPHA"
echo "CKPT:      $CHECKPOINT"
echo "Command:"
echo "$PYTHON $SCRIPT --style_name $STYLE_NAME --prompt $PROMPT --style_alpha $STYLE_ALPHA --seed $SEED --num_inference_steps $STEPS --guidance_scale $GUIDANCE --checkpoint $CHECKPOINT --zoo_dir $ZOO_DIR --style_images_dir $STYLE_IMAGES_DIR --cache_dir $CACHE_DIR --output_dir $OUT_DIR"

"$PYTHON" "$SCRIPT" \
  --style_name "$STYLE_NAME" \
  --prompt "$PROMPT" \
  --style_alpha "$STYLE_ALPHA" \
  --seed "$SEED" \
  --num_inference_steps "$STEPS" \
  --guidance_scale "$GUIDANCE" \
  --checkpoint "$CHECKPOINT" \
  --zoo_dir "$ZOO_DIR" \
  --style_images_dir "$STYLE_IMAGES_DIR" \
  --cache_dir "$CACHE_DIR" \
  --output_dir "$OUT_DIR"

echo "========================================"
echo "Finished: $(date)"
echo "Output:   $OUT_DIR"
echo "========================================"
