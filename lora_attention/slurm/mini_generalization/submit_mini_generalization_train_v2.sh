#!/bin/bash
#
# Submit the tightened singleton-only mini generalization Stage 1 canary,
# then chain the in-pool-only neutral replay behind it.

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
TRAIN_SCRIPT="$REPO_ROOT/lora_attention/slurm/mini_generalization/train_mini_generalization_v2.sh"
EVAL_SCRIPT="$REPO_ROOT/lora_attention/slurm/mini_generalization/neutral_generalization_mini_v2.sh"

ROOT_BASE="${ROOT_BASE:-/scratch/eyavuz21/lora_attention/mini_generalization_v2}"
TRAIN_OUT="${TRAIN_OUT:-$ROOT_BASE/stage1_train}"
RUN_TAG="${RUN_TAG:-neutral_mini_v2}"
CKPT_TAG="${CKPT_TAG:-stage1_train}"
CKPT_PATH="${CKPT_PATH:-$TRAIN_OUT/latest.pt}"

mkdir -p "$ROOT_BASE"

JOB_TRAIN=$(sbatch --parsable \
    --export=ALL,ROOT_BASE="$ROOT_BASE",OUTPUT_DIR="$TRAIN_OUT" \
    "$TRAIN_SCRIPT")

JOB_EVAL=$(sbatch --parsable \
    --dependency=afterok:"$JOB_TRAIN" \
    --export=ALL,ROOT_BASE="$ROOT_BASE",CKPT="$CKPT_PATH",CKPT_TAG="$CKPT_TAG",RUN_TAG="$RUN_TAG" \
    "$EVAL_SCRIPT")

echo "Submitted mini generalization v2 pipeline:"
echo "  train: $JOB_TRAIN"
echo "  eval:  $JOB_EVAL (afterok:$JOB_TRAIN)"
