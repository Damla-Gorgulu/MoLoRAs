#!/bin/bash
#
# Submit Stage 2 v2.3 training and a chained validation sweep.
#
# This launcher is meant to be run from the login node. It submits the training
# job first, then submits the validation job with an afterok dependency so the
# validation only runs if training finishes cleanly.
#
# Usage:
#   cd /home/eyavuz21/repos/MoLoRAs
#   bash lora_attention/slurm/submit_stage2_v23_pipeline.sh
#

set -euo pipefail

TRAIN_SCRIPT="${TRAIN_SCRIPT:-/home/eyavuz21/repos/MoLoRAs/lora_attention/slurm/train_stage2_v23.sh}"
VALIDATION_SCRIPT="${VALIDATION_SCRIPT:-/home/eyavuz21/repos/MoLoRAs/lora_attention/slurm/validate_stage2_v23.sh}"

TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/scratch/eyavuz21/lora_attention/stage2_v23}"
VALIDATION_OUTPUT_DIR="${VALIDATION_OUTPUT_DIR:-/scratch/eyavuz21/lora_attention/stage2_v23_validation}"

STAGE1_CKPT="${STAGE1_CKPT:-/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt}"

echo "Submitting Stage 2 training..."
TRAIN_JOB=$(sbatch --parsable \
    --export=ALL,OUTPUT_DIR="$TRAIN_OUTPUT_DIR",STAGE1_CKPT="$STAGE1_CKPT" \
    "$TRAIN_SCRIPT")
echo "Train job: $TRAIN_JOB"

echo "Submitting chained validation..."
VALIDATION_JOB=$(sbatch --parsable \
    --dependency=afterok:"$TRAIN_JOB" \
    --export=ALL,STAGE2_CKPT="$TRAIN_OUTPUT_DIR/latest.pt",OUT_ROOT="$VALIDATION_OUTPUT_DIR" \
    "$VALIDATION_SCRIPT")
echo "Validation job: $VALIDATION_JOB"

echo "Pipeline submitted."
echo "Train output: $TRAIN_OUTPUT_DIR"
echo "Validation output: $VALIDATION_OUTPUT_DIR"
