#!/bin/bash
#
# Follow-up for the "partial" Stage 1 outcome:
# rerun exact-instance Stage 1 with sharper training and more augmented views.
#

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
TRAIN="$REPO_ROOT/lora_attention/slurm/mini_exact/train_stage1_mini_exact_v1.sh"
VAL="$REPO_ROOT/lora_attention/slurm/mini_exact/validate_stage1_mini_exact_v1.sh"
ROOT="/scratch/eyavuz21/lora_attention/mini_exact_v1"
TRAIN_SUBDIR="stage1_sharp"
VAL_SUBDIR="stage1_sharp_validation"

JOB_TRAIN=$(sbatch --parsable \
    --export=ALL,ROOT="$ROOT",OUTPUT_SUBDIR="$TRAIN_SUBDIR",MAX_STEPS=1500,VIEW_COUNT=64,LR=2e-4,TRAIN_TEMPERATURE=0.6 \
    "$TRAIN")

JOB_VAL=$(sbatch --parsable --dependency=afterok:"$JOB_TRAIN" \
    --export=ALL,ROOT="$ROOT",TRAIN_OUTPUT_SUBDIR="$TRAIN_SUBDIR",VAL_OUTPUT_SUBDIR="$VAL_SUBDIR",STAGE1_CKPT="$ROOT/outputs/$TRAIN_SUBDIR/latest.pt" \
    "$VAL")

echo "Submitted partial-outcome exact Stage 1 follow-up:"
echo "  train:     $JOB_TRAIN"
echo "  validate:  $JOB_VAL (afterok:$JOB_TRAIN)"
