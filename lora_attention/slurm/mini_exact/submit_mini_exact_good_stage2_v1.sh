#!/bin/bash
#
# Follow-up for the "good" Stage 1 outcome:
# run a tiny exact-instance Stage 2 on the same exemplar-tied setup.
#

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
TRAIN="$REPO_ROOT/lora_attention/slurm/mini_exact/train_stage2_mini_exact_v1.sh"
VAL="$REPO_ROOT/lora_attention/slurm/mini_exact/validate_stage2_mini_exact_v1.sh"
ROOT="/scratch/eyavuz21/lora_attention/mini_exact_v1"

JOB_TRAIN=$(sbatch --parsable \
    --export=ALL,ROOT="$ROOT",STAGE1_CKPT="$ROOT/outputs/stage1/latest.pt",OUTPUT_DIR="$ROOT/outputs/stage2_exact_followup",MAX_STEPS=250,VIEW_COUNT=32,PROMPT_MODE=neutral \
    "$TRAIN")

JOB_VAL=$(sbatch --parsable --dependency=afterok:"$JOB_TRAIN" \
    --export=ALL,ROOT="$ROOT",STAGE2_CKPT="$ROOT/outputs/stage2_exact_followup/latest.pt",OUTPUT_DIR="$ROOT/outputs/stage2_exact_followup_validation" \
    "$VAL")

echo "Submitted good-outcome exact Stage 2 follow-up:"
echo "  train:     $JOB_TRAIN"
echo "  validate:  $JOB_VAL (afterok:$JOB_TRAIN)"
