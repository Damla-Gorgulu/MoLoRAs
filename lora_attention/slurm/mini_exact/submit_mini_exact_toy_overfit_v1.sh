#!/bin/bash
#
# Follow-up for the "bad" Stage 1 outcome:
# prepare a 2-style toy setup and force the router to overfit it.
#

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
PREP="$REPO_ROOT/lora_attention/scripts/prepare_mini_exact_experiment.py"
TRAIN="$REPO_ROOT/lora_attention/slurm/mini_exact/train_stage1_mini_exact_v1.sh"
VAL="$REPO_ROOT/lora_attention/slurm/mini_exact/validate_stage1_mini_exact_v1.sh"
ROOT="/scratch/eyavuz21/lora_attention/mini_exact_toy_v1"

python3 "$PREP" \
    --root "$ROOT" \
    --styles Baroque Cubism \
    --train_views_per_style 128 \
    --val_views_per_style 32

JOB_TRAIN=$(sbatch --parsable \
    --export=ALL,ROOT="$ROOT",OUTPUT_SUBDIR=stage1_overfit,MAX_STEPS=400,VIEW_COUNT=128,POOL_MIN=2,POOL_MAX=2,LR=3e-4,TRAIN_TEMPERATURE=0.7 \
    "$TRAIN")

JOB_VAL=$(sbatch --parsable --dependency=afterok:"$JOB_TRAIN" \
    --export=ALL,ROOT="$ROOT",TRAIN_OUTPUT_SUBDIR=stage1_overfit,VAL_OUTPUT_SUBDIR=stage1_overfit_validation,STAGE1_CKPT="$ROOT/outputs/stage1_overfit/latest.pt" \
    "$VAL")

echo "Submitted bad-outcome toy overfit follow-up:"
echo "  train:     $JOB_TRAIN"
echo "  validate:  $JOB_VAL (afterok:$JOB_TRAIN)"
