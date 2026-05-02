#!/bin/bash
#
# Prepare and submit the mini exact-exemplar routing pipeline.
#

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
PREP="$REPO_ROOT/lora_attention/scripts/prepare_mini_exact_experiment.py"
TRAIN="$REPO_ROOT/lora_attention/slurm/mini_exact/train_stage1_mini_exact_v1.sh"
VAL="$REPO_ROOT/lora_attention/slurm/mini_exact/validate_stage1_mini_exact_v1.sh"

python3 "$PREP"

JOB_TRAIN=$(sbatch --parsable "$TRAIN")
JOB_VAL=$(sbatch --parsable --dependency=afterok:"$JOB_TRAIN" "$VAL")

echo "Submitted mini exact pipeline:"
echo "  train:     $JOB_TRAIN"
echo "  validate:  $JOB_VAL (afterok:$JOB_TRAIN)"
