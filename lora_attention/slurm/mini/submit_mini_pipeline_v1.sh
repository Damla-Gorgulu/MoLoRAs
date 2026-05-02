#!/bin/bash
#
# Prepare and submit the mini Stage 1 train -> validation chain.
#
# This is the fastest way to tell whether the router is learning anything on
# seen styles without spending a full 109-expert SDXL run.
#
# Usage:
#   bash slurm/mini/submit_mini_pipeline_v1.sh
#

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
PREP="$REPO_ROOT/lora_attention/scripts/prepare_mini_experiment.py"
TRAIN="$REPO_ROOT/lora_attention/slurm/mini/train_stage1_mini_v1.sh"
VAL="$REPO_ROOT/lora_attention/slurm/mini/validate_stage1_mini_v1.sh"

python3 "$PREP"

JOB_TRAIN=$(sbatch --parsable "$TRAIN")
JOB_VAL=$(sbatch --parsable --dependency=afterok:"$JOB_TRAIN" "$VAL")

echo "Submitted mini pipeline:"
echo "  train:     $JOB_TRAIN"
echo "  validate:  $JOB_VAL (afterok:$JOB_TRAIN)"
