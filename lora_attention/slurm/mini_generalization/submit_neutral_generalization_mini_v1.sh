#!/bin/bash
#
# Submit the neutral-first mini generalization replay for three checkpoints:
#   - stage1_v21/latest.pt
#   - stage2_v22/latest.pt
#   - stage2_v23/latest.pt
#
# The analysis job is chained after all three complete.
#

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
BENCHMARK="$REPO_ROOT/lora_attention/slurm/mini_generalization/neutral_generalization_mini_v1.sh"
ANALYSE="$REPO_ROOT/lora_attention/slurm/mini_generalization/analyse_neutral_generalization_mini_v1.sh"

ROOT_BASE="${ROOT_BASE:-/scratch/eyavuz21/lora_attention/neutral_generalization_mini_v1}"
RUN_TAG="${RUN_TAG:-neutral_mini_v1}"

mkdir -p "$ROOT_BASE"

JOB_STAGE1=$(sbatch --parsable \
    --export=ALL,ROOT_BASE="$ROOT_BASE",CKPT="/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt",CKPT_TAG="stage1_v21",RUN_TAG="$RUN_TAG" \
    "$BENCHMARK")

JOB_STAGE2_22=$(sbatch --parsable \
    --export=ALL,ROOT_BASE="$ROOT_BASE",CKPT="/scratch/eyavuz21/lora_attention/stage2_v22/latest.pt",CKPT_TAG="stage2_v22",RUN_TAG="$RUN_TAG" \
    "$BENCHMARK")

JOB_STAGE2_23=$(sbatch --parsable \
    --export=ALL,ROOT_BASE="$ROOT_BASE",CKPT="/scratch/eyavuz21/lora_attention/stage2_v23/latest.pt",CKPT_TAG="stage2_v23",RUN_TAG="$RUN_TAG" \
    "$BENCHMARK")

JOB_ANALYSE=$(sbatch --parsable \
    --dependency=afterok:"$JOB_STAGE1":"$JOB_STAGE2_22":"$JOB_STAGE2_23" \
    --export=ALL,ROOT_BASE="$ROOT_BASE",CSV_PATH="$ROOT_BASE/report.csv" \
    "$ANALYSE")

echo "Submitted neutral mini generalization replay:"
echo "  stage1_v21:  $JOB_STAGE1"
echo "  stage2_v22:  $JOB_STAGE2_22"
echo "  stage2_v23:  $JOB_STAGE2_23"
echo "  analyse:     $JOB_ANALYSE (afterok:$JOB_STAGE1:$JOB_STAGE2_22:$JOB_STAGE2_23)"
