#!/bin/bash
# Submit v3 CLIP and VAE mini canaries, each followed by validation.

set -euo pipefail

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
TRAIN="$REPO_ROOT/lora_attention/slurm/mini_exact/train_stage1_v3_mini.sh"
VAL="$REPO_ROOT/lora_attention/slurm/mini_exact/validate_stage1_v3_mini.sh"

ROOT_BASE="/scratch/eyavuz21/lora_attention"

CLIP_OUT="$ROOT_BASE/mini_v3_clip"
VAE_OUT="$ROOT_BASE/mini_v3_vae"

JOB_CLIP=$(sbatch --parsable \
    --export=ALL,IMAGE_ENCODER=clip,OUTPUT_DIR="$CLIP_OUT" \
    "$TRAIN")
JOB_CLIP_VAL=$(sbatch --parsable --dependency=afterok:"$JOB_CLIP" \
    --export=ALL,IMAGE_ENCODER=clip,ROOT_OUT="$CLIP_OUT",OUTPUT_DIR="$CLIP_OUT/validation" \
    "$VAL")

JOB_VAE=$(sbatch --parsable \
    --export=ALL,IMAGE_ENCODER=vae,OUTPUT_DIR="$VAE_OUT" \
    "$TRAIN")
JOB_VAE_VAL=$(sbatch --parsable --dependency=afterok:"$JOB_VAE" \
    --export=ALL,IMAGE_ENCODER=vae,ROOT_OUT="$VAE_OUT",OUTPUT_DIR="$VAE_OUT/validation" \
    "$VAL")

echo "Submitted v3 mini canaries:"
echo "  CLIP train: $JOB_CLIP"
echo "  CLIP val:   $JOB_CLIP_VAL (afterok:$JOB_CLIP)"
echo "  VAE train:  $JOB_VAE"
echo "  VAE val:    $JOB_VAE_VAL (afterok:$JOB_VAE)"
