#!/bin/bash
#
# Metadata-only MegaStyle shrink analysis.
# No CLIP embeddings yet; this just quantifies how much the search base can be
# reduced using normalized style text and exact (style, content) deduplication.
#

#SBATCH --job-name=MegaStyle-Meta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --time=02:00:00
#SBATCH --exclude=ai14
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================"

module load conda3/latest
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/scripts/analyze_megastyle_metadata.py"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-/scratch/eyavuz21/datasets/MegaStyle-1.4M}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/eyavuz21/datasets/MegaStyle-1.4M_analysis}"
STYLE_CAPS="${STYLE_CAPS:-}"

CMD=(python "$SCRIPT" \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR")

if [[ -n "$STYLE_CAPS" ]]; then
  read -r -a CAPS_ARR <<< "$STYLE_CAPS"
  CMD+=(--style_caps "${CAPS_ARR[@]}")
fi

"${CMD[@]}"

echo "========================================"
echo "Finished:  $(date)"
echo "Output:    $OUTPUT_DIR"
echo "========================================"
