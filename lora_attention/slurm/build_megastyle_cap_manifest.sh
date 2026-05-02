#!/bin/bash
#SBATCH --job-name=MegaStyle-CapManifest
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
module load conda3/latest
source activate B-LoRA_2 || conda activate B-LoRA_2
export PYTHONUNBUFFERED=1

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/scripts/build_megastyle_cap_manifest.py"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASET_ROOT="${DATASET_ROOT:-/scratch/eyavuz21/datasets/MegaStyle-1.4M}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"
CAP="${CAP:-1}"
LIMIT_STYLES="${LIMIT_STYLES:-0}"

python "$SCRIPT" \
  --dataset_root "$DATASET_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --cap "$CAP" \
  --limit_styles "$LIMIT_STYLES"
