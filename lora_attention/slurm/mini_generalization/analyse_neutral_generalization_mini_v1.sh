#!/bin/bash
#
# Post-run analysis for the neutral-first mini generalization replay.
#

#SBATCH --job-name=MoELoRA-NeutralMiniGen-Analyse
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --time=01:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "========================================"

module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"

ROOT_BASE="${ROOT_BASE:-/scratch/eyavuz21/lora_attention/neutral_generalization_mini_v1}"
CSV_PATH="${CSV_PATH:-$ROOT_BASE/report.csv}"

mkdir -p "$(dirname "$CSV_PATH")"

"$PYTHON" "$REPO_ROOT/lora_attention/analyse_generalization.py" \
    --results_dir "$ROOT_BASE" \
    --csv "$CSV_PATH"

echo "========================================"
echo "Finished: $(date)"
echo "CSV:      $CSV_PATH"
echo "========================================"
