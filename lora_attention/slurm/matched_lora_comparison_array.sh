#!/bin/bash
#
# Parallel matched-ID inference over /scratch/eyavuz21/lora_zoo/_trained_loras.
# Submit after creating an ids file:
#   sbatch --array=0-47%4 lora_attention/slurm/matched_lora_comparison_array.sh

#SBATCH --job-name=MatchedLoRA-Compare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=03:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%A_%a.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
ROOT="${ROOT:-/scratch/eyavuz21/lora_zoo/_trained_loras}"
RUN_TAG="${RUN_TAG:-matched_lora_compare_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-/scratch/eyavuz21/lora_attention/$RUN_TAG}"
IDS_FILE="${IDS_FILE:-$OUT_ROOT/matched_ids.txt}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"

if [[ ! -f "$IDS_FILE" ]]; then
  comm -12 \
    <(find "$ROOT/blora" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
    <(find "$ROOT/unziplora" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
    > "$IDS_FILE"
fi

ID="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$IDS_FILE")"
if [[ -z "${ID:-}" ]]; then
  echo "No ID for array index $SLURM_ARRAY_TASK_ID in $IDS_FILE"
  exit 0
fi

echo "========================================"
echo "Array job: $SLURM_ARRAY_JOB_ID task $SLURM_ARRAY_TASK_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "ID:        $ID"
echo "OUT_ROOT:  $OUT_ROOT"
echo "========================================"

echo "Command:"
echo "$PYTHON $REPO_ROOT/lora_attention/scripts/run_matched_lora_comparisons.py --root $ROOT --output_dir $OUT_ROOT --id $ID ${EXTRA_ARGS:-}"

"$PYTHON" "$REPO_ROOT/lora_attention/scripts/run_matched_lora_comparisons.py" \
  --root "$ROOT" \
  --output_dir "$OUT_ROOT" \
  --id "$ID" \
  ${EXTRA_ARGS:-}

echo "Finished: $(date)"
