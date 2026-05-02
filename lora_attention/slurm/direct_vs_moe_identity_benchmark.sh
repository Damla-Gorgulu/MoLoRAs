#!/bin/bash
#
# Tiny follow-up benchmark for query-style identity:
# query | vanilla | direct reference B-LoRA | synth top1 | synth top1 norm-match
# across neutral and style-word dog prompts.

#SBATCH --job-name=MoELoRA-IdentityBench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=04:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "========================================"

module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
RUN_TAG="${RUN_TAG:-identity_benchmark_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-/scratch/eyavuz21/lora_attention/diagnostics/$RUN_TAG}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

echo "OUT_DIR: $OUT_DIR"
echo "Command:"
echo "$PYTHON $REPO_ROOT/lora_attention/scripts/direct_vs_moe_identity_benchmark.py --output_dir $OUT_DIR"

"$PYTHON" "$REPO_ROOT/lora_attention/scripts/direct_vs_moe_identity_benchmark.py" \
  --output_dir "$OUT_DIR"

echo "========================================"
echo "Finished: $(date)"
echo "Output:   $OUT_DIR"
echo "========================================"
