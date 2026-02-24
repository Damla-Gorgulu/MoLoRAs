#!/bin/bash
#SBATCH --job-name=diagnose-lora-inject
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/diagnose_lora_inject_%j.out
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/diagnose_lora_inject_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
export PYTHONPATH="${REPO_ROOT}:/home/eyavuz21/repos/B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

echo "=== Node: $(hostname)"
echo "=== Job:  $SLURM_JOB_ID"
echo "=== GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"

cd "$REPO_ROOT"

mkdir -p lora_attention/logs

$PYTHON lora_attention/diagnose_lora_inject.py

echo "=== Done. Check: /scratch/eyavuz21/lora_attention/diagnose_lora_inject/"
