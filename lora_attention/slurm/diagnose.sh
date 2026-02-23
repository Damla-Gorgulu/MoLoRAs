#!/bin/bash
#SBATCH --job-name=MoELoRA-Diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=01:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail

module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest
source activate B-LoRA_2 || conda activate B-LoRA_2

cd /home/eyavuz21/repos/MoLoRAs

# Add B-LoRA-fresh to path for blora_utils import
export PYTHONPATH="/home/eyavuz21/repos/B-LoRA-fresh/B-LoRA:$PYTHONPATH"

python lora_attention/diagnose_injection.py
