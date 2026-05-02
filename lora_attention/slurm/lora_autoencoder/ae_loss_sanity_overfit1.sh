#!/bin/bash
#SBATCH --job-name=LoRA-AE-Sanity1
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/eyavuz21/logs/lora-ae-sanity1-%j.out
#SBATCH --error=/home/eyavuz21/logs/lora-ae-sanity1-%j.err

set -euo pipefail

module load conda3/latest cuda/11.8.0
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONPATH=/home/eyavuz21/repos/MoLoRAs:${PYTHONPATH:-}

cd /home/eyavuz21/repos/MoLoRAs/lora_attention

python diagnostics/ae_loss_sanity.py \
  --checkpoint /scratch/eyavuz21/lora_autoencoder/overfit1_debug/latest.pt \
  --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
  --limit 2 \
  --idx 0 \
  --other_idx 1 \
  --output /scratch/eyavuz21/lora_autoencoder/overfit1_debug/sanity_latest.json \
  --device cuda
