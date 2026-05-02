#!/bin/bash
#SBATCH --job-name=LoRA-AE-MetaDec-lr4
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/eyavuz21/logs/lora-ae-metadec-lr4-%j.out
#SBATCH --error=/home/eyavuz21/logs/lora-ae-metadec-lr4-%j.err

set -euo pipefail

module load conda3/latest cuda/11.8.0
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONPATH=/home/eyavuz21/repos/MoLoRAs:${PYTHONPATH:-}

cd /home/eyavuz21/repos/MoLoRAs/lora_attention

python diagnostics/meta_decoder_baseline.py \
  --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
  --limit 1 --idx 0 \
  --output_dir /scratch/eyavuz21/lora_autoencoder/meta_decoder_lr1e4 \
  --max_steps 5000 --lr 1e-4 --clip 0.1 --log_every 10 --device cuda
