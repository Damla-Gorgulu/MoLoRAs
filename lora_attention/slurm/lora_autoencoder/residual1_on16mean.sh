#!/bin/bash
#SBATCH --job-name=LoRA-AE-Res1on16
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/eyavuz21/logs/lora-ae-res1on16-%j.out
#SBATCH --error=/home/eyavuz21/logs/lora-ae-res1on16-%j.err

set -euo pipefail

module load conda3/latest cuda/11.8.0
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONPATH=/home/eyavuz21/repos/MoLoRAs:${PYTHONPATH:-}

cd /home/eyavuz21/repos/MoLoRAs/lora_attention

python diagnostics/residual1_on_n_mean.py \
  --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
  --mean_limit 16 --train_idx 0 \
  --output_dir /scratch/eyavuz21/lora_autoencoder/residual1_on16mean \
  --max_steps 5000 --lr 1e-4 --clip 0.1 --device cuda
