#!/bin/bash
#SBATCH --job-name=LoRA-AE-Resid16
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/eyavuz21/logs/lora-ae-resid16-%j.out
#SBATCH --error=/home/eyavuz21/logs/lora-ae-resid16-%j.err

set -euo pipefail

module load conda3/latest cuda/11.8.0
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONPATH=/home/eyavuz21/repos/MoLoRAs:${PYTHONPATH:-}
WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_API_KEY

cd /home/eyavuz21/repos/MoLoRAs/lora_attention

python train_lora_autoencoder_residual.py \
  --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
  --output_dir /scratch/eyavuz21/lora_autoencoder/residual16_v1 \
  --limit 16 \
  --latent_dim 41472 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 8 \
  --batch_size 1 \
  --max_steps 10000 \
  --lr 1e-4 \
  --clip 0.1 \
  --tensor_weight 1.0 \
  --delta_weight 0.0 \
  --cos_weight 0.0 \
  --rel_weight 0.0 \
  --norm_weight 0.0 \
  --log_every 10 \
  --save_every 1000 \
  --wandb \
  --wandb_project lora-autoencoder \
  --wandb_run_name residual16_v1
