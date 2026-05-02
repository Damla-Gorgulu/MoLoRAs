#!/bin/bash
#SBATCH --job-name=LoRA-AE-Overfit16-v4
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/eyavuz21/logs/lora-ae-overfit16-v4-%j.out
#SBATCH --error=/home/eyavuz21/logs/lora-ae-overfit16-v4-%j.err

set -euo pipefail

module load conda3/latest cuda/11.8.0
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONPATH=/home/eyavuz21/repos/MoLoRAs:${PYTHONPATH:-}
WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_API_KEY

cd /home/eyavuz21/repos/MoLoRAs/lora_attention

python train_lora_autoencoder.py \
  --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
  --output_dir /scratch/eyavuz21/lora_autoencoder/overfit16_v4_global_pair_fuse \
  --limit 16 \
  --latent_dim 41472 \
  --d_model 512 \
  --num_layers 4 \
  --num_heads 8 \
  --batch_size 1 \
  --max_steps 3000 \
  --lr 1e-4 \
  --tensor_weight 1.0 \
  --delta_weight 0.5 \
  --cos_weight 5.0 \
  --rel_weight 1.0 \
  --norm_weight 1.0 \
  --log_every 10 \
  --save_every 250 \
  --wandb \
  --wandb_project lora-autoencoder \
  --wandb_run_name overfit16_v4_global_pair_fuse
