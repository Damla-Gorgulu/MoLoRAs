#!/bin/bash
#SBATCH --job-name=LoRA-AE-Eval16
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/eyavuz21/logs/lora-ae-eval16-%j.out
#SBATCH --error=/home/eyavuz21/logs/lora-ae-eval16-%j.err

set -euo pipefail

module load conda3/latest cuda/11.8.0
source activate B-LoRA_2 || conda activate B-LoRA_2

export PYTHONPATH=/home/eyavuz21/repos/MoLoRAs:${PYTHONPATH:-}

cd /home/eyavuz21/repos/MoLoRAs/lora_attention

python diagnostics/eval_lora_autoencoder_overfit.py \
  --checkpoint /scratch/eyavuz21/lora_autoencoder/overfit16_v1/latest.pt \
  --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
  --output_dir /scratch/eyavuz21/lora_autoencoder_eval/overfit16_v1 \
  --limit 16 \
  --indices 0 1 2 \
  --prompt "A [v] dog" \
  --seed 42 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --style_alpha 1.0
