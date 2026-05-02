#!/bin/bash
#SBATCH --job-name=lora_generator
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1    
#SBATCH --partition=ai       
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --time=5:00:00        
#SBATCH --output=logs/job_%j.out    
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ohitit20@ku.edu.tr  

echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

mkdir -p /scratch/ohitit20/logs/lora_generator
mkdir -p /scratch/ohitit20/checkpoints/lora_generator

module load conda3/latest
module load cuda/11.8

conda activate ip-lora

export CUDA_VISIBLE_DEVICES=0

export WANDB_API_KEY="wandb_v1_T5rkRLL9eC5pHTtVsVb3ilqyaDW_L7Y6X0kgFjk2
SeHOefqlZ9nEj4GJgm9TMCSNioPJllR3WlOWF"

export WANDB_MODE=offline

REPO_ROOT="/scratch/ohitit20/MoLoRAs"
cd "$REPO_ROOT"

echo "Starting LoRA Generator training..."

python -m lora_generator.train \
    --checkpoint_dir /scratch/ohitit20/checkpoints/blora_universal_style_token \
    --image_dir      /scratch/ohitit20/ref_images/lora_generator/train \
    --sdxl_model_id  stabilityai/stable-diffusion-xl-base-1.0 \
    --vae_model_id   madebyollin/sdxl-vae-fp16-fix \
    --clip_model_id  openai/clip-vit-large-patch14 \
    --output_dir     /scratch/ohitit20/checkpoints/lora_generator/run_${SLURM_JOB_ID} \
    --d_model        512 \
    --n_layers       2 \
    --n_heads        4 \
    --lr             1e-4 \
    --weight_decay   1e-4 \
    --epochs         100 \
    --clip_ramp_epochs 10 \
    --batch_size     1 \
    --lambda_clip    1.0 \
    --clip_loss_warmup_epochs 5 \
    --generation_prompt "A dog in [v] style" \
    --num_inference_steps 50 \
    --generation_height 128 \
    --generation_width  128 \
    --lora_scale     1.0 \
    --save_every     10 \
    --save_training_images \
    --save_training_images_every 1 \
    --wandb \
    --wandb_project  lora-generator \
    --wandb_run_name "run-$SLURM_JOB_ID"

echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
