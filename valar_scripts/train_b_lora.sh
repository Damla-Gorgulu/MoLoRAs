#!/bin/bash
#SBATCH --job-name=B-LoRA-Training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=24:00:00
#SBATCH --output=/logs/b_lora_training/b-lora-training-%J.log
#SBATCH --error=/logs/b_lora_training/b-lora-training-%J.err

# Parse command-line arguments
INSTANCE_DIR=$1
OUTPUT_DIR=$2
PROMPT=$3

# Validate arguments
if [ -z "$INSTANCE_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$PROMPT" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch train_b_lora.sh <instance_data_dir> <output_dir> <instance_prompt>"
    exit 1
fi

# Ensure log directory exists
mkdir -p /logs/b_lora_training  

echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="
echo "Instance directory: $INSTANCE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Instance prompt: $PROMPT"
echo "========================================="

# Load modules
module load conda3/latest
module load cuda/11.8

# Navigate to B-LoRA directory (relative to repo root)
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
cd "$REPO_ROOT/B-LoRA_files"

conda activate B-LoRA_2 

# Set GPU device
export CUDA_VISIBLE_DEVICES=0

echo ""
echo "========================================="
echo "Starting B-LoRA training"
echo "========================================="

accelerate launch --num_processes=1 train_dreambooth_b-lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --instance_data_dir="$INSTANCE_DIR" \
    --output_dir="$OUTPUT_DIR" \
    --instance_prompt="$PROMPT" \
    --resolution=1024 \
    --rank=8 \
    --train_batch_size=1 \
    --learning_rate=5e-5 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=1000 \
    --checkpointing_steps=1000 \
    --seed=0 \
    --gradient_checkpointing \
    --mixed_precision="fp16"

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
