#!/bin/bash
#SBATCH --job-name=B-LoRA-Training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --gres=gpu:ampere_a40:1
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/ohitit20/logs/job_%J.out
#SBATCH --error=/scratch/ohitit20/logs/job_%J.out

echo "========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================="

# Load modules
module load conda3/latest
module load cuda/11.8

conda activate ip-lora

export CUDA_VISIBLE_DEVICES=0

content="microscope"
WORKDIR="/scratch/ohitit20"
EXPERIMENT_NAME="blora_universal_style_token"
CHECKPOINTS_DIR="${WORKDIR}/checkpoints/${EXPERIMENT_NAME}"

echo ""
echo "========================================="
echo "Starting inference with universal style token"
echo "========================================="

for style in $(ls ${CHECKPOINTS_DIR}); do

    output_path="${WORKDIR}/outputs/${EXPERIMENT_NAME}/${content}/${style}"
    mkdir -p $output_path

    python3 /scratch/ohitit20/MoLoRAs/B-LoRA_files/inference_modular_blocks.py \
        --prompt "A ${content} in [v] style" \
        --output_path $output_path \
        --content_B_LoRA "${WORKDIR}/checkpoints/blora/${content}/pytorch_lora_weights.safetensors" \
        --style_B_LoRA "${CHECKPOINTS_DIR}/${style}/pytorch_lora_weights.safetensors"
done

echo ""
echo "========================================="
echo "Job completed at: $(date)"
echo "========================================="
