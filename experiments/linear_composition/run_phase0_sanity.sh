#!/bin/bash
#SBATCH --job-name=lc_phase0_sanity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/dgorgulu21/mo-lora/MoLoRAs/experiments/linear_composition/logs/phase0_sanity_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgorgulu21@ku.edu.tr

# ============================================================
# Phase 0 — Sanity Check: Image Generation
# ============================================================
# Tasks: 0.6–0.10 from TODO.md
#
# This script:
#   1. Generates base image (no LoRA)
#   2. Generates target image via direct weight merge (α=2.0)
#   3. Generates target image via pipe.load_lora_weights()
#   4. Compares injection methods (pixel MSE)
#
# Requires: GPU for SDXL inference
# Expected runtime: ~15 min
#
# PREREQUISITE: run_phase0_extract.sh must have completed.
# ============================================================

export PYTHONUNBUFFERED=1
export EXPERIMENT_DIR=/scratch/dgorgulu21/mo-lora/MoLoRAs/experiments/linear_composition

echo "========================================"
echo "Linear Composition — Phase 0: Sanity Check"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo ""

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate B-LoRA

echo "Python     : $(which python)"
echo "CUDA avail : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU info   :"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null
echo ""

cd "$EXPERIMENT_DIR"

# ============================================================
# Image Generation Sanity Check
# ============================================================
echo "========================================"
echo "Sanity Check: 3 Image Comparison"
echo "========================================"
python sanity_check.py --target-index 0
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: sanity_check.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

echo ""
echo "========================================"
echo "Phase 0 Sanity Check Complete"
echo "========================================"
echo "Generated images:"
ls -lh "$EXPERIMENT_DIR/results/phase0/images/"
echo ""
echo "End time: $(date)"
