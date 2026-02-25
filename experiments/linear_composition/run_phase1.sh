#!/bin/bash
#SBATCH --job-name=lc_phase1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/dgorgulu21/mo-lora/MoLoRAs/experiments/linear_composition/logs/phase1_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgorgulu21@ku.edu.tr

# ============================================================
# Phase 1 — Global Linear Reconstruction
# ============================================================
# Tasks: 1.1–1.13 from TODO.md
#
# This script:
#   1. Self-reconstruction sanity check
#   2. Ridge / Lasso / ElasticNet sweep on 10 representative targets
#   3. Normalization ablation
#   4. Generates comparison images (best/median/worst)
#
# Requires: ~64 GB RAM for matrix, GPU for image generation
# Expected runtime: ~2–4 hours
#
# PREREQUISITE: run_phase0_extract.sh must have completed
#               (needs all_deltaw_matrix.pt)
# ============================================================

export PYTHONUNBUFFERED=1
export EXPERIMENT_DIR=/scratch/dgorgulu21/mo-lora/MoLoRAs/experiments/linear_composition

echo "========================================"
echo "Linear Composition — Phase 1: Global Reconstruction"
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
echo "RAM        : $(free -h | head -2)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null
echo ""

cd "$EXPERIMENT_DIR"

# Check prerequisite
if [ ! -f "$EXPERIMENT_DIR/results/all_deltaw_matrix.pt" ]; then
    echo "ERROR: all_deltaw_matrix.pt not found. Run run_phase0_extract.sh first."
    exit 1
fi

# ============================================================
# STEP 1 — Self-reconstruction check
# ============================================================
echo "========================================"
echo "STEP 1: Self-reconstruction sanity check"
echo "========================================"
python global_reconstruction.py --self-check
STEP1_EXIT=$?
if [ $STEP1_EXIT -ne 0 ]; then
    echo "ERROR: Self-check failed with exit code $STEP1_EXIT"
    exit $STEP1_EXIT
fi
echo "Step 1 complete at $(date)"
echo ""

# ============================================================
# STEP 2 — Full regression sweep + normalization ablation
# ============================================================
echo "========================================"
echo "STEP 2: Regression sweep + normalization ablation"
echo "========================================"
python global_reconstruction.py --normalize
STEP2_EXIT=$?
if [ $STEP2_EXIT -ne 0 ]; then
    echo "ERROR: Regression sweep failed with exit code $STEP2_EXIT"
    exit $STEP2_EXIT
fi
echo "Step 2 complete at $(date)"
echo ""

# ============================================================
# STEP 3 — Generate comparison images
# ============================================================
echo "========================================"
echo "STEP 3: Comparison images (best/median/worst)"
echo "========================================"
python global_reconstruction.py --generate-images
STEP3_EXIT=$?
if [ $STEP3_EXIT -ne 0 ]; then
    echo "WARNING: Image generation failed with exit code $STEP3_EXIT"
    echo "Regression results are still valid."
fi
echo "Step 3 complete at $(date)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "Phase 1 Complete"
echo "========================================"
echo "Result files:"
ls -lh "$EXPERIMENT_DIR/results/phase1/"
echo ""
echo "Coefficients:"
ls "$EXPERIMENT_DIR/results/phase1/coefficients/" | wc -l
echo " coefficient files saved"
echo ""
echo "Images:"
ls -lh "$EXPERIMENT_DIR/results/phase1/images/" 2>/dev/null
echo ""
echo "End time: $(date)"
