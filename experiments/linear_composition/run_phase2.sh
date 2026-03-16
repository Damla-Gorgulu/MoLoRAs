#!/bin/bash
#SBATCH --job-name=lc_phase2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/eyavuz21/mo-lora/experiments/linear_composition/logs/phase2_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyavuz21@ku.edu.tr

# ============================================================
# Phase 2 — Layer-wise Linear Reconstruction
# ============================================================
# Tasks: 2.1–2.14 from TODO.md
#
# This script:
#   1. Grouping A: self-attn vs cross-attn
#   2. Grouping B: early/mid/late × attn type (6 groups)
#   3. Grouping C: by projection type (q/k/v/out)
#   4. Per-tensor regression (upper bound)
#   5. Comparison images
#
# Requires: ~64 GB RAM, GPU for images
# Expected runtime: ~3–5 hours
#
# PREREQUISITE: run_phase0_extract.sh + run_phase1.sh completed
# ============================================================

export PYTHONUNBUFFERED=1
export EXPERIMENT_DIR=/scratch/eyavuz21/mo-lora/experiments/linear_composition
export CODE_DIR=/home/eyavuz21/repos/MoLoRAs/experiments/linear_composition

echo "========================================"
echo "Linear Composition — Phase 2: Layer-wise Reconstruction"
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

cd "$CODE_DIR"

# Check prerequisites
if [ ! -f "$EXPERIMENT_DIR/results/all_deltaw_matrix.pt" ]; then
    echo "ERROR: all_deltaw_matrix.pt not found. Run run_phase0_extract.sh first."
    exit 1
fi
if [ ! -f "$EXPERIMENT_DIR/results/phase1/best_methods.json" ]; then
    echo "WARNING: Phase 1 best_methods.json not found. Will use default alpha=1.0"
fi

# ============================================================
# STEP 1 — All grouping schemes
# ============================================================
echo "========================================"
echo "STEP 1: Grouping schemes A, B, C"
echo "========================================"
python layerwise_reconstruction.py
STEP1_EXIT=$?
if [ $STEP1_EXIT -ne 0 ]; then
    echo "ERROR: Layer-wise reconstruction failed with exit code $STEP1_EXIT"
    exit $STEP1_EXIT
fi
echo "Step 1 complete at $(date)"
echo ""

# ============================================================
# STEP 2 — Per-tensor regression (upper bound)
# ============================================================
echo "========================================"
echo "STEP 2: Per-tensor regression"
echo "========================================"
python layerwise_reconstruction.py --per-tensor
STEP2_EXIT=$?
if [ $STEP2_EXIT -ne 0 ]; then
    echo "ERROR: Per-tensor regression failed with exit code $STEP2_EXIT"
    exit $STEP2_EXIT
fi
echo "Step 2 complete at $(date)"
echo ""

# ============================================================
# STEP 3 — Comparison images
# ============================================================
echo "========================================"
echo "STEP 3: Comparison images"
echo "========================================"
python layerwise_reconstruction.py --generate-images
STEP3_EXIT=$?
if [ $STEP3_EXIT -ne 0 ]; then
    echo "WARNING: Image generation failed with exit code $STEP3_EXIT"
fi
echo "Step 3 complete at $(date)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "Phase 2 Complete"
echo "========================================"
echo "Result files:"
ls -lh "$EXPERIMENT_DIR/results/phase2/"
echo ""
echo "Images:"
ls -lh "$EXPERIMENT_DIR/results/phase2/images/" 2>/dev/null
echo ""
echo "End time: $(date)"
