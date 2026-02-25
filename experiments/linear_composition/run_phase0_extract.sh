#!/bin/bash
#SBATCH --job-name=lc_phase0_extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/dgorgulu21/mo-lora/MoLoRAs/experiments/linear_composition/logs/phase0_extract_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dgorgulu21@ku.edu.tr

# ============================================================
# Phase 0 — ΔW Extraction, Validation, and Matrix Build
# ============================================================
# Tasks: 0.1–0.5, I.4, 1.1 from TODO.md
#
# This script:
#   1. Discovers all 109 LoRAs in the pool
#   2. Validates tensor key structure (80 adapters, 160 keys)
#   3. Computes ΔW = B @ A, logs norms
#   4. Builds the full (D × 109) matrix for Phase 1+
#
# Requires: ~64 GB RAM for matrix construction
# GPU: Not strictly required, but reserved for later steps
# Expected runtime: ~30 min
# ============================================================

export PYTHONUNBUFFERED=1
export EXPERIMENT_DIR=/scratch/dgorgulu21/mo-lora/MoLoRAs/experiments/linear_composition

echo "========================================"
echo "Linear Composition — Phase 0: ΔW Extraction"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo ""

# ── Create directories ──────────────────────────────────────
mkdir -p "$EXPERIMENT_DIR/logs"
mkdir -p "$EXPERIMENT_DIR/results/phase0/images"
mkdir -p "$EXPERIMENT_DIR/results/phase1/images"
mkdir -p "$EXPERIMENT_DIR/results/phase1/coefficients"
mkdir -p "$EXPERIMENT_DIR/results/phase2/images"
mkdir -p "$EXPERIMENT_DIR/results/phase3/plots"

# ── Activate conda environment ──────────────────────────────
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate B-LoRA

echo "Python     : $(which python)"
echo "Python ver : $(python --version)"
echo "Conda env  : $CONDA_DEFAULT_ENV"
echo "RAM        : $(free -h | head -2)"
echo ""

cd "$EXPERIMENT_DIR"

# ============================================================
# STEP 1 — Validate LoRA pool & extract ΔW norms
# ============================================================
echo "========================================"
echo "STEP 1: Validate & Extract"
echo "========================================"
python extract_deltaw.py --validate-only
STEP1_EXIT=$?
if [ $STEP1_EXIT -ne 0 ]; then
    echo "ERROR: extract_deltaw.py --validate-only failed with exit code $STEP1_EXIT"
    exit $STEP1_EXIT
fi
echo "Step 1 complete at $(date)"
echo ""

# ============================================================
# STEP 2 — Build full D×109 matrix
# ============================================================
echo "========================================"
echo "STEP 2: Build full ΔW matrix"
echo "========================================"
python extract_deltaw.py --build-matrix
STEP2_EXIT=$?
if [ $STEP2_EXIT -ne 0 ]; then
    echo "ERROR: extract_deltaw.py --build-matrix failed with exit code $STEP2_EXIT"
    exit $STEP2_EXIT
fi
echo "Step 2 complete at $(date)"
echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "Phase 0 Extraction Complete"
echo "========================================"
echo "Results:"
ls -lh "$EXPERIMENT_DIR/results/phase0/"
echo ""
echo "Matrix:"
ls -lh "$EXPERIMENT_DIR/results/all_deltaw_matrix.pt" 2>/dev/null || echo "  (matrix not found)"
echo ""
echo "End time: $(date)"
