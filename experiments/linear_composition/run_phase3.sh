#!/bin/bash
#SBATCH --job-name=lc_phase3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/eyavuz21/mo-lora/experiments/linear_composition/logs/phase3_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyavuz21@ku.edu.tr

# ============================================================
# Phase 3 — Span Membership Interpretation
# ============================================================
# Tasks: 3.1–3.9 from TODO.md
#
# This script:
#   1. Full leave-one-out for all 109 styles
#   2. Random donor & random tensor baselines
#   3. SVD spectrum analysis
#   4. Sparsity analysis & hub donors
#   5. Span classification
#   6. All plots
#
# Requires: ~64 GB RAM, minimal GPU
# Expected runtime: ~4–6 hours
#
# PREREQUISITE: run_phase0_extract.sh + run_phase1.sh completed
# ============================================================

export PYTHONUNBUFFERED=1
export EXPERIMENT_DIR=/scratch/eyavuz21/mo-lora/experiments/linear_composition
export CODE_DIR=/home/eyavuz21/repos/MoLoRAs/experiments/linear_composition

echo "========================================"
echo "Linear Composition — Phase 3: Span Membership"
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPU(s)     : $CUDA_VISIBLE_DEVICES"
echo "Start time : $(date)"
echo ""

source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate B-LoRA

echo "Python     : $(which python)"
echo "RAM        : $(free -h | head -2)"
echo ""

cd "$CODE_DIR"

# Check prerequisites
if [ ! -f "$EXPERIMENT_DIR/results/all_deltaw_matrix.pt" ]; then
    echo "ERROR: all_deltaw_matrix.pt not found. Run run_phase0_extract.sh first."
    exit 1
fi

# ============================================================
# Run ALL Phase 3 tasks
# ============================================================
echo "========================================"
echo "Running full Phase 3 analysis"
echo "========================================"
python span_analysis.py --all
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: span_analysis.py failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "Phase 3 Complete"
echo "========================================"
echo "Result files:"
ls -lh "$EXPERIMENT_DIR/results/phase3/"
echo ""
echo "Plots:"
ls -lh "$EXPERIMENT_DIR/results/phase3/plots/" 2>/dev/null
echo ""
echo "End time: $(date)"
