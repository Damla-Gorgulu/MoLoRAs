#!/bin/bash
#SBATCH --job-name=lc_phase1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/eyavuz21/mo-lora/experiments/linear_composition/logs/phase1_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eyavuz21@ku.edu.tr

# ============================================================
# Phase 1 — Global Linear Reconstruction (single pass)
# ============================================================
# The script now:
#   - Builds or loads cached full Gram matrix (109×109)
#   - Skips self-check if already passed
#   - Extracts LOO sub-Grams instantly per target (no re-reading 62 GB)
#   - Runs Ridge / Lasso / ElasticNet + normalization ablation
#   - Generates comparison images
#
# Estimated runtime:
#   ~3.5 h (first run — Gram build) + ~30 min (regression + images)
#   ~30 min (subsequent runs — Gram cached)
#
# PREREQUISITE: run_phase0_extract.sh must have completed
# ============================================================

export PYTHONUNBUFFERED=1
export EXPERIMENT_DIR=/scratch/eyavuz21/mo-lora/experiments/linear_composition
export CODE_DIR=/home/eyavuz21/repos/MoLoRAs/experiments/linear_composition

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

cd "$CODE_DIR"

# Check prerequisite
if [ ! -f "$EXPERIMENT_DIR/results/all_deltaw_matrix.pt" ]; then
    echo "ERROR: all_deltaw_matrix.pt not found. Run run_phase0_extract.sh first."
    exit 1
fi

# ============================================================
# Run Phase 1 — single pass (Gram cached, self-check cached)
# ============================================================
echo "========================================"
echo "Running Phase 1: regression + normalization + images"
echo "========================================"
python global_reconstruction.py --normalize --generate-images
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Phase 1 failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
echo "Phase 1 complete at $(date)"
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
