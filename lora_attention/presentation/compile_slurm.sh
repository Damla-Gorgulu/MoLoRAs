#!/bin/bash
#SBATCH --job-name=compile-slides
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --qos=ai
#SBATCH --time=0:30:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/presentation/compile_slurm.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/presentation/compile_slurm.err

set -e

# ── Conda setup ────────────────────────────────────────────────────────────────
source /opt/ohpc/pub/compiler/conda3/latest/etc/profile.d/conda.sh
conda activate B-LoRA_2

CONDA_BIN="/home/eyavuz21/.conda/envs/B-LoRA_2/bin"

# ── Install tectonic (self-contained LaTeX, downloads packages on demand) ─────
if ! "${CONDA_BIN}/tectonic" --version &>/dev/null 2>&1; then
    echo "[compile] tectonic not found — installing from conda-forge..."
    conda install -y -c conda-forge tectonic
    echo "[compile] tectonic installed."
fi

export PATH="${CONDA_BIN}:${PATH}"

# ── Compile ────────────────────────────────────────────────────────────────────
cd /home/eyavuz21/repos/MoLoRAs/lora_attention/presentation

echo "[compile] Compiling with tectonic (downloads missing packages automatically)..."
tectonic -X compile slides.tex 2>&1 | tee compile.log

echo "[compile] Done."
ls -lh slides.pdf 2>/dev/null || { echo "ERROR: slides.pdf NOT created — see compile.log"; exit 1; }
