#!/bin/bash
#
# MoELoRA Inference
# Runs inference with Stage 1 and Stage 2 checkpoints on a set of test styles.
# Generates 4 images per (style, checkpoint) combination.
#
# Usage:
#   sbatch slurm/inference.sh
#
# Override examples:
#   CKPT_S1=/scratch/.../stage1/checkpoint-5000/checkpoint.pt sbatch slurm/inference.sh
#   STYLES="style_0000_Baroque style_0003_Cubism" sbatch slurm/inference.sh
#

#SBATCH --job-name=MoELoRA-Infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=04:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================"

# ── Modules ────────────────────────────────────────────────
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

# ── Conda environment ───────────────────────────────────────
source activate B-LoRA_2 || conda activate B-LoRA_2

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference.py"
STYLE_IMAGES_ROOT="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
LOG_DIR="$REPO_ROOT/lora_attention/logs"
mkdir -p "$LOG_DIR"

CKPT_S1="${CKPT_S1:-/scratch/eyavuz21/lora_attention/stage1/latest.pt}"
CKPT_S2="${CKPT_S2:-/scratch/eyavuz21/lora_attention/stage2/latest.pt}"
OUT_S1="${OUT_S1:-/scratch/eyavuz21/lora_attention/inference_s1}"
OUT_S2="${OUT_S2:-/scratch/eyavuz21/lora_attention/inference_s2}"

# Styles to test (name → prompt)
# Format: "style_dir_name|prompt"
declare -a TEST_CASES=(
    "style_0000_Baroque|A cat in baroque style [v]"
    "style_0002_Abstract_Expressionism|A landscape in abstract expressionism style [v]"
    "style_0003_Cubism|A portrait in cubism style [v]"
    "style_0010_Expressionism|A forest in expressionism style [v]"
    "style_0020_Minimalism|A room in minimalism style [v]"
)

mkdir -p "$OUT_S1" "$OUT_S2"

echo "========================================"
echo "Stage 1 checkpoint: $CKPT_S1"
echo "Stage 2 checkpoint: $CKPT_S2"
echo "Test cases: ${#TEST_CASES[@]}"
echo "========================================"

run_inference() {
    local ckpt="$1"
    local out_dir="$2"
    local stage_label="$3"

    for entry in "${TEST_CASES[@]}"; do
        local style_dir="${entry%%|*}"
        local prompt="${entry##*|}"
        # Find first jpg in the style directory (filenames are not always $style_dir.jpg)
        local style_image
        style_image=$(find "$STYLE_IMAGES_ROOT/$style_dir" -maxdepth 1 -name "*.jpg" | head -1)

        if [[ -z "$style_image" ]]; then
            echo "[WARN] No jpg found in $STYLE_IMAGES_ROOT/$style_dir — skipping"
            continue
        fi

        echo ""
        echo "[$stage_label] Style: $style_dir"
        echo "[$stage_label] Prompt: $prompt"

        python "$SCRIPT" \
            --checkpoint   "$ckpt" \
            --style_image  "$style_image" \
            --prompt       "$prompt" \
            --output_dir   "$out_dir/$style_dir" \
            --num_images   4 \
            --num_inference_steps 30 \
            --guidance_scale 7.5 \
            --seed 42
    done
}

echo ""
echo "=== Running Stage 1 inference ==="
run_inference "$CKPT_S1" "$OUT_S1" "S1"

echo ""
echo "=== Running Stage 2 inference ==="
run_inference "$CKPT_S2" "$OUT_S2" "S2"

echo ""
echo "========================================"
echo "Outputs:"
echo "  Stage 1: $OUT_S1"
echo "  Stage 2: $OUT_S2"
echo "Finished:  $(date)"
echo "========================================"
