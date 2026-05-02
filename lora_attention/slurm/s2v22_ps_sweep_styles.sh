#!/bin/bash
#
# Stage-2 v2.2: Product-space sweep on custom style images under B-LoRA/Styles
#
# Uses the same focused hyperparameter region as current Stage-2 sweep:
#   temp < 0.005, alpha in (1.5, 2.5)
# Prompts are built as requested:
#   "dog in x {sub_folder_name} style"
#
# Usage:
#   sbatch lora_attention/slurm/s2v22_ps_sweep_styles.sh
#

#SBATCH --job-name=S2v22-Styles
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=08:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "========================================"

module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"

REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference_v2.py"
ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
STYLE_ROOT="/home/eyavuz21/repos/B-LoRA/Styles"
CKPT="${STAGE2_CKPT:-/scratch/eyavuz21/lora_attention/stage2_v22/latest.pt}"
RUN_TAG="${RUN_TAG:-s2v22_styles_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="/scratch/eyavuz21/lora_attention/s2v22_styles_sweep_runs/${RUN_TAG}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

echo "CKPT:     $CKPT"
echo "RUN_TAG:  $RUN_TAG"
echo "OUT_ROOT: $OUT_ROOT"

SEED=42
NUM_IMAGES=1
STEPS=30
GUIDANCE=7.5

# Focused region requested by user
TEMPS=(0.001 0.0025 0.004)
TOPKS=(none 1)
ALPHAS=(1.6 1.8 2.0 2.2 2.4)

RUN_COUNT=0
FAIL_COUNT=0

run() {
    local tag="$1"
    local style_img="$2"
    local prompt="$3"
    local temp="$4"
    local topk="$5"
    local alpha="$6"

    local out_dir="$OUT_ROOT/$tag"
    mkdir -p "$out_dir"

    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "── Run $RUN_COUNT: $tag ──"

    local extra_args=(--product_synth)
    [[ "$topk" != "none" ]] && extra_args+=(--top_k "$topk")

    "$PYTHON" "$SCRIPT" \
        --checkpoint          "$CKPT" \
        --style_image         "$style_img" \
        --prompt              "$prompt" \
        --output_dir          "$out_dir" \
        --zoo_dir             "$ZOO_DIR" \
        --cache_dir           "$CACHE_DIR" \
        --temperature         "$temp" \
        --style_alpha         "$alpha" \
        --num_images          "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale      "$GUIDANCE" \
        --seed                "$SEED" \
        --run_tag             "$RUN_TAG" \
        --query_label         "$tag" \
        "${extra_args[@]}" \
    || { echo "  !! FAILED: $tag"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
}

echo ""
echo "════════════ SWEEP: Styles folder images ════════════"

# Iterate each subfolder and pick the first image as representative
while IFS= read -r -d '' style_dir; do
    style_name="$(basename "$style_dir")"

    img_path="$(find "$style_dir" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | sort | head -n 1)"
    if [[ -z "$img_path" ]]; then
        echo "[warn] no image found in: $style_dir"
        continue
    fi

    # Prompt template requested by user.
    prompt="dog in x ${style_name} style"

    for temp in "${TEMPS[@]}"; do
        for topk in "${TOPKS[@]}"; do
            for alpha in "${ALPHAS[@]}"; do
                tag="${style_name}/ps_t${temp}_k${topk}_a${alpha}"
                run "$tag" "$img_path" "$prompt" "$temp" "$topk" "$alpha"
            done
        done
    done
done < <(find "$STYLE_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo ""
echo "========================================"
echo "S2v22 Styles sweep complete."
echo "Total runs: $RUN_COUNT"
echo "Failures:   $FAIL_COUNT"
echo "Output:     $OUT_ROOT"
echo "Finished:   $(date)"
echo "========================================"
