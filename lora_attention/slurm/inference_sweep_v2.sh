#!/bin/bash
#
# v2.3 Inference Sweep: Test Stage 2 v2.3 checkpoint
# Comprehensive parameter sweep with temperature, top-k, alpha combinations
#
# Usage:
#   sbatch slurm/inference_sweep_v2.sh
#

#SBATCH --job-name=MoELoRA-S2v23-Sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=06:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -eo pipefail

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================"

# ── Modules ────────────────────────────────────────────────
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
export PYTHONUNBUFFERED=1

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
STYLE_IMAGES_ROOT="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
BLORAS_ROOT="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
OUT_ROOT="/scratch/eyavuz21/lora_attention/s2v23_sweep"

# ── Checkpoints to test ────────────────────────────────────
declare -a CHECKPOINTS=(
    "/scratch/eyavuz21/lora_attention/stage2_v23/latest.pt"
    "/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt"
)

# ── Test styles (5 distinct styles) ─────────────────────────
declare -a TEST_CASES=(
    "style_0000_Baroque|A baroque landscape [v]"
    "style_0005_Impressionism|An impressionist garden [v]"
    "style_0010_Expressionism|An expressionist forest [v]"
    "style_0007_Romanticism|A romantic mountain scene [v]"
    "style_0008_Minimalism|A minimalist cityscape [v]"
)

# ── Sweep configurations ────────────────────────────────────
# Format: "label|temperature|top_k|style_alpha"
declare -a CONFIGS=(
    # Standard temperatures
    "tau1.0|1.0|0|1.0"
    "tau0.5|0.5|0|1.0"
    "tau0.1|0.1|0|1.0"
    "tau0.01|0.01|0|1.0"
    # Top-k with sharp temperature
    "tau0.1_top1|0.1|1|1.0"
    "tau0.1_top3|0.1|3|1.0"
    "tau0.1_top5|0.1|5|1.0"
    # Higher alpha
    "tau0.1_alpha1.5|0.1|0|1.5"
    "tau0.1_alpha2.0|0.1|0|2.0"
    # Soft routing
    "tau2.0|2.0|0|1.0"
)

mkdir -p "$OUT_ROOT"

echo ""
echo "=== Sweep Configurations: ${#CONFIGS[@]} ==="
echo "=== Test Styles: ${#TEST_CASES[@]} ==="
echo "=== Checkpoints: ${#CHECKPOINTS[@]} ==="
echo ""

# ── Run sweep for each checkpoint ───────────────────────────
for CKPT in "${CHECKPOINTS[@]}"; do
    CKPT_NAME=$(basename $(dirname "$CKPT"))
    echo ""
    echo "========================================"
    echo "Checkpoint: $CKPT_NAME"
    echo "========================================"

    for cfg in "${CONFIGS[@]}"; do
        IFS='|' read -r label temp topk alpha <<< "$cfg"

        for entry in "${TEST_CASES[@]}"; do
            style_dir="${entry%%|*}"
            prompt="${entry##*|}"
            style_image=$(find "$STYLE_IMAGES_ROOT/$style_dir" -maxdepth 1 -name "*.jpg" | head -1)

            if [[ -z "$style_image" ]]; then
                echo "[WARN] No jpg for $style_dir — skipping"
                continue
            fi

            out_dir="$OUT_ROOT/${CKPT_NAME}/${label}/${style_dir}"

            echo "────────────────────────────────────"
            echo "Config: $label | Style: $style_dir"
            echo "  temp=$temp, top_k=$topk, alpha=$alpha"

            topk_arg=""
            if [[ "$topk" != "0" ]]; then
                topk_arg="--top_k $topk"
            fi

            $PYTHON "$SCRIPT" \
                --checkpoint   "$CKPT" \
                --style_image  "$style_image" \
                --prompt       "$prompt" \
                --output_dir   "$out_dir" \
                --temperature  "$temp" \
                $topk_arg \
                --style_alpha  "$alpha" \
                --product_synth \
                --num_images   4 \
                --zoo_dir      "$ZOO_DIR" \
                --cache_dir    "/scratch/eyavuz21/lora_attention"

            echo "  → $out_dir"
        done
    done
done

# ── Reference: real B-LoRA injection ────────────────────────
echo ""
echo "=== Reference B-LoRA (direct injection) ==="
for entry in "${TEST_CASES[@]}"; do
    style_dir="${entry%%|*}"
    prompt="${entry##*|}"
    style_image=$(find "$STYLE_IMAGES_ROOT/$style_dir" -maxdepth 1 -name "*.jpg" | head -1)
    ref_lora="$BLORAS_ROOT/$style_dir/pytorch_lora_weights.safetensors"

    if [[ ! -f "$ref_lora" ]]; then
        echo "[WARN] No reference LoRA for $style_dir — skipping"
        continue
    fi

    out_dir="$OUT_ROOT/reference_blora/${style_dir}"
    echo "────────────────────────────────────"
    echo "Reference: $style_dir"

    $PYTHON "$SCRIPT" \
        --checkpoint      "/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt" \
        --style_image     "$style_image" \
        --prompt          "$prompt" \
        --output_dir      "$out_dir" \
        --reference_blora "$ref_lora" \
        --style_alpha     1.0 \
        --num_images      4 \
        --zoo_dir         "$ZOO_DIR" \
        --cache_dir       "/scratch/eyavuz21/lora_attention"

    echo "  → $out_dir"
done

# ── Baseline: Vanilla SDXL (no LoRA) ────────────────────────
echo ""
echo "=== Baseline: Vanilla SDXL (no LoRA) ==="
for entry in "${TEST_CASES[@]}"; do
    style_dir="${entry%%|*}"
    prompt="${entry##*|}"
    style_image=$(find "$STYLE_IMAGES_ROOT/$style_dir" -maxdepth 1 -name "*.jpg" | head -1)

    out_dir="$OUT_ROOT/vanilla_sdxl/${style_dir}"
    echo "────────────────────────────────────"
    echo "Vanilla: $style_dir"

    $PYTHON "$SCRIPT" \
        --checkpoint    "/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt" \
        --style_image   "$style_image" \
        --prompt        "$prompt" \
        --output_dir    "$out_dir" \
        --style_alpha   0.0 \
        --num_images    4 \
        --zoo_dir       "$ZOO_DIR" \
        --cache_dir     "/scratch/eyavuz21/lora_attention"

    echo "  → $out_dir"
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "Output root: $OUT_ROOT"
echo "Finished:    $(date)"
echo ""
echo "Comparison layout:"
echo "  $OUT_ROOT/"
echo "    stage2_v23/tau0.1/style_XXXX/     ← Stage 2 v2.3"
echo "    stage1_v21/tau0.1/style_XXXX/     ← Stage 1 (comparison)"
echo "    reference_blora/style_XXXX/        ← real B-LoRA"
echo "    vanilla_sdxl/style_XXXX/           ← no LoRA baseline"
echo "========================================"
