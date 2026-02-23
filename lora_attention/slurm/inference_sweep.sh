#!/bin/bash
#
# MoELoRA Inference Sweep
# Tests different temperature and top-k settings to find optimal style transfer.
# Also runs reference B-LoRA for comparison.
#
# Usage:
#   sbatch slurm/inference_sweep.sh
#

#SBATCH --job-name=MoELoRA-Sweep
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

# ── Conda environment (use explicit path — activate unreliable in SLURM) ──
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"
echo "Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference.py"
STYLE_IMAGES_ROOT="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
BLORAS_ROOT="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CKPT="/scratch/eyavuz21/lora_attention/stage1/latest.pt"
OUT_ROOT="/scratch/eyavuz21/lora_attention/inference_sweep"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

# ── Test cases (3 distinct styles for speed) ────────────────
declare -a TEST_CASES=(
    "style_0000_Baroque|A cat in baroque style [v]"
    "style_0003_Cubism|A portrait in cubism style [v]"
    "style_0010_Expressionism|A forest in expressionism style [v]"
)

# ── Sweep configurations ────────────────────────────────────
#  Format: "label|temperature|top_k|style_alpha"
#  top_k=0 means no top-k (use all experts)
declare -a CONFIGS=(
    # Baseline: standard softmax
    "tau1.0_noTopK|1.0|0|1.0"
    # Temperature sweep
    "tau0.5_noTopK|0.5|0|1.0"
    "tau0.1_noTopK|0.1|0|1.0"
    "tau0.01_noTopK|0.01|0|1.0"
    # Top-k sweep (with sharp temperature)
    "tau0.1_top1|0.1|1|1.0"
    "tau0.1_top3|0.1|3|1.0"
    "tau0.1_top5|0.1|5|1.0"
    # Higher alpha to compensate for dilution
    "tau1.0_alpha2.0|1.0|0|2.0"
    "tau0.1_alpha1.5|0.1|0|1.5"
)

mkdir -p "$OUT_ROOT"

echo ""
echo "=== Sweep Configurations: ${#CONFIGS[@]} ==="
echo "=== Test Styles: ${#TEST_CASES[@]} ==="
echo "=== S1 Checkpoint: $CKPT ==="
echo ""

# ── Run sweep ────────────────────────────────────────────────
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

        out_dir="$OUT_ROOT/${label}/${style_dir}"

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
            --num_images   4 \
            --num_inference_steps 30 \
            --guidance_scale 7.5 \
            --seed 42

        echo "  → $out_dir"
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
        echo "[WARN] No reference LoRA for $style_dir at $ref_lora — skipping"
        continue
    fi

    out_dir="$OUT_ROOT/reference_blora/${style_dir}"
    echo "────────────────────────────────────"
    echo "Reference: $style_dir"

    $PYTHON "$SCRIPT" \
        --checkpoint      "$CKPT" \
        --style_image     "$style_image" \
        --prompt          "$prompt" \
        --output_dir      "$out_dir" \
        --reference_blora "$ref_lora" \
        --style_alpha     1.0 \
        --num_images      4 \
        --num_inference_steps 30 \
        --guidance_scale  7.5 \
        --seed 42

    echo "  → $out_dir"
done

# ── Baseline: NO LoRA at all (vanilla SDXL) ────────────────
echo ""
echo "=== Baseline: Vanilla SDXL (no LoRA) ==="
for entry in "${TEST_CASES[@]}"; do
    style_dir="${entry%%|*}"
    prompt="${entry##*|}"
    style_image=$(find "$STYLE_IMAGES_ROOT/$style_dir" -maxdepth 1 -name "*.jpg" | head -1)

    out_dir="$OUT_ROOT/vanilla_sdxl/${style_dir}"
    echo "────────────────────────────────────"
    echo "Vanilla: $style_dir"

    # Use temperature=1, alpha=0 to effectively inject nothing
    $PYTHON "$SCRIPT" \
        --checkpoint    "$CKPT" \
        --style_image   "$style_image" \
        --prompt        "$prompt" \
        --output_dir    "$out_dir" \
        --style_alpha   0.0 \
        --num_images    4 \
        --num_inference_steps 30 \
        --guidance_scale 7.5 \
        --seed 42

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
echo "    tau1.0_noTopK/style_XXXX/  ← original (no temp/topk)"
echo "    tau0.1_noTopK/style_XXXX/  ← sharp temperature"
echo "    tau0.1_top3/style_XXXX/    ← sharp + top-3"
echo "    reference_blora/style_XXXX/ ← real B-LoRA"
echo "    vanilla_sdxl/style_XXXX/    ← no LoRA baseline"
echo "========================================"
