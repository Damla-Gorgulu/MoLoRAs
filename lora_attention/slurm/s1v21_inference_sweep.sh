#!/bin/bash
#
# Stage-1 v2.1: Inference Sweep
#
# Tests the v2.1 encoder (CE routing, product-space synthesis) directly at
# Stage 1 — no Stage 2 LDM fine-tuning. This lets us evaluate routing quality
# before committing to the expensive Stage 2 run.
#
# Sweep structure (same as s1v2_ps_sweep.sh):
#   SWEEP 1: product-space synth + style prompt  (4×2×3×2 = 48 runs)
#   SWEEP 2: product-space synth + neutral prompt (4×2×2 = 16 runs)
#   SWEEP 3: reference B-LoRA baselines           (4×2 = 8 runs)
#   SWEEP 4: vanilla SDXL baseline               (4×2 = 8 runs)
#   Total: ~80 runs × ~50s ≈ ~1.1h on V100
#
# Usage:
#   cd /home/eyavuz21/repos/MoLoRAs
#   sbatch lora_attention/slurm/s1v21_inference_sweep.sh
#
# Override checkpoint:
#   CKPT=/scratch/eyavuz21/lora_attention/stage1_v21/checkpoint-13000 sbatch ...
#

#SBATCH --job-name=S1v21-inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=06:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "========================================"

# ── Environment ──────────────────────────────────────────────
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"

# ── Paths ────────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference_v2.py"
WIKIART="/home/eyavuz21/datasets/wikiart"
ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
STYLE_IMG_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
CKPT="${CKPT:-/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt}"
RUN_TAG="${RUN_TAG:-s1v21_inf_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="/scratch/eyavuz21/lora_attention/s1v21_inference_sweep_runs/${RUN_TAG}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"

echo "CKPT:     $CKPT"
echo "RUN_TAG:  $RUN_TAG"
echo "OUT_ROOT: $OUT_ROOT"

# ── Fixed settings ───────────────────────────────────────────
SEED=42
NUM_IMAGES=1
STEPS=30
GUIDANCE=7.5
# α=2.0: sweet spot found via alpha_diag (job 764053).
# synth_norm≈16 vs real B-LoRA norm≈50; α<1.5 → invisible, α>3 → distorted.
ALPHA=2.0

RUN_COUNT=0
FAIL_COUNT=0

# ── Helper ───────────────────────────────────────────────────
# run TAG IMG PROMPT TEMP TOPK ALPHA REF_BLORA GT USE_PS
run() {
    local tag="$1"
    local style_img="$2"
    local prompt="$3"
    local temp="$4"
    local topk="$5"        # "none" or integer
    local alpha="$6"
    local ref_blora="$7"   # "none" or path
    local gt="$8"          # gt_expert name or "none"
    local use_ps="$9"      # "true" or "false"

    local out_dir="$OUT_ROOT/$tag"
    mkdir -p "$out_dir"

    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "── Run $RUN_COUNT: $tag ──"

    local extra_args=()
    [[ "$topk"      != "none"  ]] && extra_args+=(--top_k "$topk")
    [[ "$ref_blora" != "none"  ]] && extra_args+=(--reference_blora "$ref_blora")
    [[ "$gt"        != "none"  ]] && extra_args+=(--gt_expert "$gt")
    [[ "$use_ps"    == "true"  ]] && extra_args+=(--product_synth)

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

# ════════════════════════════════════════════════════════════════
# STYLE DEFINITIONS
# ════════════════════════════════════════════════════════════════

STYLE_LIST=(baroque cubism impressionism expressionism)

declare -A WIKIART_IMGS POOL_IMGS STYLE_PROMPTS NEUTRAL_PROMPTS GT_EXPERTS REF_BLORAS

WIKIART_IMGS[baroque]="$WIKIART/Baroque/adriaen-brouwer_a-boor-asleep.jpg"
WIKIART_IMGS[cubism]="$WIKIART/Cubism/adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg"
WIKIART_IMGS[impressionism]="$WIKIART/Impressionism/abdullah-suriosubroto_air-terjun.jpg"
WIKIART_IMGS[expressionism]="$WIKIART/Expressionism/abidin-dino_drawing-pain-1968.jpg"

POOL_IMGS[baroque]="$STYLE_IMG_DIR/style_0000_Baroque/style_0000_Baroque.jpg"
POOL_IMGS[cubism]="$STYLE_IMG_DIR/style_0003_Cubism/style_0003_Cubism.jpg"
POOL_IMGS[impressionism]="$STYLE_IMG_DIR/style_0005_Impressionism/style_0005_Impressionism.jpg"
POOL_IMGS[expressionism]="$STYLE_IMG_DIR/style_0010_Expressionism/style_0010_Expressionism.jpg"

STYLE_PROMPTS[baroque]="A dog in Baroque style"
STYLE_PROMPTS[cubism]="A dog in Cubism style"
STYLE_PROMPTS[impressionism]="A dog in Impressionism style"
STYLE_PROMPTS[expressionism]="A dog in Expressionism style"

NEUTRAL_PROMPTS[baroque]="A painting of a village scene"
NEUTRAL_PROMPTS[cubism]="A painting of a still life"
NEUTRAL_PROMPTS[impressionism]="A painting of a landscape"
NEUTRAL_PROMPTS[expressionism]="A painting of a figure"

GT_EXPERTS[baroque]="style_0000_Baroque"
GT_EXPERTS[cubism]="style_0003_Cubism"
GT_EXPERTS[impressionism]="style_0005_Impressionism"
GT_EXPERTS[expressionism]="style_0010_Expressionism"

REF_BLORAS[baroque]="$ZOO_DIR/style_0000_Baroque/pytorch_lora_weights.safetensors"
REF_BLORAS[cubism]="$ZOO_DIR/style_0003_Cubism/pytorch_lora_weights.safetensors"
REF_BLORAS[impressionism]="$ZOO_DIR/style_0005_Impressionism/pytorch_lora_weights.safetensors"
REF_BLORAS[expressionism]="$ZOO_DIR/style_0010_Expressionism/pytorch_lora_weights.safetensors"

# ════════════════════════════════════════════════════════════════
# SWEEP 1: Product-space synth + style prompt (expanded hypergrid)
# τ × top_k × alpha search for stronger coverage.
# 4 styles × 1 source(wikiart) × 5 temps × 3 topk × 3 alpha = 180 runs
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 1: Expanded hypergrid (style prompt) ════════════"

TEMPS=(0.001 0.005 0.02 0.05 0.1)
TOPKS=(none 1 3)
ALPHAS=(1.0 2.0 3.0)

for style in "${STYLE_LIST[@]}"; do
    eval prompt="\${STYLE_PROMPTS[$style]}"
    eval gt="\${GT_EXPERTS[$style]}"

    for src in wikiart; do
        if [[ "$src" == "wikiart" ]]; then
            eval img="\${WIKIART_IMGS[$style]}"
        else
            eval img="\${POOL_IMGS[$style]}"
        fi

        for temp in "${TEMPS[@]}"; do
            for topk in "${TOPKS[@]}"; do
                for alpha in "${ALPHAS[@]}"; do
                    tag="${style}/${src}/ps_t${temp}_k${topk}_a${alpha}"
                    run "$tag" "$img" "$prompt" "$temp" "$topk" "$alpha" "none" "$gt" "true"
                done
            done
        done
    done
done

# ════════════════════════════════════════════════════════════════
# SWEEP 2: Product-space synth + NEUTRAL prompt (expanded)
# 4 styles × 1 source(wikiart) × 2 temps × 2 alpha = 16 runs
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 2: Product-space synth + neutral prompt ════════════"

NEUTRAL_TEMPS=(0.005 0.1)
NEUTRAL_ALPHAS=(1.0 2.0)

for style in "${STYLE_LIST[@]}"; do
    eval neutral="\${NEUTRAL_PROMPTS[$style]}"
    eval gt="\${GT_EXPERTS[$style]}"

    for src in wikiart; do
        if [[ "$src" == "wikiart" ]]; then
            eval img="\${WIKIART_IMGS[$style]}"
        else
            eval img="\${POOL_IMGS[$style]}"
        fi

        for temp in "${NEUTRAL_TEMPS[@]}"; do
            for alpha in "${NEUTRAL_ALPHAS[@]}"; do
                tag="${style}/${src}/ps_neutral_t${temp}_a${alpha}"
                run "$tag" "$img" "$neutral" "$temp" "none" "$alpha" "none" "$gt" "true"
            done
        done
    done
done

# ════════════════════════════════════════════════════════════════
# SWEEP 3: Reference B-LoRA baselines
# Ground truth: what does a real single-expert injection look like?
# 4 styles × 2 prompts = 8 runs
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 3: Reference B-LoRA baselines ════════════"

for style in "${STYLE_LIST[@]}"; do
    eval style_prompt="\${STYLE_PROMPTS[$style]}"
    eval neutral="\${NEUTRAL_PROMPTS[$style]}"
    eval img="\${WIKIART_IMGS[$style]}"
    eval gt="\${GT_EXPERTS[$style]}"
    eval ref="\${REF_BLORAS[$style]}"

    tag="${style}/wikiart/ref_style_prompt"
    run "$tag" "$img" "$style_prompt" "0.1" "none" "1.0" "$ref" "$gt" "false"

    tag="${style}/wikiart/ref_neutral_prompt"
    run "$tag" "$img" "$neutral" "0.1" "none" "1.0" "$ref" "$gt" "false"
done

# ════════════════════════════════════════════════════════════════
# SWEEP 4: Vanilla SDXL (no LoRA, alpha=0) — both prompts
# 4 styles × 2 prompts = 8 runs
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 4: Vanilla SDXL baseline ════════════"

for style in "${STYLE_LIST[@]}"; do
    eval style_prompt="\${STYLE_PROMPTS[$style]}"
    eval neutral="\${NEUTRAL_PROMPTS[$style]}"
    eval img="\${WIKIART_IMGS[$style]}"

    tag="${style}/wikiart/vanilla_style"
    run "$tag" "$img" "$style_prompt" "0.1" "none" "0.0" "none" "none" "false"

    tag="${style}/wikiart/vanilla_neutral"
    run "$tag" "$img" "$neutral" "0.1" "none" "0.0" "none" "none" "false"
done

echo ""
echo "========================================"
echo "S1v21 inference sweep complete."
echo "Total runs: $RUN_COUNT"
echo "Failures:   $FAIL_COUNT"
echo "Output:     $OUT_ROOT"
echo "Finished:   $(date)"
echo "========================================"
