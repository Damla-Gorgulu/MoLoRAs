#!/bin/bash
#
# Stage-2 v2.0: Focused inference sweep
#
# Identical structure to s1v2_sweep.sh — 4 sweeps, ~80 runs.
# Only changes from S1: CKPT and OUT_ROOT.
#
# Key questions answered vs S1:
#   Q1. Is S2 routing sharper (lower entropy) or flatter than S1?
#   Q2. Does the LDM reconstruction signal force coherent routing despite entropy reg?
#   Q3. Is visual style fidelity (under norm_match) better or worse than S1?
#
# ~80 runs × 1 image each ≈ 2-3h on V100.
#
# Usage:
#   cd /home/eyavuz21/repos/MoLoRAs
#   sbatch lora_attention/slurm/s2v2_sweep.sh
#

#SBATCH --job-name=S2v2-Sweep
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
CKPT="/scratch/eyavuz21/lora_attention/stage2_v2/latest.pt"       # ← Stage 2
OUT_ROOT="/scratch/eyavuz21/lora_attention/s2v2_sweep"             # ← S2 output

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"

# ── Fixed settings ───────────────────────────────────────────
SEED=42
NUM_IMAGES=1
STEPS=30
GUIDANCE=7.5

RUN_COUNT=0
FAIL_COUNT=0

# ── Helper ───────────────────────────────────────────────────
# run TAG IMG PROMPT TEMP TOPK ALPHA REF_BLORA GT NORM_MATCH
run() {
    local tag="$1"
    local style_img="$2"
    local prompt="$3"
    local temp="$4"
    local topk="$5"        # "none" or integer
    local alpha="$6"
    local ref_blora="$7"   # "none" or path
    local gt="$8"          # gt_expert name or "none"
    local norm_match="$9"  # "true" or "false"

    local out_dir="$OUT_ROOT/$tag"
    mkdir -p "$out_dir"

    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "── Run $RUN_COUNT: $tag ──"

    local extra_args=()
    [[ "$topk"       != "none"  ]] && extra_args+=(--top_k "$topk")
    [[ "$ref_blora"  != "none"  ]] && extra_args+=(--reference_blora "$ref_blora")
    [[ "$gt"         != "none"  ]] && extra_args+=(--gt_expert "$gt")
    [[ "$norm_match" == "true"  ]] && extra_args+=(--norm_match)

    "$PYTHON" "$SCRIPT" \
        --checkpoint   "$CKPT" \
        --style_image  "$style_img" \
        --prompt       "$prompt" \
        --output_dir   "$out_dir" \
        --zoo_dir      "$ZOO_DIR" \
        --cache_dir    "$CACHE_DIR" \
        --temperature  "$temp" \
        --style_alpha  "$alpha" \
        --num_images   "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed         "$SEED" \
        --query_label  "$tag" \
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

STYLE_PROMPTS[baroque]="A Baroque painting"
STYLE_PROMPTS[cubism]="A Cubism painting"
STYLE_PROMPTS[impressionism]="An Impressionism painting"
STYLE_PROMPTS[expressionism]="An Expressionism painting"

NEUTRAL_PROMPTS[baroque]="A painting of a village scene"
NEUTRAL_PROMPTS[cubism]="A painting of a still life"
NEUTRAL_PROMPTS[impressionism]="A painting of a landscape"
NEUTRAL_PROMPTS[expressionism]="A painting of a figure"

GT_EXPERTS[baroque]="style_0000_Baroque"
GT_EXPERTS[cubism]="style_0003_Cubism"
GT_EXPERTS[impressionism]="style_0005_Impressionism"
GT_EXPERTS[expressionism]="style_0010_Expressionism"

# Correct filename: pytorch_lora_weights.safetensors
REF_BLORAS[baroque]="$ZOO_DIR/style_0000_Baroque/pytorch_lora_weights.safetensors"
REF_BLORAS[cubism]="$ZOO_DIR/style_0003_Cubism/pytorch_lora_weights.safetensors"
REF_BLORAS[impressionism]="$ZOO_DIR/style_0005_Impressionism/pytorch_lora_weights.safetensors"
REF_BLORAS[expressionism]="$ZOO_DIR/style_0010_Expressionism/pytorch_lora_weights.safetensors"

# ════════════════════════════════════════════════════════════════
# SWEEP 1: Synth LoRA — norm_match ON (+style prompt)
# Tests that magnitude-matched synth produces visible style change.
# 4 styles × 2 sources × 3 temps × 2 topk = 48 runs
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 1: Synth + norm_match + style prompt ════════════"

TEMPS=(0.005 0.05 0.5)
TOPKS=(none 1)

for style in "${STYLE_LIST[@]}"; do
    eval prompt="\${STYLE_PROMPTS[$style]}"
    eval gt="\${GT_EXPERTS[$style]}"

    for src in wikiart pool; do
        if [[ "$src" == "wikiart" ]]; then
            eval img="\${WIKIART_IMGS[$style]}"
        else
            eval img="\${POOL_IMGS[$style]}"
        fi

        for temp in "${TEMPS[@]}"; do
            for topk in "${TOPKS[@]}"; do
                tag="${style}/${src}/nm_t${temp}_k${topk}"
                run "$tag" "$img" "$prompt" "$temp" "$topk" "1.0" "none" "$gt" "true"
            done
        done
    done
done

# ════════════════════════════════════════════════════════════════
# SWEEP 2: Synth LoRA — norm_match ON + NEUTRAL prompt
# Removes text-style conditioning so LoRA style direction is clearly visible.
# 4 styles × 2 sources × 2 temps = 16 runs
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 2: Synth + norm_match + NEUTRAL prompt ════════════"

for style in "${STYLE_LIST[@]}"; do
    eval neutral="\${NEUTRAL_PROMPTS[$style]}"
    eval gt="\${GT_EXPERTS[$style]}"

    for src in wikiart pool; do
        if [[ "$src" == "wikiart" ]]; then
            eval img="\${WIKIART_IMGS[$style]}"
        else
            eval img="\${POOL_IMGS[$style]}"
        fi

        for temp in "0.005" "0.5"; do
            tag="${style}/${src}/nm_neutral_t${temp}"
            run "$tag" "$img" "$neutral" "$temp" "none" "1.0" "none" "$gt" "true"
        done
    done
done

# ════════════════════════════════════════════════════════════════
# SWEEP 3: Reference B-LoRA baselines — style prompt AND neutral
# Ground truth: what does perfect single-expert injection look like?
# 4 styles × 2 prompts × 1 source = 8 runs
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 3: Reference B-LoRA baselines ════════════"

for style in "${STYLE_LIST[@]}"; do
    eval style_prompt="\${STYLE_PROMPTS[$style]}"
    eval neutral="\${NEUTRAL_PROMPTS[$style]}"
    eval img="\${WIKIART_IMGS[$style]}"
    eval gt="\${GT_EXPERTS[$style]}"
    eval ref="\${REF_BLORAS[$style]}"

    # Style prompt + reference LoRA
    tag="${style}/wikiart/ref_style_prompt"
    run "$tag" "$img" "$style_prompt" "0.1" "none" "1.0" "$ref" "$gt" "false"

    # Neutral prompt + reference LoRA
    tag="${style}/wikiart/ref_neutral_prompt"
    run "$tag" "$img" "$neutral" "0.1" "none" "1.0" "$ref" "$gt" "false"
done

# ════════════════════════════════════════════════════════════════
# SWEEP 4: Vanilla SDXL baseline (no LoRA)  — both prompts
# 4 styles × 2 prompts = 8 runs
# Use temperature=0.1 but ref_blora="none" + we skip inject by passing alpha=0
# ════════════════════════════════════════════════════════════════
echo ""
echo "════════════ SWEEP 4: Vanilla SDXL (no LoRA, alpha=0) ════════════"

for style in "${STYLE_LIST[@]}"; do
    eval style_prompt="\${STYLE_PROMPTS[$style]}"
    eval neutral="\${NEUTRAL_PROMPTS[$style]}"
    eval img="\${WIKIART_IMGS[$style]}"

    tag="${style}/wikiart/vanilla_style_prompt"
    run "$tag" "$img" "$style_prompt" "0.1" "none" "0.0" "none" "none" "false"

    tag="${style}/wikiart/vanilla_neutral_prompt"
    run "$tag" "$img" "$neutral" "0.1" "none" "0.0" "none" "none" "false"
done

echo ""
echo "========================================"
echo "S2v2 sweep complete."
echo "Total runs: $RUN_COUNT"
echo "Failures:   $FAIL_COUNT"
echo "Output:     $OUT_ROOT"
echo "Finished:   $(date)"
echo "========================================"
