#!/bin/bash
#
# Post-train validation sweep for Stage 2 v2.3.
#
# This is intentionally smaller than the broad sweeps:
#   - 2 styles
#   - style prompt and neutral prompt
#   - soft routing vs top-1 routing
#   - reference B-LoRA and vanilla SDXL baselines
#
# Use this as the chained follow-up to a training job so every checkpoint gets a
# quick sanity pass before we invest in a larger sweep.
#
# Usage:
#   sbatch lora_attention/slurm/validate_stage2_v23.sh
#

#SBATCH --job-name=MoELoRA-S2v23-Val
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=04:00:00
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
STYLE_IMG_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
CKPT="${STAGE2_CKPT:-/scratch/eyavuz21/lora_attention/stage2_v23/latest.pt}"
OUT_ROOT="${OUT_ROOT:-/scratch/eyavuz21/lora_attention/stage2_v23_validation}"
RUN_TAG="${RUN_TAG:-s2v23_validate_$(date +%Y%m%d_%H%M%S)}"
KB_PATH="${KB_PATH:-/home/eyavuz21/repos/MoLoRAs/lora_attention/experiment_kb.md}"
JSONL_PATH="${JSONL_PATH:-/home/eyavuz21/repos/MoLoRAs/lora_attention/experiment_kb.jsonl}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT"

echo "CKPT:     $CKPT"
echo "RUN_TAG:  $RUN_TAG"
echo "OUT_ROOT: $OUT_ROOT"
echo "KB_PATH:   $KB_PATH"

SEED=42
NUM_IMAGES=1
STEPS=30
GUIDANCE=7.5

STYLE_LIST=(baroque impressionism)

declare -A WIKIART_IMGS POOL_IMGS STYLE_PROMPTS NEUTRAL_PROMPTS GT_EXPERTS REF_BLORAS

WIKIART_IMGS[baroque]="/home/eyavuz21/datasets/wikiart/Baroque/adriaen-brouwer_a-boor-asleep.jpg"
WIKIART_IMGS[cubism]="/home/eyavuz21/datasets/wikiart/Cubism/adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg"
WIKIART_IMGS[impressionism]="/home/eyavuz21/datasets/wikiart/Impressionism/abdullah-suriosubroto_air-terjun.jpg"
WIKIART_IMGS[expressionism]="/home/eyavuz21/datasets/wikiart/Expressionism/abidin-dino_drawing-pain-1968.jpg"

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

RUN_COUNT=0
FAIL_COUNT=0

run() {
    local tag="$1"
    local style_img="$2"
    local prompt="$3"
    local temp="$4"
    local topk="$5"
    local alpha="$6"
    local ref_blora="$7"
    local gt="$8"
    local use_product_synth="$9"

    local out_dir="$OUT_ROOT/$tag"
    mkdir -p "$out_dir"

    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "── Run $RUN_COUNT: $tag ──"

    local extra_args=()
    [[ "$topk" != "none" ]] && extra_args+=(--top_k "$topk")
    [[ "$ref_blora" != "none" ]] && extra_args+=(--reference_blora "$ref_blora")
    [[ "$gt" != "none" ]] && extra_args+=(--gt_expert "$gt")
    [[ "$use_product_synth" == "true" ]] && extra_args+=(--product_synth)

    "$PYTHON" "$SCRIPT" \
        --checkpoint "$CKPT" \
        --style_image "$style_img" \
        --prompt "$prompt" \
        --output_dir "$out_dir" \
        --zoo_dir "$ZOO_DIR" \
        --cache_dir "$CACHE_DIR" \
        --temperature "$temp" \
        --style_alpha "$alpha" \
        --num_images "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --run_tag "$RUN_TAG" \
        --query_label "$tag" \
        "${extra_args[@]}" \
    || { echo "  !! FAILED: $tag"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
}

echo ""
echo "════════════ Validation: Stage 2 v2.3 checkpoint ════════════"

for style in "${STYLE_LIST[@]}"; do
    eval style_prompt="\${STYLE_PROMPTS[$style]}"
    eval neutral_prompt="\${NEUTRAL_PROMPTS[$style]}"
    eval wikiart_img="\${WIKIART_IMGS[$style]}"
    eval pool_img="\${POOL_IMGS[$style]}"
    eval gt="\${GT_EXPERTS[$style]}"
    eval ref="\${REF_BLORAS[$style]}"

    run "${style}/wikiart/style_tau0.1" "$wikiart_img" "$style_prompt" "0.1" "none" "1.0" "none" "$gt" "true"
    run "${style}/wikiart/style_top1" "$wikiart_img" "$style_prompt" "0.1" "1" "1.0" "none" "$gt" "true"
    run "${style}/wikiart/neutral_tau0.1" "$wikiart_img" "$neutral_prompt" "0.1" "none" "1.0" "none" "$gt" "true"
    run "${style}/wikiart/neutral_top1" "$wikiart_img" "$neutral_prompt" "0.1" "1" "1.0" "none" "$gt" "true"
    run "${style}/wikiart/reference_style" "$wikiart_img" "$style_prompt" "0.1" "none" "1.0" "$ref" "$gt" "false"
    run "${style}/wikiart/vanilla_neutral" "$wikiart_img" "$neutral_prompt" "0.1" "none" "0.0" "none" "none" "false"
done

SUMMARY_JSON="$OUT_ROOT/$RUN_TAG/summary.json"
mkdir -p "$(dirname "$SUMMARY_JSON")"
cat > "$SUMMARY_JSON" <<EOF
{"job_id":"${SLURM_JOB_ID:-local}","run_tag":"$RUN_TAG","checkpoint":"$CKPT","output_dir":"$OUT_ROOT","runs":$RUN_COUNT,"failures":$FAIL_COUNT,"synthesis":"product_space","status":"completed"}
EOF

"$PYTHON" "$REPO_ROOT/lora_attention/log_experiment.py" \
    --kb_path "$KB_PATH" \
    --jsonl_path "$JSONL_PATH" \
    --stage "validation" \
    --job_id "${SLURM_JOB_ID:-local}" \
    --run_tag "$RUN_TAG" \
    --checkpoint "$CKPT" \
    --synthesis "product_space" \
    --prompt_type "mixed" \
    --result "runs=$RUN_COUNT failures=$FAIL_COUNT output_dir=$OUT_ROOT" \
    --verdict "$( [[ "$FAIL_COUNT" -eq 0 ]] && echo partial || echo bad )" \
    --notes "Auto-recorded from stage2 v2.3 validation."

echo ""
echo "========================================"
echo "Validation complete."
echo "Total runs: $RUN_COUNT"
echo "Failures:   $FAIL_COUNT"
echo "Output:     $OUT_ROOT"
echo "Summary:    $SUMMARY_JSON"
echo "Finished:   $(date)"
echo "========================================"
