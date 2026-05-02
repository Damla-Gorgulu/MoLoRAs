#!/bin/bash
#
# Retrospective prompt-sensitivity sweep for older v2 checkpoints.
#
# Goal:
#   Compare the original style-prompt outputs against neutral-prompt outputs
#   using the same trained weights, same style images, and the same routing
#   settings. This isolates whether the earlier "no visible style" reports
#   were partly a prompt artifact rather than a routing failure.
#
# We keep this much smaller than the original broad sweeps:
#   - 2 checkpoints: stage2_v23 and stage1_v21
#   - 5 styles
#   - 2 prompt kinds: style vs neutral
#   - 2 routing settings: soft vs top1
#
# Output layout:
#   /scratch/eyavuz21/lora_attention/prompt_retro_v1/<RUN_TAG>/<ckpt>/<prompt>/<style>/<config>/
#
# Usage:
#   cd /home/eyavuz21/repos/MoLoRAs
#   sbatch lora_attention/slurm/retro_prompt_sweep_v1.sh

#SBATCH --job-name=MoELoRA-RetroPrompt
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
RUN_TAG="${RUN_TAG:-retro_prompt_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="/scratch/eyavuz21/lora_attention/prompt_retro_v1/${RUN_TAG}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"
echo "RUN_TAG:  $RUN_TAG"
echo "OUT_ROOT: $OUT_ROOT"

# Fixed inference settings
SEED=42
NUM_IMAGES=1
STEPS=30
GUIDANCE=7.5

RUN_COUNT=0
FAIL_COUNT=0

run() {
    local ckpt_name="$1"
    local ckpt="$2"
    local prompt_kind="$3"
    local style="$4"
    local style_img="$5"
    local prompt="$6"
    local temp="$7"
    local topk="$8"     # "none" or integer
    local alpha="$9"

    local out_dir="$OUT_ROOT/$ckpt_name/$prompt_kind/$style/t${temp}"
    if [[ "$topk" != "none" ]]; then
        out_dir="$out_dir/topk_${topk}"
    else
        out_dir="$out_dir/topk_none"
    fi
    out_dir="$out_dir/alpha_${alpha}"
    mkdir -p "$out_dir"

    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "── Run $RUN_COUNT: $ckpt_name | $prompt_kind | $style | t=$temp | topk=$topk | alpha=$alpha ──"

    local extra_args=()
    [[ "$topk" != "none" ]] && extra_args+=(--top_k "$topk")

    "$PYTHON" "$SCRIPT" \
        --checkpoint "$ckpt" \
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
        --query_label "$ckpt_name/$prompt_kind/$style/t${temp}_k${topk}_a${alpha}" \
        --product_synth \
        "${extra_args[@]}" \
    || { echo "  !! FAILED: $ckpt_name | $prompt_kind | $style | t=$temp | topk=$topk | alpha=$alpha"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
}

declare -a CHECKPOINTS=(
    "stage2_v23|/scratch/eyavuz21/lora_attention/stage2_v23/latest.pt"
    "stage1_v21|/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt"
)

declare -a STYLE_CASES=(
    "baroque|style_0000_Baroque"
    "impressionism|style_0005_Impressionism"
    "expressionism|style_0010_Expressionism"
    "romanticism|style_0007_Romanticism"
    "minimalism|style_0008_Minimalism"
)

declare -A STYLE_PROMPTS NEUTRAL_PROMPTS STYLE_IMAGES

STYLE_PROMPTS[baroque]="A baroque landscape [v]"
STYLE_PROMPTS[impressionism]="An impressionist garden [v]"
STYLE_PROMPTS[expressionism]="An expressionist forest [v]"
STYLE_PROMPTS[romanticism]="A romantic mountain scene [v]"
STYLE_PROMPTS[minimalism]="A minimalist cityscape [v]"

# Keep the neutral prompt broad so the prompt does not compete with the LoRA.
NEUTRAL_PROMPTS[baroque]="A detailed painting"
NEUTRAL_PROMPTS[impressionism]="A detailed painting"
NEUTRAL_PROMPTS[expressionism]="A detailed painting"
NEUTRAL_PROMPTS[romanticism]="A detailed painting"
NEUTRAL_PROMPTS[minimalism]="A detailed painting"

STYLE_IMAGES[baroque]=$(find "$STYLE_IMG_DIR/style_0000_Baroque" -maxdepth 1 -name "*.jpg" | head -1)
STYLE_IMAGES[impressionism]=$(find "$STYLE_IMG_DIR/style_0005_Impressionism" -maxdepth 1 -name "*.jpg" | head -1)
STYLE_IMAGES[expressionism]=$(find "$STYLE_IMG_DIR/style_0010_Expressionism" -maxdepth 1 -name "*.jpg" | head -1)
STYLE_IMAGES[romanticism]=$(find "$STYLE_IMG_DIR/style_0007_Romanticism" -maxdepth 1 -name "*.jpg" | head -1)
STYLE_IMAGES[minimalism]=$(find "$STYLE_IMG_DIR/style_0008_Minimalism" -maxdepth 1 -name "*.jpg" | head -1)

# Only the settings that matter for the prompt-sensitivity question.
TEMPS=(0.1)
TOPKS=(none 1)
ALPHAS=(1.0)
PROMPT_KINDS=(style neutral)

for ckpt_entry in "${CHECKPOINTS[@]}"; do
    ckpt_name="${ckpt_entry%%|*}"
    ckpt="${ckpt_entry##*|}"
    echo ""
    echo "========================================"
    echo "Checkpoint: $ckpt_name"
    echo "========================================"

    for style_entry in "${STYLE_CASES[@]}"; do
        style="${style_entry%%|*}"
        style_key="${style_entry##*|}"

        style_img="${STYLE_IMAGES[$style]}"
        style_prompt="${STYLE_PROMPTS[$style]}"
        neutral_prompt="${NEUTRAL_PROMPTS[$style]}"

        if [[ -z "$style_img" || ! -f "$style_img" ]]; then
            echo "[WARN] Missing style image for $style — skipping"
            continue
        fi

        for prompt_kind in "${PROMPT_KINDS[@]}"; do
            if [[ "$prompt_kind" == "style" ]]; then
                prompt="$style_prompt"
            else
                prompt="$neutral_prompt"
            fi

            for temp in "${TEMPS[@]}"; do
                for topk in "${TOPKS[@]}"; do
                    for alpha in "${ALPHAS[@]}"; do
                        run "$ckpt_name" "$ckpt" "$prompt_kind" "$style_key" "$style_img" "$prompt" "$temp" "$topk" "$alpha"
                    done
                done
            done
        done
    done
done

SUMMARY_JSON="$OUT_ROOT/summary.json"
cat > "$SUMMARY_JSON" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "run_tag": "$RUN_TAG",
  "output_root": "$OUT_ROOT",
  "runs": $RUN_COUNT,
  "failures": $FAIL_COUNT,
  "checkpoints": ["stage2_v23", "stage1_v21"],
  "styles": ["baroque", "impressionism", "expressionism", "romanticism", "minimalism"],
  "prompt_kinds": ["style", "neutral"],
  "status": "completed"
}
EOF

echo ""
echo "========================================"
echo "Retrospective sweep complete."
echo "Total runs: $RUN_COUNT"
echo "Failures:   $FAIL_COUNT"
echo "Output:     $OUT_ROOT"
echo "Summary:    $SUMMARY_JSON"
echo "Finished:   $(date)"
echo "========================================"
