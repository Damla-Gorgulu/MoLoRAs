#!/bin/bash
#
# Neutral-prompt alpha sweep for older v2 checkpoints.
#
# Goal:
#   Test whether the visually flat outputs from older checkpoints are caused by
#   insufficient LoRA magnitude rather than a total absence of style signal.
#   We keep the prompt neutral and vary only style_alpha and top-k routing.
#
# Output layout:
#   /scratch/eyavuz21/lora_attention/neutral_alpha_sweep_v1/<RUN_TAG>/<ckpt>/<style>/<alpha>/<routing>/
#
# Usage:
#   cd /home/eyavuz21/repos/MoLoRAs
#   sbatch lora_attention/slurm/neutral_alpha_sweep_v1.sh

#SBATCH --job-name=MoELoRA-NeutralAlpha
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
CACHE_DIR="/scratch/eyavuz21/lora_attention"
STYLE_IMG_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
RUN_TAG="${RUN_TAG:-neutral_alpha_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="/scratch/eyavuz21/lora_attention/neutral_alpha_sweep_v1/${RUN_TAG}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"
echo "RUN_TAG:  $RUN_TAG"
echo "OUT_ROOT: $OUT_ROOT"

SEED=42
NUM_IMAGES=1
STEPS=30
GUIDANCE=7.5
TEMP=0.1
PROMPT="A detailed painting"

RUN_COUNT=0
FAIL_COUNT=0

run() {
    local ckpt_name="$1"
    local ckpt="$2"
    local label="$3"
    local style_img="$4"
    local alpha="$5"
    local topk="$6"

    local out_dir="$OUT_ROOT/$ckpt_name/$label/alpha_${alpha}"
    if [[ "$topk" != "none" ]]; then
        out_dir="$out_dir/topk_${topk}"
    else
        out_dir="$out_dir/topk_none"
    fi
    mkdir -p "$out_dir"

    RUN_COUNT=$((RUN_COUNT + 1))
    echo ""
    echo "── Run $RUN_COUNT: $ckpt_name | $label | alpha=$alpha | topk=$topk ──"
    echo "   query:  $style_img"
    echo "   prompt: $PROMPT"

    local extra_args=()
    [[ "$topk" != "none" ]] && extra_args+=(--top_k "$topk")

    "$PYTHON" "$SCRIPT" \
        --checkpoint "$ckpt" \
        --style_image "$style_img" \
        --prompt "$PROMPT" \
        --output_dir "$out_dir" \
        --zoo_dir "$ZOO_DIR" \
        --cache_dir "$CACHE_DIR" \
        --temperature "$TEMP" \
        --style_alpha "$alpha" \
        --num_images "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --run_tag "$RUN_TAG" \
        --query_label "$ckpt_name/$label/alpha_${alpha}_k${topk}" \
        --product_synth \
        "${extra_args[@]}" \
    || { echo "  !! FAILED: $ckpt_name | $label | alpha=$alpha | topk=$topk"; FAIL_COUNT=$((FAIL_COUNT + 1)); }
}

declare -a CHECKPOINTS=(
    "stage1_v21|/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt"
    "stage2_v23|/scratch/eyavuz21/lora_attention/stage2_v23/latest.pt"
)

declare -a STYLE_CASES=(
    "baroque|style_0000_Baroque|style_0000_Baroque"
    "cubism|style_0003_Cubism|style_0003_Cubism"
    "fauvism|style_0148_Fauvism|style_0148_Fauvism"
)

declare -A STYLE_IMAGES
STYLE_IMAGES[style_0000_Baroque]=$(find "$STYLE_IMG_DIR/style_0000_Baroque" -maxdepth 1 -name "*.jpg" | head -1)
STYLE_IMAGES[style_0003_Cubism]=$(find "$STYLE_IMG_DIR/style_0003_Cubism" -maxdepth 1 -name "*.jpg" | head -1)
STYLE_IMAGES[style_0148_Fauvism]=$(find "$STYLE_IMG_DIR/style_0148_Fauvism" -maxdepth 1 -name "*.jpg" | head -1)

ALPHAS=(0.5 1.0 2.0 4.0)
TOPKS=(none 1)

for ckpt_entry in "${CHECKPOINTS[@]}"; do
    ckpt_name="${ckpt_entry%%|*}"
    ckpt="${ckpt_entry##*|}"
    echo ""
    echo "========================================"
    echo "Checkpoint: $ckpt_name"
    echo "========================================"

    for style_entry in "${STYLE_CASES[@]}"; do
        label="${style_entry%%|*}"
        wikiart_dir="${style_entry##*|}"
        style_img="${STYLE_IMAGES[$wikiart_dir]}"

        if [[ -z "$style_img" || ! -f "$style_img" ]]; then
            echo "[WARN] Missing style image for $label ($wikiart_dir) — skipping"
            continue
        fi

        for alpha in "${ALPHAS[@]}"; do
            for topk in "${TOPKS[@]}"; do
                run "$ckpt_name" "$ckpt" "$label" "$style_img" "$alpha" "$topk"
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
  "checkpoints": ["stage1_v21", "stage2_v23"],
  "styles": ["baroque", "cubism", "fauvism"],
  "alphas": [0.5, 1.0, 2.0, 4.0],
  "topks": ["none", 1],
  "prompt": "$PROMPT",
  "status": "completed"
}
EOF

echo ""
echo "========================================"
echo "Neutral alpha sweep complete."
echo "Total runs: $RUN_COUNT"
echo "Failures:   $FAIL_COUNT"
echo "Output:     $OUT_ROOT"
echo "Summary:    $SUMMARY_JSON"
echo "Finished:   $(date)"
echo "========================================"
