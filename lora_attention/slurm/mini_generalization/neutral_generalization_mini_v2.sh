#!/bin/bash
#
# Mini generalization v2 evaluation: singleton in-pool only.
#
# This keeps the prompt neutral and asks only the narrow question:
#   does the freshly trained singleton-only Stage 1 checkpoint retrieve the
#   correct in-pool expert strongly enough to change generation at all?
#
# Usage:
#   sbatch lora_attention/slurm/mini_generalization/neutral_generalization_mini_v2.sh

#SBATCH --job-name=MoELoRA-NeutralMiniV2
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
CACHE_DIR="${CACHE_DIR:-/scratch/eyavuz21/lora_attention}"
ROOT_BASE="${ROOT_BASE:-/scratch/eyavuz21/lora_attention/mini_generalization_v2}"
CKPT="${CKPT:-$ROOT_BASE/stage1_train/latest.pt}"
CKPT_TAG="${CKPT_TAG:-stage1_train}"
RUN_TAG="${RUN_TAG:-neutral_mini_v2}"
OUT_ROOT="${OUT_ROOT:-$ROOT_BASE/$CKPT_TAG/$RUN_TAG}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"

echo "Checkpoint: $CKPT"
echo "CKPT_TAG:   $CKPT_TAG"
echo "OUT_ROOT:   $OUT_ROOT"
echo "CACHE_DIR:  $CACHE_DIR"

NEUTRAL_PROMPT="A detailed painting"
TEMP=0.1
ALPHA=1.0
NUM_IMAGES=1
STEPS=30
GUIDANCE=7.5
SEED=42

run_inference() {
    local tag="$1"
    local style_img="$2"
    local gt_expert="$3"
    local topk="$4"

    local out_dir="$OUT_ROOT/$tag"
    mkdir -p "$out_dir"

    echo ""
    echo "── $tag ──────────────────────────────────"
    echo "   query:  $style_img"
    echo "   prompt: $NEUTRAL_PROMPT"
    echo "   GT:     $gt_expert"
    echo "   top_k:  $topk"

    local extra_args=(--gt_expert "$gt_expert")
    [[ "$topk" != "none" ]] && extra_args+=(--top_k "$topk")

    "$PYTHON" "$SCRIPT" \
        --checkpoint "$CKPT" \
        --style_image "$style_img" \
        --prompt "$NEUTRAL_PROMPT" \
        --output_dir "$out_dir" \
        --zoo_dir "$ZOO_DIR" \
        --cache_dir "$CACHE_DIR" \
        --temperature "$TEMP" \
        --style_alpha "$ALPHA" \
        --num_images "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed "$SEED" \
        --run_tag "$RUN_TAG" \
        --query_label "$tag" \
        "${extra_args[@]}"
}

declare -a EXP_INPOOL=(
    "Baroque|style_0000_Baroque|Baroque|adriaen-brouwer_a-boor-asleep.jpg"
    "Cubism|style_0003_Cubism|Cubism|adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg"
    "Fauvism|style_0148_Fauvism|Fauvism|abraham-manievich_artist-s-wife-1937.jpg"
)

for entry in "${EXP_INPOOL[@]}"; do
    IFS='|' read -r label gt_expert wikiart_dir wikiart_file <<< "$entry"
    style_img="/home/eyavuz21/datasets/wikiart/$wikiart_dir/$wikiart_file"
    if [[ ! -f "$style_img" ]]; then
        style_img=$(find "/home/eyavuz21/datasets/wikiart/$wikiart_dir" -maxdepth 1 -name "*.jpg" -print -quit 2>/dev/null || true)
    fi
    if [[ -z "${style_img:-}" || ! -f "$style_img" ]]; then
        echo "[WARN] No image for $label in /home/eyavuz21/datasets/wikiart/$wikiart_dir — skipping"
        continue
    fi

    run_inference "expA_inpool/$label/neutral_soft" "$style_img" "$gt_expert" "none"
    run_inference "expA_inpool/$label/neutral_top1" "$style_img" "$gt_expert" "1"
done

SUMMARY_JSON="$OUT_ROOT/summary.json"
cat > "$SUMMARY_JSON" <<EOF
{
  "job_id": "${SLURM_JOB_ID:-local}",
  "checkpoint": "$CKPT",
  "ckpt_tag": "$CKPT_TAG",
  "run_tag": "$RUN_TAG",
  "output_dir": "$OUT_ROOT",
  "prompt": "$NEUTRAL_PROMPT",
  "temperature": $TEMP,
  "style_alpha": $ALPHA,
  "num_images": $NUM_IMAGES,
  "steps": $STEPS,
  "guidance_scale": $GUIDANCE,
  "status": "completed"
}
EOF

echo ""
echo "========================================"
echo "Neutral mini v2 generalization complete!"
echo "Output:   $OUT_ROOT"
echo "Summary:  $SUMMARY_JSON"
echo "Finished: $(date)"
echo "========================================"
