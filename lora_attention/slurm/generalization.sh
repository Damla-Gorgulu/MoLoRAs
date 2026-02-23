#!/bin/bash
#
# MoELoRA Generalization Experiments
#
# Three experiment types:
#   Exp A — Held-out singleton: GT expert excluded from pool; can siblings reconstruct it?
#   Exp B — Sibling reconstruction: multiple experts of same category; routes to right cluster?
#   Exp C — Novel style: no expert in pool at all; which experts get recruited?
#
# All runs use τ=0.1, α=1.0 (best config from sweep).
# WikiArt images used as queries (fresh / unseen during training).
#
# Usage:
#   sbatch lora_attention/slurm/generalization.sh
#

#SBATCH --job-name=MoELoRA-Generalize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=08:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -eo pipefail

echo "========================================"
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "========================================"

# ── Modules ────────────────────────────────────────────────
module load cuda/11.8.0
module load cudnn/8.2.1

# ── Python (explicit path — avoid conda activate in SLURM) ─
PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference.py"
WIKIART="/home/eyavuz21/datasets/wikiart"
CKPT="/scratch/eyavuz21/lora_attention/stage1/latest.pt"
OUT_ROOT="/scratch/eyavuz21/lora_attention/generalization"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

echo "Python: $PYTHON ($($PYTHON --version 2>&1))"
echo "Output: $OUT_ROOT"
mkdir -p "$OUT_ROOT"

# ── Shared inference settings ───────────────────────────────
TEMP=0.1
ALPHA=1.0
NUM_IMAGES=4
STEPS=30
GUIDANCE=7.5
SEED=42

# ── Shared prompt (generic enough for any style) ────────────
PROMPT="A painting in [v] style"

# ════════════════════════════════════════════════════════════
# EXPERIMENT A — HELD-OUT SINGLETON
# For each of the 8 singleton pool experts, we run:
#   A1: full pool (GT in pool) — baseline routing
#   A2: pool minus GT expert  — can siblings reconstruct?
# Query image: fresh WikiArt image of the same style.
# ════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo "EXPERIMENT A — HELD-OUT SINGLETON"
echo "════════════════════════════════════════"

# Format: "style_token|gt_expert_substring|wikiart_dir|wikiart_file"
declare -a EXP_A=(
    "Baroque|Baroque|Baroque|adriaen-brouwer_a-boor-asleep.jpg"
    "Cubism|Cubism|Cubism|adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg"
    "Fauvism|Fauvism|Fauvism|abraham-manievich_artist-s-wife-1937.jpg"
    "Northern_Ren|Northern_Renaissance|Northern_Renaissance|albrecht-altdorfer_alpine-landscape-with-church-1522.jpg"
    "Early_Ren|Early_Renaissance|Early_Renaissance|andrea-del-castagno_crucifixion-1.jpg"
    "High_Ren|High_Renaissance|High_Renaissance|andrea-del-sarto_archangel-raphael-with-tobias-st-lawrence-and-the-donor-leonardo-di-lorenzo-morelli-1512.jpg"
    "Color_Field|Color_Field_Painting|Color_Field_Painting|ad-reinhardt_abstract-painiting-1963.jpg"
)

for entry in "${EXP_A[@]}"; do
    IFS='|' read -r label gt_sub wikart_dir wikart_file <<< "$entry"
    style_image="$WIKIART/$wikart_dir/$wikart_file"

    if [[ ! -f "$style_image" ]]; then
        # Fall back: pick first available jpg
        style_image=$(find "$WIKIART/$wikart_dir" -maxdepth 1 -name "*.jpg" -print -quit 2>/dev/null || true)
        if [[ -z "$style_image" ]]; then
            echo "[WARN] No image for $label in $WIKIART/$wikart_dir — skipping"
            continue
        fi
        echo "[INFO] Using fallback image: $style_image"
    fi

    echo ""
    echo "── A1 (GT in pool): $label ─────────────"
    out_dir="$OUT_ROOT/expA_inpool/$label"
    mkdir -p "$out_dir"
    $PYTHON "$SCRIPT" \
        --checkpoint          "$CKPT" \
        --style_image         "$style_image" \
        --prompt              "$PROMPT" \
        --output_dir          "$out_dir" \
        --temperature         "$TEMP" \
        --style_alpha         "$ALPHA" \
        --num_images          "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale      "$GUIDANCE" \
        --seed                "$SEED" \
        --gt_expert           "$gt_sub" \
        --query_label         "A1_inpool_${label}_"
    echo "  → $out_dir"

    echo ""
    echo "── A2 (GT held-out): $label ─────────────"
    out_dir="$OUT_ROOT/expA_heldout/$label"
    mkdir -p "$out_dir"
    $PYTHON "$SCRIPT" \
        --checkpoint          "$CKPT" \
        --style_image         "$style_image" \
        --prompt              "$PROMPT" \
        --output_dir          "$out_dir" \
        --temperature         "$TEMP" \
        --style_alpha         "$ALPHA" \
        --num_images          "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale      "$GUIDANCE" \
        --seed                "$SEED" \
        --gt_expert           "$gt_sub" \
        --exclude_experts     "$gt_sub" \
        --query_label         "A2_heldout_${label}_"
    echo "  → $out_dir"
done

# ════════════════════════════════════════════════════════════
# EXPERIMENT B — SIBLING RECONSTRUCTION
# Styles with multiple experts in pool.  Query from WikiArt
# (unseen during training). Expect routing to cluster on same
# category experts — tests within-category consistency.
# ════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo "EXPERIMENT B — SIBLING RECONSTRUCTION"
echo "════════════════════════════════════════"

declare -a EXP_B=(
    "Impressionism|Impressionism|abdullah-suriosubroto_air-terjun.jpg"
    "Expressionism|Expressionism|abidin-dino_drawing-pain-1968.jpg"
    "Post_Impressionism|Post_Impressionism|a.y.-jackson_barns-1926.jpg"
    "Romanticism|Romanticism|adolphe-joseph-thomas-monticelli_an-evening-at-the-paiva.jpg"
    "Abstract_Expr|Abstract_Expressionism|aaron-siskind_acolman-1-1955.jpg"
)

for entry in "${EXP_B[@]}"; do
    IFS='|' read -r label wikart_dir wikart_file <<< "$entry"
    style_image="$WIKIART/$wikart_dir/$wikart_file"

    if [[ ! -f "$style_image" ]]; then
        style_image=$(find "$WIKIART/$wikart_dir" -maxdepth 1 -name "*.jpg" -print -quit 2>/dev/null || true)
        if [[ -z "$style_image" ]]; then
            echo "[WARN] No image for $label in $WIKIART/$wikart_dir — skipping"
            continue
        fi
        echo "[INFO] Using fallback image: $style_image"
    fi

    echo ""
    echo "── B: $label ─────────────"
    out_dir="$OUT_ROOT/expB_sibling/$label"
    mkdir -p "$out_dir"
    $PYTHON "$SCRIPT" \
        --checkpoint          "$CKPT" \
        --style_image         "$style_image" \
        --prompt              "$PROMPT" \
        --output_dir          "$out_dir" \
        --temperature         "$TEMP" \
        --style_alpha         "$ALPHA" \
        --num_images          "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale      "$GUIDANCE" \
        --seed                "$SEED" \
        --gt_expert           "$label" \
        --query_label         "B_sibling_${label}_"
    echo "  → $out_dir"
done

# ════════════════════════════════════════════════════════════
# EXPERIMENT C — NOVEL / DISTANT STYLES
# No expert from this category exists in pool.
# Sub-groups:
#   C1 (closer to pool):  Pointillism, Analytical_Cubism, Synthetic_Cubism
#   C2 (distant from pool): Ukiyo_e, Rococo, Mannerism_Late_Renaissance, Action_painting
# ════════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo "EXPERIMENT C — NOVEL STYLES"
echo "════════════════════════════════════════"

# Format: "label|group|wikiart_dir|wikiart_file"
declare -a EXP_C=(
    # C1 — Closer analogues in pool
    "Pointillism|C1_close|Pointillism|andre-derain_boats-at-collioure-1905.jpg"
    "Analytical_Cubism|C1_close|Analytical_Cubism|albert-gleizes_acrobats-1916.jpg"
    "Synthetic_Cubism|C1_close|Synthetic_Cubism|ad-reinhardt_collage-1940.jpg"
    # C2 — Distant from all pool styles
    "Ukiyo_e|C2_distant|Ukiyo_e|hiroshige_a-bridge-across-a-deep-gorge.jpg"
    "Rococo|C2_distant|Rococo|allan-ramsay_charlotte-sophia-of-mecklenburg-strelitz-1762.jpg"
    "Mannerism|C2_distant|Mannerism_Late_Renaissance|agnolo-bronzino_a-portrait-of-giuliano-di-piero-de-medici.jpg"
    "Action_painting|C2_distant|Action_painting|antonio-palolo_untitled-1992.jpg"
)

for entry in "${EXP_C[@]}"; do
    IFS='|' read -r label group wikart_dir wikart_file <<< "$entry"
    style_image="$WIKIART/$wikart_dir/$wikart_file"

    if [[ ! -f "$style_image" ]]; then
        style_image=$(find "$WIKIART/$wikart_dir" -maxdepth 1 -name "*.jpg" -print -quit 2>/dev/null || true)
        if [[ -z "$style_image" ]]; then
            echo "[WARN] No image for $label in $WIKIART/$wikart_dir — skipping"
            continue
        fi
        echo "[INFO] Using fallback image: $style_image"
    fi

    echo ""
    echo "── C ($group): $label ─────────────"
    out_dir="$OUT_ROOT/expC_novel/${group}/${label}"
    mkdir -p "$out_dir"
    $PYTHON "$SCRIPT" \
        --checkpoint          "$CKPT" \
        --style_image         "$style_image" \
        --prompt              "$PROMPT" \
        --output_dir          "$out_dir" \
        --temperature         "$TEMP" \
        --style_alpha         "$ALPHA" \
        --num_images          "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale      "$GUIDANCE" \
        --seed                "$SEED" \
        --gt_expert           "" \
        --query_label         "C_${group}_${label}_"
    echo "  → $out_dir"
done

echo ""
echo "========================================"
echo "Generalization experiments complete!"
echo "Output: $OUT_ROOT"
echo "Finished: $(date)"
echo ""
echo "Next: run analysis:"
echo "  $PYTHON $REPO_ROOT/lora_attention/analyse_generalization.py \\"
echo "    --results_dir $OUT_ROOT \\"
echo "    --csv $OUT_ROOT/report.csv"
echo "========================================"
