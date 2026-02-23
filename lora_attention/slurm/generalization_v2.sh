#!/bin/bash
#
# v2.0 Generalization sweep — tests routing quality after Stage 1 v2.0 training.
#
# Runs the same WikiArt query images used in the v1.0 generalization experiment
# (§15 of roadmap) but with the new LoRARankEncoder (per-tensor routing).
# Produces comparable results so v1.0 vs v2.0 routing can be directly compared.
#
# Experiment groups:
#   Exp A — Singleton styles: one expert per style.  Run twice (GT in / GT out).
#   Exp B — Multi-expert styles: multiple experts per style.
#   Exp C — Zero-shot styles: no pool expert for this style at all.
#
# Usage:
#   cd /home/eyavuz21/repos/MoLoRAs
#   sbatch lora_attention/slurm/generalization_v2.sh
#
# Override checkpoint:
#   CKPT=/scratch/eyavuz21/lora_attention/stage1_v2/checkpoint-10000/encoder.pt \
#   sbatch lora_attention/slurm/generalization_v2.sh
#

#SBATCH --job-name=MoEv2-Generalize
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

# ── Modules ────────────────────────────────────────────────
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

PYTHON="/home/eyavuz21/.conda/envs/B-LoRA_2/bin/python"

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/inference_v2.py"
WIKIART="/home/eyavuz21/datasets/wikiart"
ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
CKPT="${CKPT:-/scratch/eyavuz21/lora_attention/stage1_v2/latest.pt}"
OUT_ROOT="${OUT_ROOT:-/scratch/eyavuz21/lora_attention/generalization_v2}"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

echo "Checkpoint: $CKPT"
echo "Output:     $OUT_ROOT"
mkdir -p "$OUT_ROOT"

# ── Shared settings ─────────────────────────────────────────
TEMP=0.1
ALPHA=1.0
NUM_IMAGES=4
STEPS=30
GUIDANCE=7.5
SEED=42

# Helper: run one inference call
run_inference() {
    local label="$1"        # e.g. expA_inpool/Baroque
    local style_img="$2"    # full path to query image
    local prompt="$3"       # text prompt
    local gt_expert="$4"    # e.g. style_0000_Baroque (empty string = no GT label)
    local exclude="$5"      # space-separated experts to exclude (empty = none)

    local out_dir="$OUT_ROOT/$label"
    mkdir -p "$out_dir"

    echo ""
    echo "── $label ──────────────────────────────────"
    echo "   query:  $style_img"
    echo "   prompt: $prompt"
    [[ -n "$gt_expert"  ]] && echo "   GT:     $gt_expert"
    [[ -n "$exclude"    ]] && echo "   excl:   $exclude"

    "$PYTHON" "$SCRIPT" \
        --checkpoint   "$CKPT" \
        --style_image  "$style_img" \
        --prompt       "$prompt" \
        --output_dir   "$out_dir" \
        --zoo_dir      "$ZOO_DIR" \
        --cache_dir    "$CACHE_DIR" \
        --temperature  "$TEMP" \
        --style_alpha  "$ALPHA" \
        --num_images   "$NUM_IMAGES" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$GUIDANCE" \
        --seed         "$SEED" \
        --query_label  "$label" \
        ${gt_expert:+--gt_expert "$gt_expert"} \
        ${exclude:+--exclude_experts $exclude}
}

# ════════════════════════════════════════════════════════════
# EXPERIMENT A — SINGLETON RECOGNITION
# Each style has exactly 1 expert. Run A1 (GT in pool) and
# A2 (GT excluded) with the same WikiArt query.
# Direct comparison baseline: v1.0 got ~0% top-1 here.
# ════════════════════════════════════════════════════════════
echo ""
echo "════════ Exp A — Singleton recognition ════════"

declare -a EXP_A=(
    "Baroque|style_0000_Baroque|Baroque|adriaen-brouwer_a-boor-asleep.jpg|A Baroque painting"
    "Cubism|style_0003_Cubism|Cubism|adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg|A Cubism painting"
    "Fauvism|style_0148_Fauvism|Fauvism|abraham-manievich_artist-s-wife-1937.jpg|A Fauvism painting"
    "Northern_Renaissance|style_0014_Northern_Renaissance|Northern_Renaissance|albrecht-altdorfer_alpine-landscape-with-church-1522.jpg|A Northern Renaissance painting"
    "Early_Renaissance|style_0084_Early_Renaissance|Early_Renaissance|andrea-del-castagno_crucifixion-1.jpg|An Early Renaissance painting"
    "High_Renaissance|style_0172_High_Renaissance|High_Renaissance|andrea-del-sarto_archangel-raphael-with-tobias-st-lawrence-and-the-donor-leonardo-di-lorenzo-morelli-1512.jpg|A High Renaissance painting"
    "Color_Field|style_0189_Color_Field_Painting|Color_Field_Painting|ad-reinhardt_abstract-painiting-1963.jpg|A Color Field painting"
)

for entry in "${EXP_A[@]}"; do
    IFS='|' read -r label gt_expert wikiart_dir wikiart_file prompt <<< "$entry"
    style_img="$WIKIART/$wikiart_dir/$wikiart_file"
    [[ ! -f "$style_img" ]] && style_img=$(find "$WIKIART/$wikiart_dir" -maxdepth 1 -name "*.jpg" | head -1)

    # A1: GT expert in pool
    run_inference "expA_inpool/$label"   "$style_img" "$prompt" "$gt_expert" ""
    # A2: GT expert excluded
    run_inference "expA_holdout/$label"  "$style_img" "$prompt" "$gt_expert" "$gt_expert"
done

# ════════════════════════════════════════════════════════════
# EXPERIMENT B — MULTI-EXPERT SPECIALISATION
# Style has multiple experts. Do the matching ones win?
# v1.0: only Post-Impressionism succeeded.
# ════════════════════════════════════════════════════════════
echo ""
echo "════════ Exp B — Multi-expert specialisation ════════"

declare -a EXP_B=(
    "Post_Impressionism|Post_Impressionism|abraham-manievich_autumn-day.jpg|A Post-Impressionism painting"
    "Impressionism|Impressionism|abdullah-suriosubroto_air-terjun.jpg|An Impressionism painting"
    "Expressionism|Expressionism|abidin-dino_drawing-pain-1968.jpg|An Expressionism painting"
    "Romanticism|Romanticism|adolphe-joseph-thomas-monticelli_an-evening-at-the-paiva.jpg|A Romanticism painting"
    "Abstract_Expressionism|Abstract_Expressionism|aaron-siskind_acolman-1-1955.jpg|An Abstract Expressionism painting"
)

for entry in "${EXP_B[@]}"; do
    IFS='|' read -r label wikiart_dir wikiart_file prompt <<< "$entry"
    style_img="$WIKIART/$wikiart_dir/$wikiart_file"
    [[ ! -f "$style_img" ]] && style_img=$(find "$WIKIART/$wikiart_dir" -maxdepth 1 -name "*.jpg" | head -1)

    run_inference "expB/$label" "$style_img" "$prompt" "" ""
done

# ════════════════════════════════════════════════════════════
# EXPERIMENT C — ZERO-SHOT TRANSFER
# Style not in pool at all. Are proxies art-historically sensible?
# ════════════════════════════════════════════════════════════
echo ""
echo "════════ Exp C — Zero-shot transfer ════════"

declare -a EXP_C=(
    "Pointillism|Pointillism|andre-derain_boats-at-collioure-1905.jpg|A Pointillism painting"
    "Analytical_Cubism|Analytical_Cubism|albert-gleizes_acrobats-1916.jpg|An Analytical Cubism painting"
    "Synthetic_Cubism|Synthetic_Cubism|georges-braque_aria-de-bach-1913.jpg|A Synthetic Cubism painting"
    "Action_painting|Action_painting|franz-kline_accent-grave-1955.jpg|An Action Painting"
    "Mannerism|Mannerism_Late_Renaissance|agnolo-bronzino_adoration-of-the-cross-with-the-brazen-serpent.jpg|A Mannerism painting"
    "Rococo|Rococo|allan-ramsay_charlotte-sophia-of-mecklenburg-strelitz-1762.jpg|A Rococo painting"
    "Ukiyo_e|Ukiyo_e|hiroshige_a-bridge-across-a-deep-gorge.jpg|A Ukiyo-e painting"
)

for entry in "${EXP_C[@]}"; do
    IFS='|' read -r label wikiart_dir wikiart_file prompt <<< "$entry"
    style_img="$WIKIART/$wikiart_dir/$wikiart_file"
    [[ ! -f "$style_img" ]] && style_img=$(find "$WIKIART/$wikiart_dir" -maxdepth 1 -name "*.jpg" | head -1)

    run_inference "expC/$label" "$style_img" "$prompt" "" ""
done

echo ""
echo "========================================"
echo "All runs complete."
echo "Results: $OUT_ROOT"
echo "Finished: $(date)"
echo "========================================"
