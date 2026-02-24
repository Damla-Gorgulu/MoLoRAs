#!/bin/bash
#
# Alpha diagnostic sweep.
# One query (baroque), one prompt, sweep alpha 0.5→5.0 + norm_match + oracle.
# Purpose: determine if the synthesised LoRA has any visual effect at higher scale.
#
# ~12 runs × ~50s ≈ 10 min on V100.
#
# Usage:
#   sbatch lora_attention/slurm/alpha_diag.sh
#

#SBATCH --job-name=alpha-diag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=01:00:00
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
WIKIART="/home/eyavuz21/datasets/wikiart"
ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
CKPT="${CKPT:-/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt}"
OUT_ROOT="/scratch/eyavuz21/lora_attention/alpha_diag"

export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"
echo "CKPT:     $CKPT"
echo "OUT_ROOT: $OUT_ROOT"

# ── Fixed query ──────────────────────────────────────────────
QUERY="$WIKIART/Baroque/adriaen-brouwer_a-boor-asleep.jpg"
PROMPT="A Baroque painting"
GT="style_0000_Baroque"
REF_BLORA="$ZOO_DIR/style_0000_Baroque/pytorch_lora_weights.safetensors"
SEED=42
STEPS=30

FAIL=0

run() {
    local tag="$1"; shift
    local extra=("$@")
    local out="$OUT_ROOT/$tag"
    mkdir -p "$out"
    echo "── $tag ──"
    "$PYTHON" "$SCRIPT" \
        --checkpoint          "$CKPT" \
        --style_image         "$QUERY" \
        --prompt              "$PROMPT" \
        --output_dir          "$out" \
        --zoo_dir             "$ZOO_DIR" \
        --cache_dir           "$CACHE_DIR" \
        --num_images          1 \
        --num_inference_steps "$STEPS" \
        --guidance_scale      7.5 \
        --seed                "$SEED" \
        --gt_expert           "$GT" \
        --query_label         "$tag" \
        "${extra[@]}" \
    || { echo "  !! FAILED: $tag"; FAIL=$((FAIL+1)); }
}

# ── 1. Vanilla SDXL — zero LoRA ─────────────────────────────
run "vanilla"  --temperature 0.005 --product_synth --style_alpha 0.0

# ── 2. Reference real B-LoRA (ground truth) ─────────────────
run "ref_blora_a1"  --reference_blora "$REF_BLORA" --style_alpha 1.0

# ── 3. Synthesised — alpha sweep, τ=0.005, full pool ─────────
for alpha in 0.5 1.0 2.0 3.0 5.0; do
    run "synth_t0.005_a${alpha}"  --temperature 0.005 --product_synth --style_alpha "$alpha"
done

# ── 4. Synthesised — oracle top_k=1, alpha sweep ─────────────
for alpha in 1.0 3.0 5.0; do
    run "synth_oracle_a${alpha}"  --temperature 0.005 --top_k 1 --product_synth --style_alpha "$alpha"
done

# ── 5. norm_match (auto-scales synth to match real B-LoRA norm) ─
run "synth_normmatch"   --temperature 0.005  --product_synth --style_alpha 1.0 --norm_match
run "synth_oracle_normmatch" --temperature 0.005 --top_k 1 --product_synth --style_alpha 1.0 --norm_match

echo ""
echo "========================================"
echo "alpha_diag complete.  Failures: $FAIL"
echo "Output: $OUT_ROOT"
echo "Finished: $(date)"
echo "========================================"

# ── 6. Build contact sheet ───────────────────────────────────
echo "Building contact sheet..."
"$PYTHON" "$REPO_ROOT/lora_attention/scripts/alpha_diag_sheet.py" \
    --run_dir  "$OUT_ROOT" \
    --query    "$QUERY" \
    --out      "$OUT_ROOT/contact_sheet.png" \
|| echo "[warn] contact sheet failed — check scripts/alpha_diag_sheet.py"
