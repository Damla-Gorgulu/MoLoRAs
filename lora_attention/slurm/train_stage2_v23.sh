#!/bin/bash
#
# v2.3 Stage 2: 3-Stage Training - LDM → LDM+LB → LDM+smaller LB
#
# Key improvements over v2.2:
#   - 3-Stage loss scheduling:
#       Stage 1 (0-2000): LDM loss only (learn routing for image quality)
#       Stage 2 (2000-6000): LDM + LB (balance expert usage)
#       Stage 3 (6000-8000): LDM + smaller LB (refine routing)
#   - 3-Stage temperature schedule:
#       Stage 1: τ=1.0 → 2.0 (increase diversity for exploration)
#       Stage 2: τ=2.0 → 0.3 (sharpen routing for exploitation)
#
# SLURM output goes to stage2_v23/ (fresh run)
#
# Usage:
#   sbatch slurm/train_stage2_v23.sh
#
# Override examples:
#   MAX_STEPS=10000 sbatch slurm/train_stage2_v23.sh
#   RESUME=/scratch/eyavuz21/lora_attention/stage2_v23/latest.pt sbatch slurm/train_stage2_v23.sh

#SBATCH --job-name=MoELoRA-S2v23
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --partition=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:tesla_v100:1
#SBATCH --qos=ai
#SBATCH --time=24:00:00
#SBATCH --output=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.log
#SBATCH --error=/home/eyavuz21/repos/MoLoRAs/lora_attention/logs/%x-%j.err

set -euo pipefail

echo "========================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================"

# ── Modules ────────────────────────────────────────────────
module load cuda/11.8.0
module load cudnn/8.2.1
module load conda3/latest

# ── Conda environment ───────────────────────────────────────
source activate B-LoRA_2 || conda activate B-LoRA_2
PYTHON="python"
export PYTHONUNBUFFERED=1

# ── Paths ───────────────────────────────────────────────────
REPO_ROOT="/home/eyavuz21/repos/MoLoRAs"
SCRIPT="$REPO_ROOT/lora_attention/train_stage2_v2.py"
export PYTHONPATH="$REPO_ROOT:${REPO_ROOT}/../B-LoRA-fresh/B-LoRA:${PYTHONPATH:-}"

ZOO_DIR="/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
CACHE_DIR="/scratch/eyavuz21/lora_attention"
WIKIART_DIR="/home/eyavuz21/datasets/wikiart"
LABEL_MAP_PATH="$CACHE_DIR/wikiart_label_map.json"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/eyavuz21/lora_attention/stage2_v23}"
STAGE1_CKPT="${STAGE1_CKPT:-/scratch/eyavuz21/lora_attention/stage1_v21/latest.pt}"
LOG_DIR="$REPO_ROOT/lora_attention/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# ── Hyperparameters (override via env vars) ─────────────────
MAX_STEPS="${MAX_STEPS:-8000}"
LR="${LR:-5e-5}"
POOL_MIN="${POOL_MIN:-5}"
POOL_MAX="${POOL_MAX:-20}"
LORA_ALPHA="${LORA_ALPHA:-1.0}"
LAMBDA_START="${LAMBDA_START:-0.1}"
LAMBDA_END="${LAMBDA_END:-0.01}"       # reduced final lambda for refinement
LB_START_STEP="${LB_START_STEP:-2000}"  # start LB after LDM warmup
LB_END_STEP="${LB_END_STEP:-6000}"      # end LB before final refinement
TAU_START="${TAU_START:-1.0}"          # start sharp for quality learning
TAU_MID="${TAU_MID:-2.0}"              # peak diversity for exploration
TAU_END="${TAU_END:-0.3}"              # sharp routing for final refinement
TAU_MID_STEP="${TAU_MID_STEP:-3000}"   # midpoint of temp schedule
EMA_BETA="${EMA_BETA:-0.99}"           # ~100-step EMA window for expert usage
MAX_IMAGES="${MAX_IMAGES:-500}"
SAVE_EVERY="${SAVE_EVERY:-500}"
LOG_EVERY="${LOG_EVERY:-25}"
RESUME="${RESUME:-}"
PRODUCT_SYNTH="${PRODUCT_SYNTH:-1}"

# Auto-resume from latest.pt if it exists and RESUME not explicitly set
if [[ -z "$RESUME" && -f "$OUTPUT_DIR/latest.pt" ]]; then
    RESUME="$OUTPUT_DIR/latest.pt"
    echo "Auto-resuming from: $RESUME"
fi

echo "----------------------------------------"
echo "stage:        Stage 2 v2.3 (3-stage training)"
echo "STAGE1_CKPT:  $STAGE1_CKPT"
echo "MAX_STEPS:    $MAX_STEPS"
echo "LR:           $LR"
echo "POOL:         [$POOL_MIN, $POOL_MAX]"
echo "LORA_ALPHA:   $LORA_ALPHA"
echo "λ_lb:         $LAMBDA_START → $LAMBDA_END (steps $LB_START_STEP-$LB_END_STEP)"
echo "τ:            $TAU_START → $TAU_MID → $TAU_END (mid at $TAU_MID_STEP)"
echo "EMA_BETA:     $EMA_BETA"
echo "PRODUCT_SYNTH: $PRODUCT_SYNTH"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "----------------------------------------"

SYNTH_ARG="--product_synth"
if [[ "$PRODUCT_SYNTH" == "0" ]]; then
    SYNTH_ARG="--legacy_synth"
fi

# Verify prerequisites
if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "ERROR: Stage 1 v2.1 checkpoint not found: $STAGE1_CKPT"
    echo "Run: sbatch slurm/train_stage1_v21.sh"
    exit 1
fi
if [[ ! -f "$LABEL_MAP_PATH" ]]; then
    echo "ERROR: WikiArt label map not found: $LABEL_MAP_PATH"
    echo "Run: sbatch slurm/precompute_v2.sh"
    exit 1
fi

"$PYTHON" "$SCRIPT" \
    --zoo_dir            "$ZOO_DIR" \
    --cache_dir          "$CACHE_DIR" \
    --output_dir         "$OUTPUT_DIR" \
    --wikiart_dir        "$WIKIART_DIR" \
    --label_map_path     "$LABEL_MAP_PATH" \
    --stage1_ckpt        "$STAGE1_CKPT" \
    --max_steps          "$MAX_STEPS" \
    --lr                 "$LR" \
    --min_pool_size      "$POOL_MIN" \
    --max_pool_size      "$POOL_MAX" \
    --lora_alpha         "$LORA_ALPHA" \
    --lb_start_step      "$LB_START_STEP" \
    --lb_end_step        "$LB_END_STEP" \
    --lambda_start       "$LAMBDA_START" \
    --lambda_end         "$LAMBDA_END" \
    --tau_start          "$TAU_START" \
    --tau_mid            "$TAU_MID" \
    --tau_end            "$TAU_END" \
    --tau_mid_step       "$TAU_MID_STEP" \
    --ema_beta           "$EMA_BETA" \
    --max_images_per_style "$MAX_IMAGES" \
    --save_every         "$SAVE_EVERY" \
    --log_every          "$LOG_EVERY" \
    --mixed_precision    fp16 \
    --no_normalize_keys  \
    $SYNTH_ARG \
    ${RESUME:+--resume_from "$RESUME"}

echo "========================================"
echo "Finished:  $(date)"
echo "========================================"
