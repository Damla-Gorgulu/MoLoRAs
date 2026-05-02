#!/bin/bash
# Submit smoke, mini, and full MegaStyle diversity pipelines.

set -euo pipefail

ROOT="/scratch/eyavuz21/datasets/MegaStyle-diverse"
SLURM_DIR="/home/eyavuz21/repos/MoLoRAs/lora_attention/slurm"

submit_chain() {
  local name="$1"
  local limit_styles="$2"
  local k="$3"
  local batch_size="$4"
  local out="$ROOT/$name"
  local manifest_dir="$out/manifest"
  local embed_dir="$out/embeddings"
  local select_dir="$out/selection"
  local export_dir="$out/export"

  mkdir -p "$out"

  local j1
  local j2
  local j3
  local j4

  j1=$(sbatch --parsable \
    --export=ALL,OUTPUT_DIR="$manifest_dir",CAP=1,LIMIT_STYLES="$limit_styles" \
    "$SLURM_DIR/build_megastyle_cap_manifest.sh")

  j2=$(sbatch --parsable --dependency=afterok:"$j1" \
    --export=ALL,MANIFEST_PATH="$manifest_dir/manifest.jsonl",OUTPUT_DIR="$embed_dir",BATCH_SIZE="$batch_size" \
    "$SLURM_DIR/embed_megastyle_clip.sh")

  j3=$(sbatch --parsable --dependency=afterok:"$j2" \
    --export=ALL,EMBEDDINGS_PATH="$embed_dir/embeddings.pt",METADATA_PATH="$embed_dir/metadata.json",OUTPUT_DIR="$select_dir",K="$k" \
    "$SLURM_DIR/select_megastyle_diverse_subset.sh")

  j4=$(sbatch --parsable --dependency=afterok:"$j3" \
    --export=ALL,SELECTION_PATH="$select_dir/selection.json",OUTPUT_DIR="$export_dir" \
    "$SLURM_DIR/export_megastyle_subset.sh")

  printf '%s\n' "$name manifest=$j1 embed=$j2 select=$j3 export=$j4"
}

submit_chain smoke 128 20 32
submit_chain mini 5000 200 64
submit_chain full 0 200 64
