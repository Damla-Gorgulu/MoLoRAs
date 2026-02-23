"""
CLIP Similarity Matrix & WikiArt Label Map Generation.

Pre-computes:
  1. S ∈ ℝ^{N×N} — pairwise CLIP cosine similarity between all pool expert
     style images. Used for soft targets in v2.0 Stage 1.
  2. wikiart_label_map.json — mapping from WikiArt category name to list of
     pool expert indices that share that style label.

Usage:
    python -m lora_attention.utils.clip_similarity \
        --zoo_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/bloras \
        --image_dir /home/eyavuz21/repos/B-LoRA/blora_zoo/style_images \
        --wikiart_dir /home/eyavuz21/datasets/wikiart \
        --output_dir /scratch/eyavuz21/lora_attention
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DEFAULT_ZOO_DIR = "/home/eyavuz21/repos/B-LoRA/blora_zoo/bloras"
DEFAULT_IMAGE_DIR = "/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images"
DEFAULT_WIKIART_DIR = "/home/eyavuz21/datasets/wikiart"
DEFAULT_OUTPUT_DIR = "/scratch/eyavuz21/lora_attention"

_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPEG", ".JPG", ".PNG"}


# ──────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────
def _find_style_image(style_name: str, image_dirs: List[Path]) -> Optional[Path]:
    """Find the first image matching a style name in given directories."""
    for d in image_dirs:
        if not d.exists():
            continue
        # Sub-directory named after style
        sub = d / style_name
        if sub.is_dir():
            for p in sorted(sub.iterdir()):
                if p.suffix in _IMG_EXTENSIONS:
                    return p
        # Direct file match
        for p in sorted(d.iterdir()):
            if p.suffix in _IMG_EXTENSIONS and (
                p.stem == style_name or p.stem.startswith(style_name)
            ):
                return p
    return None


def compute_clip_embeddings(
    style_names: List[str],
    image_dirs: List[Path],
    clip_model_id: str = "openai/clip-vit-base-patch32",
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute CLIP embeddings for all pool experts.

    Args:
        style_names: List of style names (from LoRAPool.style_names).
        image_dirs:  Directories to search for images.
        clip_model_id: CLIP model identifier.
        device: Computation device.

    Returns:
        embeddings: Tensor (N, clip_dim) — L2-normalised CLIP features.
    """
    from transformers import CLIPProcessor, CLIPModel

    print(f"[CLIP] Loading model: {clip_model_id}")
    processor = CLIPProcessor.from_pretrained(clip_model_id)
    model = CLIPModel.from_pretrained(clip_model_id).to(device)
    model.eval()

    embeddings = []
    missing = []

    for idx, name in enumerate(style_names):
        img_path = _find_style_image(name, image_dirs)
        if img_path is None:
            missing.append(name)
            # Use zero embedding as placeholder (will have zero similarity)
            embeddings.append(torch.zeros(512))
            continue

        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            feat = model.get_image_features(pixel_values=pixel_values)
        embeddings.append(feat.squeeze(0).cpu())

        if (idx + 1) % 20 == 0:
            print(f"  [{idx + 1}/{len(style_names)}] encoded")

    if missing:
        print(f"  [warn] Missing images for {len(missing)} styles: {missing[:5]}...")

    return torch.stack(embeddings, dim=0)  # (N, clip_dim)


def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        embeddings: (N, d) — assumed L2-normalised from CLIP.

    Returns:
        S: (N, N) — S[i,j] = cos(emb_i, emb_j). Diagonal is ~1.0.
    """
    # Ensure normalisation
    emb = torch.nn.functional.normalize(embeddings, dim=-1)
    return emb @ emb.T  # (N, N)


def build_wikiart_label_map(
    style_names: List[str],
    wikiart_dir: str = DEFAULT_WIKIART_DIR,
) -> Dict[str, List[int]]:
    """
    Map WikiArt category names to lists of pool expert indices.

    Expert naming convention: "style_XXXX_CategoryName"
    WikiArt categories: directory names under wikiart_dir.

    Returns:
        Dict[str, List[int]]: e.g. {"Impressionism": [5, 86, 152, 204], ...}
    """
    # Extract style label from expert directory name
    # e.g. "style_0022_Post_Impressionism" → "Post_Impressionism"
    pattern = re.compile(r"^style_\d+_(.+)$")

    expert_to_category: Dict[int, str] = {}
    for idx, name in enumerate(style_names):
        m = pattern.match(name)
        if m:
            expert_to_category[idx] = m.group(1)

    # Get actual WikiArt categories (directory names)
    wikiart_path = Path(wikiart_dir)
    wikiart_categories = set()
    if wikiart_path.exists():
        wikiart_categories = {
            d.name for d in wikiart_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        }

    # Build map: category → expert indices
    label_map: Dict[str, List[int]] = {}
    for idx, cat in expert_to_category.items():
        label_map.setdefault(cat, []).append(idx)

    # Sort indices for determinism
    for cat in label_map:
        label_map[cat].sort()

    # Report coverage
    pool_categories = set(label_map.keys())
    covered = pool_categories & wikiart_categories
    uncovered_pool = pool_categories - wikiart_categories
    uncovered_wikiart = wikiart_categories - pool_categories

    print(f"\n[LabelMap] Pool categories with WikiArt data: {len(covered)}")
    print(f"  Pool categories without WikiArt: {uncovered_pool or 'none'}")
    print(f"  WikiArt categories without pool experts: {uncovered_wikiart or 'none'}")
    print(f"  Total expert→category mappings: {len(expert_to_category)}")

    return label_map


def soft_target(
    similarity_matrix: torch.Tensor,
    gt_idx: int,
    pool_indices: List[int],
    tau_label: float = 0.3,
) -> torch.Tensor:
    """
    Generate a soft target distribution from CLIP similarity.

    Args:
        similarity_matrix: (N_total, N_total) pairwise cosine sim.
        gt_idx:            Ground-truth expert index (global).
        pool_indices:      List of N expert indices in current pool.
        tau_label:         Temperature for soft target sharpness.
                           Lower → sharper (more like one-hot).

    Returns:
        target: (N,) — soft probability distribution over pool experts.
                Sums to 1.0. GT expert gets highest weight.
    """
    sims = similarity_matrix[gt_idx, pool_indices]   # (N,)
    return torch.softmax(sims / tau_label, dim=0)    # (N,)


# ──────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute CLIP similarity matrix and WikiArt label map"
    )
    parser.add_argument("--zoo_dir", type=str, default=DEFAULT_ZOO_DIR)
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR,
                        help="Directory containing style images.")
    parser.add_argument("--wikiart_dir", type=str, default=DEFAULT_WIKIART_DIR)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clip_model_id", type=str,
                        default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # ── 1. Discover styles ────────────────────────────────────
    zoo_path = Path(args.zoo_dir)
    style_names = sorted([
        d.name for d in zoo_path.iterdir()
        if d.is_dir() and (d / "pytorch_lora_weights.safetensors").exists()
    ])
    print(f"[Main] Found {len(style_names)} experts in {args.zoo_dir}")

    # ── 2. Compute CLIP embeddings ────────────────────────────
    image_dirs = [Path(args.image_dir)]
    embeddings = compute_clip_embeddings(
        style_names, image_dirs, args.clip_model_id, args.device
    )

    # ── 3. Similarity matrix ─────────────────────────────────
    S = compute_similarity_matrix(embeddings)
    print(f"\n[Similarity] Matrix shape: {S.shape}")
    print(f"  Diagonal mean (self-sim): {S.diag().mean():.4f}")
    print(f"  Off-diagonal mean: {(S.sum() - S.diag().sum()) / (S.numel() - S.shape[0]):.4f}")
    off_diag = S[~torch.eye(S.shape[0], dtype=bool)]
    print(f"  Off-diagonal min/max: {off_diag.min():.4f} / {off_diag.max():.4f}")

    # ── 4. Save similarity matrix ─────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sim_path = output_dir / "clip_similarity.pt"
    torch.save({
        "similarity_matrix": S,
        "embeddings": embeddings,
        "style_names": style_names,
    }, sim_path)
    print(f"  Saved to {sim_path}")

    # ── 5. WikiArt label map ──────────────────────────────────
    label_map = build_wikiart_label_map(style_names, args.wikiart_dir)
    map_path = output_dir / "wikiart_label_map.json"
    with open(map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map saved to {map_path}")

    # ── 6. Also save in data/ for easy import ─────────────────
    code_data_dir = Path(__file__).resolve().parent.parent / "data"
    code_map_path = code_data_dir / "wikiart_label_map.json"
    with open(code_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"  Label map also saved to {code_map_path}")

    print("\n[Done] Pre-computation complete.")


if __name__ == "__main__":
    main()
