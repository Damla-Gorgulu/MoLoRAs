"""
sanity_check.py — Phase 0: Image Generation Sanity Check

Tasks covered: 0.6–0.10 from TODO.md

This script:
  1. Generates Image 1: Base SDXL (no LoRA)
  2. Generates Image 2: Base + target LoRA via direct weight merge (α=2.0)
  3. Generates Image 3: Base + target LoRA via pipe.load_lora_weights()
  4. Compares Images 2 vs 3 (pixel MSE → injection equivalence)
  5. Saves all images and comparison metrics

Usage:
  python sanity_check.py
  python sanity_check.py --target-index 0
  python sanity_check.py --skip-api  # Skip pipe.load_lora_weights comparison
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from utils import (
    load_config,
    discover_lora_pool,
    load_lora_deltaw,
    get_adapter_key_order,
    flatten_deltaw,
    inject_deltaw,
    revert_injection,
    load_pipeline,
    generate_image,
    save_json,
    log_entry,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Sanity Check — Image Generation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--target-index", type=int, default=0,
                        help="Index of the target LoRA to test (default: 0)")
    parser.add_argument("--skip-api", action="store_true",
                        help="Skip pipe.load_lora_weights() comparison")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(config["experiment_dir"]) / "results" / "phase0" / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 0 — Sanity Check: Image Generation")
    print("=" * 60)

    # ── Discover pool & load target ──
    styles = discover_lora_pool(config["lora_pool_dir"])
    target = styles[args.target_index]
    print(f"  Target LoRA: [{target['index']}] {target['name']}")
    print(f"  Prompt: {config['prompt']}")
    print(f"  Seed: {config['seed']}")
    print(f"  Alpha: {config['lora_alpha']}")
    print()

    # Load ΔW for target
    deltaw = load_lora_deltaw(target["path"], dtype=torch.float32)
    key_order = get_adapter_key_order(deltaw)
    flat = flatten_deltaw(deltaw, key_order)
    print(f"  ΔW loaded: {len(key_order)} adapters, D={flat.shape[0]:,}")
    print(f"  ΔW overall norm: {torch.norm(flat).item():.4f}")
    print()

    # ── Load pipeline ──
    print("Loading SDXL pipeline...")
    pipe = load_pipeline(config, device="cuda")
    print("  Pipeline loaded.\n")

    # ── Image 1: Base model ──
    print("Generating Image 1 (base model, no LoRA)...")
    gen = torch.Generator(device="cuda").manual_seed(config["seed"])
    img_base = generate_image(pipe, config, generator=gen)
    img_base.save(out_dir / "p0_base.png")
    print(f"  Saved: {out_dir / 'p0_base.png'}\n")

    # ── Image 2: Direct weight merge ──
    print("Generating Image 2 (direct weight merge, α=2.0)...")
    originals = inject_deltaw(pipe, deltaw, alpha=config["lora_alpha"])
    gen = torch.Generator(device="cuda").manual_seed(config["seed"])
    img_merge = generate_image(pipe, config, generator=gen)
    img_merge.save(out_dir / "p0_target_merge.png")
    print(f"  Saved: {out_dir / 'p0_target_merge.png'}")
    revert_injection(pipe, originals)
    print("  Weights reverted.\n")

    # ── Image 3: API injection (optional) ──
    img_api = None
    mse_value = None
    if not args.skip_api:
        print("Generating Image 3 (pipe.load_lora_weights, scale=2.0)...")
        try:
            pipe.load_lora_weights(str(Path(target["path"]).parent))
            pipe.fuse_lora(lora_scale=config["lora_alpha"])
            gen = torch.Generator(device="cuda").manual_seed(config["seed"])
            img_api = generate_image(pipe, config, generator=gen)
            img_api.save(out_dir / "p0_target_api.png")
            print(f"  Saved: {out_dir / 'p0_target_api.png'}")
            pipe.unfuse_lora()
            pipe.unload_lora_weights()
            print("  LoRA unloaded.\n")

            # Compare Images 2 and 3
            arr_merge = np.array(img_merge, dtype=np.float32)
            arr_api = np.array(img_api, dtype=np.float32)
            mse_value = float(np.mean((arr_merge - arr_api) ** 2))
            print(f"  Pixel MSE (Image2 vs Image3): {mse_value:.6f}")
            if mse_value < 1.0:
                print("  ✓ Injection methods are equivalent (MSE < 1.0)")
            else:
                print("  ⚠ Injection methods differ — investigate alpha scaling")
        except Exception as e:
            print(f"  WARNING: API injection failed: {e}")
            print("  Continuing without API comparison.\n")

    # ── Visual difference check ──
    arr_base = np.array(img_base, dtype=np.float32)
    arr_merged = np.array(img_merge, dtype=np.float32)
    mse_base_vs_target = float(np.mean((arr_base - arr_merged) ** 2))
    print(f"\n  Pixel MSE (base vs target): {mse_base_vs_target:.2f}")
    if mse_base_vs_target > 10.0:
        print("  ✓ LoRA produces visible style change")
    else:
        print("  ⚠ LoRA effect may be too subtle — check alpha scaling")

    # ── Save results ──
    results = {
        "target_style": target["name"],
        "target_index": target["index"],
        "prompt": config["prompt"],
        "seed": config["seed"],
        "alpha": config["lora_alpha"],
        "num_adapters": len(key_order),
        "deltaw_dimensionality": flat.shape[0],
        "deltaw_overall_norm": torch.norm(flat).item(),
        "mse_base_vs_target": mse_base_vs_target,
        "mse_merge_vs_api": mse_value,
        "images": {
            "base": str(out_dir / "p0_base.png"),
            "target_merge": str(out_dir / "p0_target_merge.png"),
            "target_api": str(out_dir / "p0_target_api.png") if img_api else None,
        },
    }
    save_json(results, Path(config["experiment_dir"]) / "results" / "phase0" / "sanity_check_results.json")

    log_entry(config, {
        "phase": "0",
        "task": "0.6-0.10",
        "description": "Sanity check: image generation comparison",
        **results,
    })

    print("\n✓ Phase 0 sanity check complete.")


if __name__ == "__main__":
    main()
