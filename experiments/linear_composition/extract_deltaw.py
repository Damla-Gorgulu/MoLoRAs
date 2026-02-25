"""
extract_deltaw.py — Phase 0: ΔW Extraction & Validation

Tasks covered: 0.1–0.5, I.4 from TODO.md

This script:
  1. Discovers all 109 LoRAs in the pool
  2. Validates tensor key structure
  3. Computes ΔW = B @ A for a test LoRA
  4. Logs norms, shapes, dimensionality
  5. Builds the full (D × 109) matrix for Phase 1+
  6. Saves style_index.json, tensor_key_map.json, tensor_norms.json

Usage:
  python extract_deltaw.py                          # Full extraction
  python extract_deltaw.py --validate-only           # Just validate one LoRA
  python extract_deltaw.py --build-matrix             # Build full D×109 matrix
"""

import argparse
import sys
import time
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np

from utils import (
    load_config,
    discover_lora_pool,
    load_lora_deltaw,
    get_adapter_key_order,
    get_adapter_shapes,
    flatten_deltaw,
    parse_tensor_key_structure,
    save_json,
    log_entry,
)


def task_01_discover_pool(config):
    """Task 0.1: List all LoRA directories, assign indices, save style_index.json."""
    print("=" * 60)
    print("TASK 0.1 — Discover LoRA pool")
    print("=" * 60)

    pool_dir = config["lora_pool_dir"]
    print(f"  Pool directory: {pool_dir}")

    styles = discover_lora_pool(pool_dir)
    print(f"  Found {len(styles)} styles with safetensors files")

    if len(styles) == 0:
        print("  ERROR: No styles found! Check pool_dir path and permissions.")
        sys.exit(1)

    # Show first/last few
    for s in styles[:3]:
        print(f"    [{s['index']:3d}] {s['name']}")
    print(f"    ...")
    for s in styles[-3:]:
        print(f"    [{s['index']:3d}] {s['name']}")

    out_path = Path(config["experiment_dir"]) / "results" / "phase0" / "style_index.json"
    save_json(styles, out_path)

    log_entry(config, {
        "phase": "0",
        "task": "0.1",
        "description": "Discover LoRA pool",
        "num_styles": len(styles),
        "pool_dir": pool_dir,
    })

    return styles


def task_02_03_validate_keys(config, styles):
    """
    Tasks 0.2 & 0.3: Load one LoRA, validate key structure, save tensor_key_map.json.
    """
    print("\n" + "=" * 60)
    print("TASKS 0.2–0.3 — Validate tensor key structure")
    print("=" * 60)

    test_style = styles[0]
    print(f"  Test LoRA: {test_style['name']}")
    print(f"  Path: {test_style['path']}")

    deltaw = load_lora_deltaw(test_style["path"], dtype=torch.float32)
    key_order = get_adapter_key_order(deltaw)

    print(f"  Number of adapter pairs: {len(key_order)}")

    # Parse each key into structured info
    key_map = []
    for key in key_order:
        info = parse_tensor_key_structure(key)
        info["adapter_name"] = key
        info["shape"] = list(deltaw[key].shape)
        key_map.append(info)

    # Validate: should be 10 blocks × 2 attn × 4 proj = 80
    expected = 80
    if len(key_map) != expected:
        print(f"  WARNING: Expected {expected} adapters, got {len(key_map)}")
        print(f"  This may be okay if the LoRA has a different structure.")
    else:
        print(f"  ✓ Exactly {expected} adapter pairs confirmed")

    # Print sample keys
    print("\n  Sample adapter keys:")
    for entry in key_map[:5]:
        print(f"    {entry['adapter_name']} → block={entry.get('block_idx')}, "
              f"attn={entry.get('attn_type')}, proj={entry.get('projection')}, "
              f"shape={entry['shape']}")

    out_path = Path(config["experiment_dir"]) / "results" / "phase0" / "tensor_key_map.json"
    save_json(key_map, out_path)

    log_entry(config, {
        "phase": "0",
        "task": "0.2-0.3",
        "description": "Validate tensor key structure",
        "num_adapters": len(key_map),
        "test_style": test_style["name"],
        "all_keys": [k["adapter_name"] for k in key_map],
    })

    return deltaw, key_order, key_map


def task_04_05_compute_norms(config, deltaw, key_order):
    """
    Tasks 0.4 & 0.5: Compute ΔW norms and total dimensionality D.
    """
    print("\n" + "=" * 60)
    print("TASKS 0.4–0.5 — Compute ΔW norms and dimensionality")
    print("=" * 60)

    norms = []
    total_params = 0
    all_nonzero = True

    for key in key_order:
        dw = deltaw[key]
        frob_norm = torch.norm(dw).item()
        numel = dw.numel()
        total_params += numel
        norms.append({
            "adapter_name": key,
            "frobenius_norm": frob_norm,
            "shape": list(dw.shape),
            "num_params": numel,
        })
        if frob_norm < 1e-12:
            all_nonzero = False
            print(f"  WARNING: Zero norm for {key}")

    print(f"  Total dimensionality D = {total_params:,}")
    print(f"  All norms nonzero: {'✓' if all_nonzero else '✗ FAILURE'}")

    norm_values = [n["frobenius_norm"] for n in norms]
    print(f"  Norm stats: min={min(norm_values):.6f}, max={max(norm_values):.6f}, "
          f"mean={np.mean(norm_values):.6f}, std={np.std(norm_values):.6f}")

    # Flatten and verify D
    flat = flatten_deltaw(deltaw, key_order)
    assert flat.shape[0] == total_params, f"Flatten mismatch: {flat.shape[0]} vs {total_params}"
    print(f"  Flattened vector shape: {flat.shape}")

    out_path = Path(config["experiment_dir"]) / "results" / "phase0" / "tensor_norms.json"
    save_json({
        "total_dimensionality": total_params,
        "num_adapters": len(norms),
        "all_nonzero": all_nonzero,
        "norm_stats": {
            "min": float(min(norm_values)),
            "max": float(max(norm_values)),
            "mean": float(np.mean(norm_values)),
            "std": float(np.std(norm_values)),
        },
        "per_adapter_norms": norms,
    }, out_path)

    log_entry(config, {
        "phase": "0",
        "task": "0.4-0.5",
        "description": "Compute ΔW norms and dimensionality",
        "total_dimensionality": total_params,
        "all_nonzero": all_nonzero,
    })

    return total_params


def build_full_matrix(config, styles, key_order):
    """
    Task 1.1 (prep): Build the full (D × N) matrix of all LoRA ΔW vectors.

    Uses streaming approach: load one LoRA at a time, flatten, store as column.
    Saves in fp16 to conserve disk/memory.
    """
    print("\n" + "=" * 60)
    print("BUILD FULL MATRIX — All LoRA ΔW vectors")
    print("=" * 60)

    N = len(styles)
    print(f"  Number of styles: {N}")

    # Load first LoRA to determine D
    deltaw_0 = load_lora_deltaw(styles[0]["path"], dtype=torch.float32)
    flat_0 = flatten_deltaw(deltaw_0, key_order)
    D = flat_0.shape[0]
    print(f"  Dimensionality D = {D:,}")
    print(f"  Matrix size: ({D}, {N})")
    mem_gb = D * N * 2 / (1024**3)  # fp16
    print(f"  Estimated storage (fp16): {mem_gb:.2f} GB")

    # Also save shapes for unflatten later
    shapes = get_adapter_shapes(deltaw_0)

    # Allocate matrix in fp16
    matrix = torch.zeros(D, N, dtype=torch.float16)
    matrix[:, 0] = flat_0.to(torch.float16)

    t0 = time.time()
    for i in range(1, N):
        if i % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / i * (N - i)
            print(f"  Loading LoRA {i}/{N} ... elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

        deltaw_i = load_lora_deltaw(styles[i]["path"], dtype=torch.float32)
        flat_i = flatten_deltaw(deltaw_i, key_order)
        assert flat_i.shape[0] == D, f"Shape mismatch at style {i}: {flat_i.shape[0]} vs {D}"
        matrix[:, i] = flat_i.to(torch.float16)

    elapsed = time.time() - t0
    print(f"  All {N} LoRAs loaded in {elapsed:.1f}s")

    # Save matrix
    out_dir = Path(config["experiment_dir"]) / "results"
    matrix_path = out_dir / "all_deltaw_matrix.pt"
    torch.save(matrix, matrix_path)
    print(f"  Saved matrix: {matrix_path} ({matrix_path.stat().st_size / 1e9:.2f} GB)")

    # Save metadata
    meta = {
        "D": D,
        "N": N,
        "dtype": "float16",
        "key_order": key_order,
        "shapes": {k: list(v) for k, v in shapes.items()},
        "style_names": [s["name"] for s in styles],
        "build_time_seconds": elapsed,
    }
    save_json(meta, out_dir / "matrix_metadata.json")

    log_entry(config, {
        "phase": "0+1",
        "task": "build_matrix",
        "description": "Built full D×N ΔW matrix",
        "D": D,
        "N": N,
        "build_time_seconds": elapsed,
        "matrix_path": str(matrix_path),
    })

    return matrix, meta


def main():
    parser = argparse.ArgumentParser(description="Phase 0: ΔW Extraction & Validation")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate one LoRA (tasks 0.1–0.5)")
    parser.add_argument("--build-matrix", action="store_true",
                        help="Also build the full D×109 matrix (task 1.1)")
    args = parser.parse_args()

    config = load_config(args.config)
    print("Linear Composition Experiment — Phase 0: ΔW Extraction")
    print(f"Experiment dir: {config['experiment_dir']}")
    print()

    # Task 0.1: Discover pool
    styles = task_01_discover_pool(config)

    # Tasks 0.2–0.3: Validate key structure
    deltaw, key_order, key_map = task_02_03_validate_keys(config, styles)

    # Tasks 0.4–0.5: Norms and dimensionality
    total_params = task_04_05_compute_norms(config, deltaw, key_order)

    if args.validate_only:
        print("\n✓ Validation complete. Run with --build-matrix to build the full matrix.")
        return

    if args.build_matrix:
        matrix, meta = build_full_matrix(config, styles, key_order)
        print(f"\n✓ Matrix built: {meta['D']:,} × {meta['N']}")

    print("\n✓ Phase 0 extraction complete.")


if __name__ == "__main__":
    main()
