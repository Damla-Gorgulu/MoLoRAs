"""
layerwise_reconstruction.py — Phase 2: Layer-wise Linear Reconstruction

Tasks covered: 2.1–2.14 from TODO.md

Scientific question:
  Does allowing different weights per tensor group improve reconstruction?
  ΔW_target[g] ≈ Σ_i w_i[g] * ΔW_donor_i[g]  for each group g independently

This script:
  1. Defines tensor grouping schemes (A, B, C + per-tensor)
  2. Extracts per-group sub-matrices
  3. Solves independent regression per group
  4. Reconstructs full ΔW and compares vs Phase 1 global
  5. Optionally generates comparison images

Usage:
  python layerwise_reconstruction.py                       # All grouping schemes
  python layerwise_reconstruction.py --scheme A            # Only scheme A
  python layerwise_reconstruction.py --per-tensor           # Per-tensor upper bound
  python layerwise_reconstruction.py --generate-images      # Also produce images
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from sklearn.linear_model import Ridge

from utils import (
    load_config,
    discover_lora_pool,
    load_lora_deltaw,
    get_adapter_key_order,
    get_adapter_shapes,
    flatten_deltaw,
    unflatten_deltaw,
    assign_tensor_groups,
    parse_tensor_key_structure,
    inject_deltaw,
    revert_injection,
    load_pipeline,
    generate_image,
    compute_metrics,
    relative_reconstruction_error,
    cosine_similarity,
    save_json,
    log_entry,
)


def load_matrix_and_meta(config):
    """Load precomputed matrix and metadata."""
    exp_dir = Path(config["experiment_dir"])
    matrix_path = exp_dir / "results" / "all_deltaw_matrix.pt"
    meta_path = exp_dir / "results" / "matrix_metadata.json"

    if not matrix_path.exists():
        print(f"ERROR: Matrix not found at {matrix_path}")
        print("Run extract_deltaw.py --build-matrix first.")
        sys.exit(1)

    matrix = torch.load(matrix_path, map_location="cpu")
    with open(meta_path) as f:
        meta = json.load(f)

    return matrix, meta


def load_target_selection(config):
    """Load target indices from Phase 1."""
    path = Path(config["experiment_dir"]) / "results" / "phase1" / "target_selection.json"
    if not path.exists():
        print(f"WARNING: Target selection not found at {path}")
        print("Using first 10 styles as targets.")
        return list(range(10))
    with open(path) as f:
        targets = json.load(f)
    return [t["index"] for t in targets]


def load_best_alpha(config):
    """Load best Ridge alpha from Phase 1 results."""
    path = Path(config["experiment_dir"]) / "results" / "phase1" / "best_methods.json"
    if not path.exists():
        print("  WARNING: No Phase 1 best_methods.json found. Using alpha=1.0")
        return 1.0
    with open(path) as f:
        best = json.load(f)
    # Most common best alpha
    from collections import Counter
    alphas = [r.get("alpha", 1.0) for r in best if r.get("method") == "ridge"]
    if alphas:
        return Counter(alphas).most_common(1)[0][0]
    return 1.0


def extract_group_subvectors(matrix_fp16, key_order, shapes, group_keys):
    """
    Extract sub-matrix corresponding to a group of tensor keys.

    Args:
        matrix_fp16: (D, N) tensor in fp16
        key_order: Full list of adapter key names
        shapes: Dict of adapter_name → (rows, cols)
        group_keys: List of adapter keys in this group

    Returns:
        sub_matrix: (D_group, N) numpy float64
        offset_info: list of (start, end) in the full flat vector
    """
    # Compute offsets for each adapter in the flat vector
    offsets = {}
    pos = 0
    for key in key_order:
        s = shapes[key]
        numel = s[0] * s[1]
        offsets[key] = (pos, pos + numel)
        pos += numel

    # Extract rows belonging to group keys
    group_indices = []
    for key in group_keys:
        start, end = offsets[key]
        group_indices.extend(range(start, end))

    group_indices = np.array(group_indices)
    sub_matrix = matrix_fp16[group_indices, :].numpy().astype(np.float32)
    return sub_matrix, group_indices


def run_groupwise_regression(matrix_fp16, key_order, shapes, group_def, target_indices,
                             styles, alpha, config):
    """
    Run per-group regression for a given grouping scheme.

    Returns:
        results: List of per-target result dicts
    """
    N = matrix_fp16.shape[1]
    groups = assign_tensor_groups(key_order, group_def)

    # Validate: every key should be in exactly one group
    all_assigned = set()
    for g_name, g_keys in groups.items():
        overlap = all_assigned & set(g_keys)
        if overlap:
            print(f"  WARNING: Keys in multiple groups: {overlap}")
        all_assigned.update(g_keys)
    unassigned = set(key_order) - all_assigned
    if unassigned:
        print(f"  WARNING: {len(unassigned)} keys not assigned to any group")

    print(f"  Groups: {list(groups.keys())}")
    for g_name, g_keys in groups.items():
        print(f"    {g_name}: {len(g_keys)} adapters")

    results = []

    for tidx in target_indices:
        print(f"\n  Target [{tidx}] {styles[tidx]['name']}:")
        per_group_results = {}
        reconstructed_indices = []
        reconstructed_values = []

        for g_name, g_keys in groups.items():
            if not g_keys:
                continue

            # Extract sub-matrix for this group
            sub_matrix, group_idx = extract_group_subvectors(
                matrix_fp16, key_order, shapes, g_keys
            )
            D_g = sub_matrix.shape[0]

            # Leave-one-out
            x_target_g = sub_matrix[:, tidx].copy()
            donor_cols = [i for i in range(N) if i != tidx]
            X_donors_g = sub_matrix[:, donor_cols].copy()

            # Solve
            t0 = time.time()
            model = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
            model.fit(X_donors_g, x_target_g)
            w = model.coef_
            elapsed = time.time() - t0

            x_recon_g = X_donors_g @ w

            # Metrics for this group
            x_t = torch.from_numpy(x_target_g.astype(np.float32))
            x_r = torch.from_numpy(x_recon_g.astype(np.float32))
            err = relative_reconstruction_error(x_t, x_r)
            cos = cosine_similarity(x_t, x_r)

            per_group_results[g_name] = {
                "relative_error": err,
                "cosine_similarity": cos,
                "D_group": D_g,
                "num_adapters": len(g_keys),
                "wall_time_seconds": elapsed,
            }

            # Store for full reconstruction (keep only small-ish recon vector)
            reconstructed_indices.append(group_idx)
            reconstructed_values.append(x_recon_g)

            # Free the large float32 arrays before next group iteration
            del sub_matrix, X_donors_g, x_target_g
            gc.collect()

            print(f"    {g_name}: error={err:.4f}, cos={cos:.4f}")

        # Combine all groups into full reconstruction
        D = matrix_fp16.shape[0]
        full_recon = np.zeros(D, dtype=np.float32)
        for idx_arr, val_arr in zip(reconstructed_indices, reconstructed_values):
            full_recon[idx_arr] = val_arr

        full_target = matrix_fp16[:, tidx].numpy().astype(np.float32)
        x_t_full = torch.from_numpy(full_target.astype(np.float32))
        x_r_full = torch.from_numpy(full_recon.astype(np.float32))

        overall_err = relative_reconstruction_error(x_t_full, x_r_full)
        overall_cos = cosine_similarity(x_t_full, x_r_full)

        print(f"    OVERALL: error={overall_err:.4f}, cos={overall_cos:.4f}")

        results.append({
            "target_index": tidx,
            "target_name": styles[tidx]["name"],
            "overall_relative_error": overall_err,
            "overall_cosine_similarity": overall_cos,
            "per_group": per_group_results,
        })

    return results


def run_per_tensor_regression(matrix_fp16, key_order, shapes, target_indices,
                              styles, alpha, config):
    """
    Task 2.11: Per-tensor regression — one independent regression per adapter.
    This is the upper bound on how well layer-wise decomposition can do.
    """
    print("\n" + "=" * 60)
    print("PER-TENSOR REGRESSION (upper bound)")
    print("=" * 60)

    N = matrix_fp16.shape[1]
    results = []

    for tidx in target_indices:
        print(f"\n  Target [{tidx}] {styles[tidx]['name']}:")
        per_tensor_results = {}
        all_recon = []
        all_target = []

        for key in key_order:
            sub_matrix, group_idx = extract_group_subvectors(
                matrix_fp16, key_order, shapes, [key]
            )

            x_target_t = sub_matrix[:, tidx].copy()
            donor_cols = [i for i in range(N) if i != tidx]
            X_donors_t = sub_matrix[:, donor_cols].copy()

            model = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
            model.fit(X_donors_t, x_target_t)
            x_recon_t = X_donors_t @ model.coef_

            x_t = torch.from_numpy(x_target_t.astype(np.float32))
            x_r = torch.from_numpy(x_recon_t.astype(np.float32))
            err = relative_reconstruction_error(x_t, x_r)
            cos = cosine_similarity(x_t, x_r)

            per_tensor_results[key] = {
                "relative_error": err,
                "cosine_similarity": cos,
            }

            all_recon.append(x_recon_t)
            all_target.append(x_target_t)

        full_recon = np.concatenate(all_recon)
        full_target = np.concatenate(all_target)
        x_t_full = torch.from_numpy(full_target.astype(np.float32))
        x_r_full = torch.from_numpy(full_recon.astype(np.float32))
        overall_err = relative_reconstruction_error(x_t_full, x_r_full)
        overall_cos = cosine_similarity(x_t_full, x_r_full)

        print(f"    Overall (per-tensor): error={overall_err:.4f}, cos={overall_cos:.4f}")

        results.append({
            "target_index": tidx,
            "target_name": styles[tidx]["name"],
            "overall_relative_error": overall_err,
            "overall_cosine_similarity": overall_cos,
            "per_tensor": per_tensor_results,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Layer-wise Reconstruction")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--scheme", type=str, default=None,
                        choices=["A", "B", "C"],
                        help="Run only one grouping scheme")
    parser.add_argument("--per-tensor", action="store_true",
                        help="Run per-tensor regression (upper bound)")
    parser.add_argument("--generate-images", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_dir = Path(config["experiment_dir"])

    print("=" * 60)
    print("Phase 2 — Layer-wise Linear Reconstruction")
    print("=" * 60)

    # Load matrix
    matrix, meta = load_matrix_and_meta(config)
    key_order = meta["key_order"]
    shapes = {k: tuple(v) for k, v in meta["shapes"].items()}
    D, N = matrix.shape
    print(f"  Matrix: ({D:,}, {N})")

    # Load targets and best alpha from Phase 1
    target_indices = load_target_selection(config)
    alpha = load_best_alpha(config)
    print(f"  Ridge alpha: {alpha}")
    print(f"  Targets: {target_indices}")

    styles = discover_lora_pool(config["lora_pool_dir"])

    # Load Phase 1 global results for comparison
    phase1_path = exp_dir / "results" / "phase1" / "best_methods.json"
    phase1_global = {}
    if phase1_path.exists():
        with open(phase1_path) as f:
            for r in json.load(f):
                phase1_global[r["target_index"]] = r

    grouping_config = config["phase2"]["grouping_schemes"]
    schemes_to_run = [args.scheme] if args.scheme else ["A", "B", "C"]

    all_scheme_results = {}

    for scheme_name in schemes_to_run:
        scheme = grouping_config[scheme_name]
        print(f"\n{'='*60}")
        print(f"GROUPING SCHEME {scheme_name}: {scheme['description']}")
        print(f"{'='*60}")

        group_def = scheme["groups"]
        results = run_groupwise_regression(
            matrix, key_order, shapes, group_def, target_indices, styles, alpha, config
        )
        all_scheme_results[scheme_name] = results

        # Save results
        save_json(results, exp_dir / "results" / "phase2" / f"group{scheme_name}_results.json")

        # Compare with Phase 1 global
        print(f"\n  Comparison with Phase 1 global:")
        print(f"  {'Target':40s} {'Global':>10s} {'Scheme '+scheme_name:>12s} {'Δ':>10s}")
        print(f"  {'-'*72}")
        for r in results:
            tidx = r["target_index"]
            p1_err = phase1_global.get(tidx, {}).get("relative_error", float("nan"))
            p2_err = r["overall_relative_error"]
            delta = p2_err - p1_err
            print(f"  {r['target_name']:40s} {p1_err:10.4f} {p2_err:12.4f} {delta:+10.4f}")

    # ── Per-tensor regression ──
    if args.per_tensor:
        per_tensor_results = run_per_tensor_regression(
            matrix, key_order, shapes, target_indices, styles, alpha, config
        )
        save_json(per_tensor_results, exp_dir / "results" / "phase2" / "per_tensor_results.json")

    # ── Generate images ──
    if args.generate_images:
        print("\n" + "=" * 60)
        print("Generating comparison images for best grouping scheme")
        print("=" * 60)

        # Find best scheme by average overall error
        best_scheme = min(
            all_scheme_results.keys(),
            key=lambda s: np.mean([r["overall_relative_error"] for r in all_scheme_results[s]])
        )
        print(f"  Best grouping scheme: {best_scheme}")

        pipe = load_pipeline(config, device="cuda")
        out_dir = exp_dir / "results" / "phase2" / "images"

        # Get best and worst target from best scheme
        scheme_results = all_scheme_results[best_scheme]
        sorted_results = sorted(scheme_results, key=lambda r: r["overall_relative_error"])
        picks = [("best", sorted_results[0]), ("worst", sorted_results[-1])]

        for label, result in picks:
            tidx = result["target_index"]
            print(f"\n  {label}: [{tidx}] {styles[tidx]['name']} (error={result['overall_relative_error']:.4f})")

            # Reconstruct using per-group regression
            # Re-run regression to get the actual reconstruction
            scheme_def = grouping_config[best_scheme]["groups"]
            groups = assign_tensor_groups(key_order, scheme_def)

            full_recon = np.zeros(D, dtype=np.float32)
            # matrix already passed as fp16; extract_group_subvectors casts slices internally
            donor_cols = [i for i in range(N) if i != tidx]

            for g_name, g_keys in groups.items():
                if not g_keys:
                    continue
                sub_matrix, group_idx = extract_group_subvectors(
                    matrix, key_order, shapes, g_keys
                )
                x_target_g = sub_matrix[:, tidx]
                X_donors_g = sub_matrix[:, donor_cols]
                model = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
                model.fit(X_donors_g, x_target_g)
                full_recon[group_idx] = X_donors_g @ model.coef_
                del sub_matrix, X_donors_g, x_target_g
                gc.collect()

            # Unflatten
            recon_flat = torch.from_numpy(full_recon.astype(np.float32))
            deltaw_recon = unflatten_deltaw(recon_flat, key_order, shapes)

            # Base image
            gen = torch.Generator(device="cuda").manual_seed(config["seed"])
            img_base = generate_image(pipe, config, generator=gen)
            img_base.save(out_dir / f"p2_{tidx:04d}_base.png")

            # Target image
            deltaw_target = load_lora_deltaw(styles[tidx]["path"], dtype=torch.float32)
            originals = inject_deltaw(pipe, deltaw_target, alpha=config["lora_alpha"])
            gen = torch.Generator(device="cuda").manual_seed(config["seed"])
            img_target = generate_image(pipe, config, generator=gen)
            img_target.save(out_dir / f"p2_{tidx:04d}_target.png")
            revert_injection(pipe, originals)

            # Reconstructed image
            originals = inject_deltaw(pipe, deltaw_recon, alpha=config["lora_alpha"])
            gen = torch.Generator(device="cuda").manual_seed(config["seed"])
            img_recon = generate_image(pipe, config, generator=gen)
            img_recon.save(out_dir / f"p2_{tidx:04d}_reconstructed.png")
            revert_injection(pipe, originals)

        del pipe
        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    for scheme_name, results in all_scheme_results.items():
        errors = [r["overall_relative_error"] for r in results]
        print(f"  Scheme {scheme_name}: mean_error={np.mean(errors):.4f}, "
              f"std={np.std(errors):.4f}, min={min(errors):.4f}, max={max(errors):.4f}")

    log_entry(config, {
        "phase": "2",
        "task": "2.1-2.14",
        "description": "Layer-wise reconstruction complete",
        "schemes_run": list(all_scheme_results.keys()),
        "per_tensor_run": args.per_tensor,
    })

    print("\n✓ Phase 2 complete.")


if __name__ == "__main__":
    main()
