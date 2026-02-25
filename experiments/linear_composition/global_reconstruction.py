"""
global_reconstruction.py — Phase 1: Global Linear Reconstruction

Tasks covered: 1.1–1.13 from TODO.md

Scientific question:
  Can ΔW_target ≈ Σ_i w_i * ΔW_donor_i  using ONE scalar per donor?

This script:
  1. Loads the precomputed (D × N) matrix from Phase 0
  2. Runs leave-one-out regression with Ridge, Lasso, ElasticNet
  3. Computes reconstruction error, cosine sim, sparsity
  4. Optionally generates comparison images for best/median/worst targets
  5. Runs normalization ablation

Usage:
  python global_reconstruction.py                       # Run regression only
  python global_reconstruction.py --generate-images     # Also generate comparison images
  python global_reconstruction.py --self-check          # Self-reconstruction sanity check
  python global_reconstruction.py --normalize           # Normalization ablation
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet

from utils import (
    load_config,
    discover_lora_pool,
    load_lora_deltaw,
    get_adapter_key_order,
    flatten_deltaw,
    unflatten_deltaw,
    inject_deltaw,
    revert_injection,
    load_pipeline,
    generate_image,
    compute_metrics,
    save_json,
    log_entry,
)


# ================================================================
# Regression helpers
# ================================================================

def solve_regression(X_donors, x_target, method="ridge", alpha=1.0, l1_ratio=0.5):
    """
    Solve: x_target ≈ X_donors @ w

    Uses the Gram matrix approach (normal equations) when D >> N for efficiency:
      X^T X w = X^T x_target
    scikit-learn handles this automatically with the 'auto' solver.

    Args:
        X_donors: (D, K) numpy array — donor matrix (K = num donors)
        x_target: (D,) numpy array — target vector
        method: "ridge", "lasso", or "elasticnet"
        alpha: Regularization strength
        l1_ratio: L1/L2 ratio for ElasticNet

    Returns:
        w: (K,) coefficient vector
        model: fitted sklearn model
    """
    # Transpose: sklearn expects (n_samples, n_features) = (D, K)
    # We want to find w s.t. X @ w ≈ x_target
    # Reshape: X is (D, K), x_target is (D,)
    # sklearn.Ridge.fit(X, y) solves min ||y - X @ w||^2 + alpha ||w||^2

    if method == "ridge":
        model = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    elif method == "lasso":
        model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    elif method == "elasticnet":
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=10000)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.fit(X_donors, x_target)
    w = model.coef_
    return w, model


def run_regression_sweep(matrix_np, target_idx, styles, config, normalize=False):
    """
    Run all regression methods/hyperparameters for one target.

    Args:
        matrix_np: (D, N) numpy array
        target_idx: Index of target style to reconstruct
        styles: List of style dicts
        config: Experiment config
        normalize: If True, normalize each column to unit norm

    Returns:
        List of result dicts
    """
    N = matrix_np.shape[1]
    D = matrix_np.shape[0]

    # Extract target and build donor matrix
    x_target = matrix_np[:, target_idx].copy()
    donor_indices = [i for i in range(N) if i != target_idx]
    X_donors = matrix_np[:, donor_indices].copy()

    # Optional normalization
    norms_donors = None
    norm_target = None
    if normalize:
        norms_donors = np.linalg.norm(X_donors, axis=0, keepdims=True)
        norms_donors = np.maximum(norms_donors, 1e-12)
        X_donors = X_donors / norms_donors
        norm_target = np.linalg.norm(x_target)
        if norm_target > 1e-12:
            x_target = x_target / norm_target

    results = []
    cfg1 = config["phase1"]

    # Ridge
    for alpha in cfg1["ridge_alphas"]:
        t0 = time.time()
        w, _ = solve_regression(X_donors, x_target, "ridge", alpha=alpha)
        elapsed = time.time() - t0

        x_recon = X_donors @ w
        if normalize and norm_target is not None:
            x_recon_orig = x_recon * norm_target
            x_target_orig = matrix_np[:, target_idx]
        else:
            x_recon_orig = x_recon
            x_target_orig = matrix_np[:, target_idx]

        x_t = torch.from_numpy(x_target_orig.astype(np.float32))
        x_r = torch.from_numpy(x_recon_orig.astype(np.float32))
        metrics = compute_metrics(x_t, x_r, w, cfg1["sparsity_threshold"])
        results.append({
            "method": "ridge",
            "alpha": alpha,
            "normalized": normalize,
            "target_index": target_idx,
            "target_name": styles[target_idx]["name"],
            "wall_time_seconds": elapsed,
            **metrics,
            "coefficients": w.tolist(),
        })

    # Lasso
    for alpha in cfg1["lasso_alphas"]:
        t0 = time.time()
        w, _ = solve_regression(X_donors, x_target, "lasso", alpha=alpha)
        elapsed = time.time() - t0

        x_recon = X_donors @ w
        if normalize and norm_target is not None:
            x_recon_orig = x_recon * norm_target
            x_target_orig = matrix_np[:, target_idx]
        else:
            x_recon_orig = x_recon
            x_target_orig = matrix_np[:, target_idx]

        x_t = torch.from_numpy(x_target_orig.astype(np.float32))
        x_r = torch.from_numpy(x_recon_orig.astype(np.float32))
        metrics = compute_metrics(x_t, x_r, w, cfg1["sparsity_threshold"])
        results.append({
            "method": "lasso",
            "alpha": alpha,
            "normalized": normalize,
            "target_index": target_idx,
            "target_name": styles[target_idx]["name"],
            "wall_time_seconds": elapsed,
            **metrics,
            "coefficients": w.tolist(),
        })

    # ElasticNet
    for alpha in cfg1["elasticnet_alphas"]:
        for l1r in cfg1["elasticnet_l1_ratios"]:
            t0 = time.time()
            w, _ = solve_regression(X_donors, x_target, "elasticnet",
                                    alpha=alpha, l1_ratio=l1r)
            elapsed = time.time() - t0

            x_recon = X_donors @ w
            if normalize and norm_target is not None:
                x_recon_orig = x_recon * norm_target
                x_target_orig = matrix_np[:, target_idx]
            else:
                x_recon_orig = x_recon
                x_target_orig = matrix_np[:, target_idx]

            x_t = torch.from_numpy(x_target_orig.astype(np.float32))
            x_r = torch.from_numpy(x_recon_orig.astype(np.float32))
            metrics = compute_metrics(x_t, x_r, w, cfg1["sparsity_threshold"])
            results.append({
                "method": "elasticnet",
                "alpha": alpha,
                "l1_ratio": l1r,
                "normalized": normalize,
                "target_index": target_idx,
                "target_name": styles[target_idx]["name"],
                "wall_time_seconds": elapsed,
                **metrics,
                "coefficients": w.tolist(),
            })

    return results


def self_reconstruction_check(matrix_np, styles, config):
    """
    Task 1.5: Self-reconstruction — include target in donor pool.
    Coefficient for target should be ~1.0, others ~0.
    """
    print("\n" + "=" * 60)
    print("TASK 1.5 — Self-reconstruction sanity check")
    print("=" * 60)

    target_idx = 0
    x_target = matrix_np[:, target_idx].copy()
    X_all = matrix_np.copy()  # Include target in donors

    w, _ = solve_regression(X_all, x_target, "ridge", alpha=0.01)
    x_recon = X_all @ w

    x_t = torch.from_numpy(x_target.astype(np.float32))
    x_r = torch.from_numpy(x_recon.astype(np.float32))
    metrics = compute_metrics(x_t, x_r, w)

    print(f"  Target coefficient: {w[target_idx]:.6f} (expected ~1.0)")
    print(f"  Max other coeff:    {np.max(np.abs(np.delete(w, target_idx))):.6f} (expected ~0)")
    print(f"  Relative error:     {metrics['relative_error']:.6f} (expected ~0)")
    print(f"  Cosine similarity:  {metrics['cosine_similarity']:.6f} (expected ~1)")

    ok = (
        abs(w[target_idx] - 1.0) < 0.05
        and metrics["relative_error"] < 0.01
    )
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")

    result = {
        "target_coefficient": float(w[target_idx]),
        "max_other_coefficient": float(np.max(np.abs(np.delete(w, target_idx)))),
        **metrics,
        "pass": ok,
    }
    save_json(result, Path(config["experiment_dir"]) / "results" / "phase1" / "self_check.json")
    return ok


def select_representative_targets(styles, config):
    """
    Task 1.4: Select diverse target styles.
    Uses name-based heuristic to pick from different art movements.
    Falls back to evenly spaced indices if not enough variety.
    """
    N = len(styles)
    k = config["phase1"]["num_representative_targets"]

    # Try to pick diverse movements
    movement_keywords = [
        "Baroque", "Realism", "Impressionism", "Cubism",
        "Expressionism", "Romanticism", "Abstract", "Renaissance",
        "Symbolism", "Post_Impressionism", "Fauvism", "Surrealism",
        "Minimalism", "Art_Nouveau", "Pop_Art", "Color_Field",
    ]

    selected = []
    used = set()
    for kw in movement_keywords:
        if len(selected) >= k:
            break
        for s in styles:
            if kw.lower() in s["name"].lower() and s["index"] not in used:
                selected.append(s["index"])
                used.add(s["index"])
                break

    # Fill remaining with evenly-spaced indices
    if len(selected) < k:
        step = N // (k - len(selected) + 1)
        for i in range(0, N, step):
            if i not in used and len(selected) < k:
                selected.append(i)
                used.add(i)

    selected = sorted(selected[:k])
    print(f"  Selected {len(selected)} representative targets:")
    for idx in selected:
        print(f"    [{idx:3d}] {styles[idx]['name']}")

    return selected


def generate_comparison_images(matrix_np, all_results, styles, config, meta):
    """
    Tasks 1.11–1.12: Reconstruct ΔW and generate comparison images.
    """
    print("\n" + "=" * 60)
    print("TASKS 1.11–1.12 — Generate comparison images")
    print("=" * 60)

    # Find best method per target (min relative error)
    from collections import defaultdict
    best_per_target = {}
    for r in all_results:
        if r.get("normalized", False):
            continue
        tidx = r["target_index"]
        if tidx not in best_per_target or r["relative_error"] < best_per_target[tidx]["relative_error"]:
            best_per_target[tidx] = r

    # Sort targets by error: pick best, median, worst
    sorted_targets = sorted(best_per_target.items(), key=lambda x: x[1]["relative_error"])
    if len(sorted_targets) < 3:
        print("  Not enough targets for best/median/worst selection")
        return

    picks = [
        ("best", sorted_targets[0]),
        ("median", sorted_targets[len(sorted_targets) // 2]),
        ("worst", sorted_targets[-1]),
    ]

    # Load pipeline
    pipe = load_pipeline(config, device="cuda")
    key_order = meta["key_order"]
    shapes = {k: tuple(v) for k, v in meta["shapes"].items()}
    out_dir = Path(config["experiment_dir"]) / "results" / "phase1" / "images"

    for label, (tidx, result) in picks:
        print(f"\n  Generating images for {label} target: [{tidx}] {styles[tidx]['name']}")
        print(f"    Relative error: {result['relative_error']:.4f}")
        print(f"    Cosine sim:     {result['cosine_similarity']:.4f}")

        # Reconstruct ΔW from coefficients
        w = np.array(result["coefficients"])
        donor_indices = [i for i in range(matrix_np.shape[1]) if i != tidx]
        X_donors = matrix_np[:, donor_indices]
        x_recon = X_donors @ w

        # Unflatten
        recon_flat = torch.from_numpy(x_recon.astype(np.float32))
        deltaw_recon = unflatten_deltaw(recon_flat, key_order, shapes)

        # Image: base
        gen = torch.Generator(device="cuda").manual_seed(config["seed"])
        img_base = generate_image(pipe, config, generator=gen)
        img_base.save(out_dir / f"p1_{tidx:04d}_base.png")

        # Image: target LoRA
        deltaw_target = load_lora_deltaw(styles[tidx]["path"], dtype=torch.float32)
        originals = inject_deltaw(pipe, deltaw_target, alpha=config["lora_alpha"])
        gen = torch.Generator(device="cuda").manual_seed(config["seed"])
        img_target = generate_image(pipe, config, generator=gen)
        img_target.save(out_dir / f"p1_{tidx:04d}_target.png")
        revert_injection(pipe, originals)

        # Image: reconstructed LoRA
        originals = inject_deltaw(pipe, deltaw_recon, alpha=config["lora_alpha"])
        gen = torch.Generator(device="cuda").manual_seed(config["seed"])
        img_recon = generate_image(pipe, config, generator=gen)
        img_recon.save(out_dir / f"p1_{tidx:04d}_reconstructed.png")
        revert_injection(pipe, originals)

        print(f"    Saved: p1_{tidx:04d}_base/target/reconstructed.png")

    del pipe
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Global Linear Reconstruction")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--self-check", action="store_true",
                        help="Run self-reconstruction sanity check only")
    parser.add_argument("--generate-images", action="store_true",
                        help="Generate comparison images for best/median/worst")
    parser.add_argument("--normalize", action="store_true",
                        help="Run normalization ablation")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_dir = Path(config["experiment_dir"])

    print("=" * 60)
    print("Phase 1 — Global Linear Reconstruction")
    print("=" * 60)

    # Load precomputed matrix
    matrix_path = exp_dir / "results" / "all_deltaw_matrix.pt"
    meta_path = exp_dir / "results" / "matrix_metadata.json"

    if not matrix_path.exists():
        print(f"ERROR: Matrix not found at {matrix_path}")
        print("Run extract_deltaw.py --build-matrix first.")
        sys.exit(1)

    print(f"Loading matrix from {matrix_path}...")
    matrix = torch.load(matrix_path, map_location="cpu")
    with open(meta_path) as f:
        meta = json.load(f)

    D, N = matrix.shape
    print(f"  Matrix shape: ({D:,}, {N})")

    # Convert to numpy float64 for regression
    print("  Converting to float64 for regression...")
    matrix_np = matrix.numpy().astype(np.float64)
    del matrix

    # Load styles
    styles = discover_lora_pool(config["lora_pool_dir"])

    # ── Self-check ──
    if args.self_check:
        self_reconstruction_check(matrix_np, styles, config)
        return

    # ── Select targets ──
    print("\nSelecting representative targets...")
    target_indices = select_representative_targets(styles, config)
    save_json(
        [{"index": i, "name": styles[i]["name"]} for i in target_indices],
        exp_dir / "results" / "phase1" / "target_selection.json"
    )

    # ── Run self-check first ──
    self_reconstruction_check(matrix_np, styles, config)

    # ── Run regression sweep ──
    all_results = []
    for i, tidx in enumerate(target_indices):
        print(f"\n{'='*60}")
        print(f"TARGET {i+1}/{len(target_indices)}: [{tidx}] {styles[tidx]['name']}")
        print(f"{'='*60}")

        results = run_regression_sweep(matrix_np, tidx, styles, config, normalize=False)
        all_results.extend(results)

        # Print best result for this target
        best = min(results, key=lambda r: r["relative_error"])
        print(f"  Best: {best['method']}(α={best.get('alpha')}) → "
              f"error={best['relative_error']:.4f}, cos={best['cosine_similarity']:.4f}")

    # Save results (without coefficients for the main JSON — too large)
    results_summary = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "coefficients"}
        results_summary.append(r_copy)

    # Save per-method results
    for method in ["ridge", "lasso", "elasticnet"]:
        method_results = [r for r in results_summary if r["method"] == method]
        save_json(method_results, exp_dir / "results" / "phase1" / f"{method}_results.json")

    # Save coefficients separately
    for r in all_results:
        tidx = r["target_index"]
        method = r["method"]
        alpha = r.get("alpha", 0)
        l1r = r.get("l1_ratio", "")
        fname = f"coeffs_{method}_a{alpha}_l1r{l1r}_{tidx:04d}.npy"
        np.save(exp_dir / "results" / "phase1" / "coefficients" / fname,
                np.array(r["coefficients"]))

    # Find best method per target
    best_methods = {}
    for r in all_results:
        tidx = r["target_index"]
        if tidx not in best_methods or r["relative_error"] < best_methods[tidx]["relative_error"]:
            best_methods[tidx] = {k: v for k, v in r.items() if k != "coefficients"}

    save_json(list(best_methods.values()),
              exp_dir / "results" / "phase1" / "best_methods.json")

    # ── Normalization ablation ──
    if args.normalize or config["phase1"].get("normalization_ablation", False):
        print("\n" + "=" * 60)
        print("NORMALIZATION ABLATION")
        print("=" * 60)

        norm_results = []
        for tidx in target_indices:
            results = run_regression_sweep(matrix_np, tidx, styles, config, normalize=True)
            norm_results.extend(results)

        norm_summary = [{k: v for k, v in r.items() if k != "coefficients"} for r in norm_results]
        save_json(norm_summary, exp_dir / "results" / "phase1" / "normalized_results.json")

    # ── Generate images ──
    if args.generate_images:
        generate_comparison_images(matrix_np, all_results, styles, config, meta)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    for tidx, best in sorted(best_methods.items()):
        print(f"  [{tidx:3d}] {best['target_name']:40s} → "
              f"error={best['relative_error']:.4f}  cos={best['cosine_similarity']:.4f}  "
              f"method={best['method']}(α={best.get('alpha')})")

    errors = [b["relative_error"] for b in best_methods.values()]
    print(f"\n  Error stats: min={min(errors):.4f}, max={max(errors):.4f}, "
          f"mean={np.mean(errors):.4f}, std={np.std(errors):.4f}")

    log_entry(config, {
        "phase": "1",
        "task": "1.6-1.13",
        "description": "Global linear reconstruction complete",
        "num_targets": len(target_indices),
        "error_mean": float(np.mean(errors)),
        "error_std": float(np.std(errors)),
        "error_min": float(min(errors)),
        "error_max": float(max(errors)),
    })

    print("\n✓ Phase 1 complete.")


if __name__ == "__main__":
    main()
