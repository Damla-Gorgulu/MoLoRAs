"""
span_analysis.py — Phase 3: Span Membership Interpretation

Tasks covered: 3.1–3.9 from TODO.md

This script:
  1. Runs leave-one-out reconstruction for ALL 109 styles
  2. Computes distribution statistics
  3. Runs baseline comparisons (random donors, random tensors)
  4. Performs sparsity analysis
  5. Computes SVD spectrum for subspace dimensionality
  6. Classifies each style by span membership
  7. Produces summary plots

Usage:
  python span_analysis.py                    # Full Phase 3
  python span_analysis.py --full-sweep       # All 109 leave-one-out (Task 3.1)
  python span_analysis.py --baselines        # Random donor + random tensor baselines
  python span_analysis.py --svd              # SVD spectrum analysis
  python span_analysis.py --plots            # Generate all plots
"""

import argparse
import json
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    load_config,
    discover_lora_pool,
    relative_reconstruction_error,
    cosine_similarity,
    compute_metrics,
    save_json,
    log_entry,
)


def load_matrix_and_meta(config):
    """Load precomputed matrix and metadata."""
    exp_dir = Path(config["experiment_dir"])
    matrix = torch.load(exp_dir / "results" / "all_deltaw_matrix.pt", map_location="cpu")
    with open(exp_dir / "results" / "matrix_metadata.json") as f:
        meta = json.load(f)
    return matrix, meta


def load_best_method(config):
    """
    Load the best method/alpha from Phase 1 results.
    Returns (method_name, alpha, l1_ratio_or_None).
    """
    path = Path(config["experiment_dir"]) / "results" / "phase1" / "best_methods.json"
    if not path.exists():
        print("  WARNING: No Phase 1 results. Using Ridge alpha=1.0")
        return "ridge", 1.0, None

    with open(path) as f:
        results = json.load(f)

    # Most common best method
    methods = [(r.get("method", "ridge"), r.get("alpha", 1.0)) for r in results]
    (method, alpha), _ = Counter(methods).most_common(1)[0]
    return method, alpha, None


def solve_ridge(X_donors, x_target, alpha):
    """Quick Ridge regression helper."""
    model = Ridge(alpha=alpha, fit_intercept=False, solver="auto")
    model.fit(X_donors, x_target)
    return model.coef_


# ================================================================
# Task 3.1: Full leave-one-out sweep
# ================================================================

def full_leave_one_out(matrix_fp16, styles, config, method, alpha):
    """Run reconstruction for ALL N styles."""
    print("=" * 60)
    print("TASK 3.1 — Full leave-one-out sweep (all styles)")
    print("=" * 60)

    N = matrix_fp16.shape[1]
    results = []

    t_total = time.time()
    for tidx in range(N):
        t0 = time.time()

        # Cast per-iteration: ~186 GB peak (fp16 matrix + fp32 donors slice)
        x_target = matrix_fp16[:, tidx].float().numpy()
        donor_cols = [i for i in range(N) if i != tidx]
        X_donors = matrix_fp16[:, donor_cols].float().numpy()

        w = solve_ridge(X_donors, x_target, alpha)
        x_recon = X_donors @ w
        del X_donors  # free immediately
        elapsed = time.time() - t0

        x_t = torch.from_numpy(x_target)
        x_r = torch.from_numpy(x_recon.astype(np.float32))

        metrics = compute_metrics(x_t, x_r, w)
        results.append({
            "target_index": tidx,
            "target_name": styles[tidx]["name"],
            "method": method,
            "alpha": alpha,
            "wall_time_seconds": elapsed,
            **metrics,
            "coefficients": w.tolist(),
        })

        if (tidx + 1) % 10 == 0 or tidx == N - 1:
            elapsed_total = time.time() - t_total
            eta = elapsed_total / (tidx + 1) * (N - tidx - 1)
            print(f"  [{tidx+1:3d}/{N}] {styles[tidx]['name']:40s} "
                  f"error={metrics['relative_error']:.4f}  "
                  f"cos={metrics['cosine_similarity']:.4f}  "
                  f"ETA={eta:.0f}s")

    print(f"\n  Total time: {time.time() - t_total:.1f}s")
    return results


# ================================================================
# Task 3.2: Distribution statistics
# ================================================================

def compute_distribution_stats(results):
    """Compute statistics over all targets."""
    errors = [r["relative_error"] for r in results]
    cosines = [r["cosine_similarity"] for r in results]

    stats = {
        "reconstruction_error": {
            "mean": float(np.mean(errors)),
            "std": float(np.std(errors)),
            "min": float(np.min(errors)),
            "max": float(np.max(errors)),
            "q25": float(np.percentile(errors, 25)),
            "q50": float(np.percentile(errors, 50)),
            "q75": float(np.percentile(errors, 75)),
        },
        "cosine_similarity": {
            "mean": float(np.mean(cosines)),
            "std": float(np.std(cosines)),
            "min": float(np.min(cosines)),
            "max": float(np.max(cosines)),
            "q25": float(np.percentile(cosines, 25)),
            "q50": float(np.percentile(cosines, 50)),
            "q75": float(np.percentile(cosines, 75)),
        },
        "num_targets": len(results),
    }

    print("\n  Distribution Statistics:")
    print(f"    Error:  mean={stats['reconstruction_error']['mean']:.4f} ± "
          f"{stats['reconstruction_error']['std']:.4f}, "
          f"range=[{stats['reconstruction_error']['min']:.4f}, "
          f"{stats['reconstruction_error']['max']:.4f}]")
    print(f"    Cosine: mean={stats['cosine_similarity']['mean']:.4f} ± "
          f"{stats['cosine_similarity']['std']:.4f}")

    return stats


# ================================================================
# Task 3.3: Random donor baseline
# ================================================================

def random_donor_baseline(matrix_fp16, target_indices, styles, config, alpha):
    """Test reconstruction with random subsets of k donors."""
    print("\n" + "=" * 60)
    print("TASK 3.3 — Random donor baseline")
    print("=" * 60)

    cfg3 = config["phase3"]
    ks = cfg3["random_donor_ks"]
    repeats = cfg3["random_repeats"]
    N = matrix_fp16.shape[1]
    rng = np.random.RandomState(config["regression_seed"])

    results = []
    for k in ks:
        for tidx in target_indices:
            all_donors = [i for i in range(N) if i != tidx]
            x_target = matrix_fp16[:, tidx].float().numpy()
            for rep in range(repeats):
                selected = rng.choice(all_donors, size=min(k, len(all_donors)), replace=False)
                X_sel = matrix_fp16[:, selected.tolist()].float().numpy()

                w = solve_ridge(X_sel, x_target, alpha)
                x_recon = X_sel @ w

                x_t = torch.from_numpy(x_target)
                x_r = torch.from_numpy(x_recon.astype(np.float32))
                err = relative_reconstruction_error(x_t, x_r)
                cos = cosine_similarity(x_t, x_r)

                results.append({
                    "k": k,
                    "target_index": tidx,
                    "repeat": rep,
                    "relative_error": err,
                    "cosine_similarity": cos,
                })

        # Aggregate for this k
        k_results = [r for r in results if r["k"] == k]
        k_errors = [r["relative_error"] for r in k_results]
        print(f"  k={k:3d}: error={np.mean(k_errors):.4f} ± {np.std(k_errors):.4f}")

    return results


# ================================================================
# Task 3.4: Random tensor baseline
# ================================================================

def random_tensor_baseline(matrix_fp16, target_indices, styles, config, alpha):
    """Replace donors with random vectors of matching norm."""
    print("\n" + "=" * 60)
    print("TASK 3.4 — Random tensor baseline")
    print("=" * 60)

    N = matrix_fp16.shape[1]
    D = matrix_fp16.shape[0]
    rng = np.random.RandomState(config["regression_seed"] + 42)

    # Compute norms of real donors (from fp16 matrix)
    donor_norms = matrix_fp16.float().norm(dim=0).numpy()

    results = []
    for tidx in target_indices:
        x_target = matrix_fp16[:, tidx].float().numpy()
        donor_cols = [i for i in range(N) if i != tidx]

        # Generate random donors with matching norms
        X_random = rng.randn(D, len(donor_cols)).astype(np.float32)
        for j, dcol in enumerate(donor_cols):
            col_norm = np.linalg.norm(X_random[:, j])
            if col_norm > 1e-12:
                X_random[:, j] *= donor_norms[dcol] / col_norm

        w = solve_ridge(X_random, x_target, alpha)
        x_recon = X_random @ w

        x_t = torch.from_numpy(x_target)
        x_r = torch.from_numpy(x_recon.astype(np.float32))
        err = relative_reconstruction_error(x_t, x_r)
        cos = cosine_similarity(x_t, x_r)

        results.append({
            "target_index": tidx,
            "target_name": styles[tidx]["name"],
            "relative_error": err,
            "cosine_similarity": cos,
        })
        print(f"  [{tidx}] {styles[tidx]['name']:40s}: error={err:.4f}, cos={cos:.4f}")

    mean_err = np.mean([r["relative_error"] for r in results])
    print(f"\n  Random baseline mean error: {mean_err:.4f}")
    return results


# ================================================================
# Task 3.5: Sparsity analysis
# ================================================================

def sparsity_analysis(all_results, config):
    """Analyze coefficient sparsity patterns."""
    print("\n" + "=" * 60)
    print("TASK 3.5 — Sparsity analysis")
    print("=" * 60)

    energy_threshold = config["phase3"]["energy_threshold"]
    k_stars = []

    for r in all_results:
        w = np.array(r["coefficients"])
        sorted_abs = np.sort(np.abs(w))[::-1]
        total = np.sum(sorted_abs)
        if total < 1e-12:
            k_stars.append(len(w))
            continue

        cumulative = np.cumsum(sorted_abs) / total
        k_star = int(np.searchsorted(cumulative, energy_threshold) + 1)
        k_stars.append(k_star)

    print(f"  k* (donors for {energy_threshold*100:.0f}% energy):")
    print(f"    mean={np.mean(k_stars):.1f}, median={np.median(k_stars):.0f}, "
          f"min={min(k_stars)}, max={max(k_stars)}")

    sparsity_data = {
        "energy_threshold": energy_threshold,
        "k_star_per_target": [
            {"target_index": r["target_index"], "target_name": r["target_name"], "k_star": k}
            for r, k in zip(all_results, k_stars)
        ],
        "k_star_stats": {
            "mean": float(np.mean(k_stars)),
            "median": float(np.median(k_stars)),
            "min": int(min(k_stars)),
            "max": int(max(k_stars)),
            "std": float(np.std(k_stars)),
        },
    }
    return sparsity_data


# ================================================================
# Task 3.6: Hub donor analysis
# ================================================================

def hub_analysis(all_results, styles, config):
    """Find hub styles that appear as significant donors across many targets."""
    print("\n" + "=" * 60)
    print("TASK 3.6 — Hub donor analysis")
    print("=" * 60)

    N = len(styles)
    threshold = 0.05
    donor_counts = Counter()

    for r in all_results:
        tidx = r["target_index"]
        w = np.array(r["coefficients"])
        # donor indices: all except target
        donor_indices = [i for i in range(N) if i != tidx]
        for j, didx in enumerate(donor_indices):
            if abs(w[j]) > threshold:
                donor_counts[didx] += 1

    print("  Top 10 hub styles (appear as significant donor most often):")
    for didx, count in donor_counts.most_common(10):
        print(f"    [{didx:3d}] {styles[didx]['name']:40s} → {count} times")

    return {
        "threshold": threshold,
        "top_hubs": [
            {"index": didx, "name": styles[didx]["name"], "count": count}
            for didx, count in donor_counts.most_common(20)
        ],
    }


# ================================================================
# Task 3.7: SVD spectrum
# ================================================================

def svd_spectrum_analysis(matrix_fp16, config):
    """Compute SVD of the full matrix to determine effective dimensionality."""
    print("\n" + "=" * 60)
    print("TASK 3.7 — SVD spectrum analysis")
    print("=" * 60)

    # Convert to float32 for SVD (float64 would be better but may OOM)
    matrix_f32 = matrix_fp16.float()
    D, N = matrix_f32.shape
    print(f"  Matrix: ({D:,}, {N})")
    print(f"  Computing SVD (only singular values needed)...")

    t0 = time.time()
    # Compute via X^T @ X eigendecomposition (much cheaper: N×N instead of D×D)
    gram = matrix_f32.T @ matrix_f32  # (N, N)
    eigenvalues, _ = torch.linalg.eigh(gram.double())
    eigenvalues = eigenvalues.flip(0)  # Descending
    singular_values = torch.sqrt(torch.clamp(eigenvalues, min=0)).numpy()
    elapsed = time.time() - t0
    print(f"  SVD computed in {elapsed:.1f}s")

    # Cumulative explained variance (Frobenius norm = sum of squared singular values)
    sv_squared = singular_values ** 2
    total_energy = np.sum(sv_squared)
    cumulative = np.cumsum(sv_squared) / total_energy

    r95 = int(np.searchsorted(cumulative, 0.95) + 1)
    r99 = int(np.searchsorted(cumulative, 0.99) + 1)

    print(f"  Singular values: {N} total")
    print(f"  Top-5 singular values: {singular_values[:5]}")
    print(f"  Effective rank (95% energy): {r95}")
    print(f"  Effective rank (99% energy): {r99}")
    print(f"  Condition number: {singular_values[0]/max(singular_values[-1], 1e-12):.2e}")

    return {
        "singular_values": singular_values.tolist(),
        "cumulative_energy": cumulative.tolist(),
        "effective_rank_95": r95,
        "effective_rank_99": r99,
        "condition_number": float(singular_values[0] / max(singular_values[-1], 1e-12)),
        "compute_time_seconds": elapsed,
    }


# ================================================================
# Task 3.8: Span classification
# ================================================================

def classify_span_membership(all_results, config):
    """Classify each style by span membership level."""
    print("\n" + "=" * 60)
    print("TASK 3.8 — Span classification")
    print("=" * 60)

    thresholds = config["phase3"]["span_thresholds"]
    classifications = []
    counts = Counter()

    for r in all_results:
        err = r["relative_error"]
        if err < thresholds["in_span"]:
            cls = "in_span"
        elif err < thresholds["approx_in_span"]:
            cls = "approximately_in_span"
        elif err < thresholds["partial_in_span"]:
            cls = "partially_in_span"
        else:
            cls = "not_in_span"

        counts[cls] += 1
        classifications.append({
            "target_index": r["target_index"],
            "target_name": r["target_name"],
            "relative_error": err,
            "cosine_similarity": r["cosine_similarity"],
            "classification": cls,
        })

    print(f"  Classification results:")
    for cls, count in sorted(counts.items()):
        print(f"    {cls:30s}: {count}")

    return {
        "thresholds": thresholds,
        "counts": dict(counts),
        "classifications": classifications,
    }


# ================================================================
# Plotting
# ================================================================

def generate_plots(all_results, random_donor_results, svd_data, sparsity_data, config):
    """Generate all Phase 3 plots."""
    print("\n" + "=" * 60)
    print("Generating plots")
    print("=" * 60)

    plot_dir = Path(config["experiment_dir"]) / "results" / "phase3" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Error distribution histogram
    errors = [r["relative_error"] for r in all_results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=30, edgecolor="black", alpha=0.7)
    ax.axvline(x=0.1, color="green", linestyle="--", label="In span (<0.1)")
    ax.axvline(x=0.3, color="orange", linestyle="--", label="Approx in span (<0.3)")
    ax.axvline(x=0.5, color="red", linestyle="--", label="Partial (<0.5)")
    ax.set_xlabel("Relative Reconstruction Error")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Reconstruction Error (All 109 Styles)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "error_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: error_distribution.png")

    # 2. Cosine similarity distribution
    cosines = [r["cosine_similarity"] for r in all_results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cosines, bins=30, edgecolor="black", alpha=0.7, color="teal")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Cosine Similarity (All 109 Styles)")
    fig.tight_layout()
    fig.savefig(plot_dir / "cosine_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: cosine_distribution.png")

    # 3. Error vs k (random donor baseline)
    if random_donor_results:
        ks = sorted(set(r["k"] for r in random_donor_results))
        mean_errors = []
        std_errors = []
        for k in ks:
            k_errs = [r["relative_error"] for r in random_donor_results if r["k"] == k]
            mean_errors.append(np.mean(k_errs))
            std_errors.append(np.std(k_errs))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(ks, mean_errors, yerr=std_errors, marker="o", capsize=5)
        ax.set_xlabel("Number of Donors (k)")
        ax.set_ylabel("Mean Relative Error")
        ax.set_title("Reconstruction Error vs Number of Donors")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(plot_dir / "error_vs_k.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: error_vs_k.png")

    # 4. SVD spectrum
    if svd_data:
        sv = np.array(svd_data["singular_values"])
        cumulative = np.array(svd_data["cumulative_energy"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.semilogy(range(1, len(sv) + 1), sv, "b.-")
        ax1.set_xlabel("Component Index")
        ax1.set_ylabel("Singular Value (log scale)")
        ax1.set_title("Singular Value Spectrum")

        ax2.plot(range(1, len(cumulative) + 1), cumulative, "r.-")
        ax2.axhline(y=0.95, color="green", linestyle="--", label="95% energy")
        ax2.axhline(y=0.99, color="orange", linestyle="--", label="99% energy")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Energy")
        ax2.set_title("Cumulative Explained Variance")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(plot_dir / "svd_spectrum.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: svd_spectrum.png")

    # 5. Sparsity boxplot (k* distribution)
    if sparsity_data:
        k_stars = [e["k_star"] for e in sparsity_data["k_star_per_target"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.boxplot(k_stars)
        ax.set_ylabel("k* (donors for 90% energy)")
        ax.set_title("Distribution of Required Donors")
        fig.tight_layout()
        fig.savefig(plot_dir / "sparsity_boxplot.png", dpi=150)
        plt.close(fig)
        print(f"  Saved: sparsity_boxplot.png")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: Span Membership Interpretation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--full-sweep", action="store_true",
                        help="Run full leave-one-out for all 109 styles")
    parser.add_argument("--baselines", action="store_true",
                        help="Run random donor and random tensor baselines")
    parser.add_argument("--svd", action="store_true",
                        help="Run SVD spectrum analysis")
    parser.add_argument("--plots", action="store_true",
                        help="Generate all plots")
    parser.add_argument("--all", action="store_true",
                        help="Run everything")
    args = parser.parse_args()

    if args.all:
        args.full_sweep = True
        args.baselines = True
        args.svd = True
        args.plots = True

    config = load_config(args.config)
    exp_dir = Path(config["experiment_dir"])

    print("=" * 60)
    print("Phase 3 — Span Membership Interpretation")
    print("=" * 60)

    # Load matrix
    matrix, meta = load_matrix_and_meta(config)
    D, N = matrix.shape
    print(f"  Matrix: ({D:,}, {N})")

    styles = discover_lora_pool(config["lora_pool_dir"])
    method, alpha, _ = load_best_method(config)
    print(f"  Method: {method}, alpha={alpha}")

    # Keep matrix as fp16 torch tensor.
    # Each function casts only its needed slice to float32 per-iteration.
    # Peak memory: ~62 GB (fp16) + ~123 GB (one LOO slice fp32) = ~185 GB.
    matrix_fp16 = matrix
    del matrix

    all_results = None
    random_donor_results = None
    svd_data = None
    sparsity_data = None

    # ── Task 3.1: Full sweep ──
    if args.full_sweep:
        all_results = full_leave_one_out(matrix_fp16, styles, config, method, alpha)

        # Save (without coefficients in main file for size)
        results_no_coeff = [{k: v for k, v in r.items() if k != "coefficients"} for r in all_results]
        save_json(results_no_coeff, exp_dir / "results" / "phase3" / "all_targets_results.json")

        # Save coefficients separately
        all_coeffs = {r["target_index"]: r["coefficients"] for r in all_results}
        np.savez_compressed(
            exp_dir / "results" / "phase3" / "all_coefficients.npz",
            **{str(k): np.array(v) for k, v in all_coeffs.items()}
        )

        # Statistics
        stats = compute_distribution_stats(all_results)
        save_json(stats, exp_dir / "results" / "phase3" / "distribution_stats.json")

        # Sparsity
        sparsity_data = sparsity_analysis(all_results, config)
        save_json(sparsity_data, exp_dir / "results" / "phase3" / "sparsity_analysis.json")

        # Hub analysis
        hub_data = hub_analysis(all_results, styles, config)
        save_json(hub_data, exp_dir / "results" / "phase3" / "hub_analysis.json")

        # Classification
        classification = classify_span_membership(all_results, config)
        save_json(classification, exp_dir / "results" / "phase3" / "span_classification.json")

    # Try to load if not computed
    if all_results is None:
        result_path = exp_dir / "results" / "phase3" / "all_targets_results.json"
        if result_path.exists():
            with open(result_path) as f:
                all_results = json.load(f)
            print(f"  Loaded {len(all_results)} results from previous run")

    # ── Task 3.3–3.4: Baselines ──
    if args.baselines:
        # Use Phase 1 target selection for baselines
        target_path = exp_dir / "results" / "phase1" / "target_selection.json"
        if target_path.exists():
            with open(target_path) as f:
                target_indices = [t["index"] for t in json.load(f)]
        else:
            target_indices = list(range(min(10, N)))

        random_donor_results = random_donor_baseline(matrix_fp16, target_indices, styles, config, alpha)
        save_json(random_donor_results, exp_dir / "results" / "phase3" / "random_donor_baseline.json")

        random_tensor_results = random_tensor_baseline(matrix_fp16, target_indices, styles, config, alpha)
        save_json(random_tensor_results, exp_dir / "results" / "phase3" / "random_tensor_baseline.json")

    # ── Task 3.7: SVD ──
    if args.svd:
        svd_data = svd_spectrum_analysis(matrix_fp16, config)
        save_json(svd_data, exp_dir / "results" / "phase3" / "svd_spectrum.json")

    # ── Plots ──
    if args.plots:
        # Load any missing data from disk
        if random_donor_results is None:
            rdb_path = exp_dir / "results" / "phase3" / "random_donor_baseline.json"
            if rdb_path.exists():
                with open(rdb_path) as f:
                    random_donor_results = json.load(f)

        if svd_data is None:
            svd_path = exp_dir / "results" / "phase3" / "svd_spectrum.json"
            if svd_path.exists():
                with open(svd_path) as f:
                    svd_data = json.load(f)

        if sparsity_data is None:
            sp_path = exp_dir / "results" / "phase3" / "sparsity_analysis.json"
            if sp_path.exists():
                with open(sp_path) as f:
                    sparsity_data = json.load(f)

        if all_results is not None:
            generate_plots(all_results, random_donor_results, svd_data, sparsity_data, config)
        else:
            print("  Cannot generate plots: no results available. Run --full-sweep first.")

    # ── Summary ──
    if all_results is not None:
        print("\n" + "=" * 60)
        print("PHASE 3 FINAL SUMMARY")
        print("=" * 60)

        errors = [r["relative_error"] for r in all_results]
        cosines = [r["cosine_similarity"] for r in all_results]

        thresholds = config["phase3"]["span_thresholds"]
        n_in = sum(1 for e in errors if e < thresholds["in_span"])
        n_approx = sum(1 for e in errors if thresholds["in_span"] <= e < thresholds["approx_in_span"])
        n_partial = sum(1 for e in errors if thresholds["approx_in_span"] <= e < thresholds["partial_in_span"])
        n_not = sum(1 for e in errors if e >= thresholds["partial_in_span"])

        print(f"\n  ANSWER TO THE SCIENTIFIC QUESTION:")
        print(f"  ─────────────────────────────────────")
        print(f"  In span (error < {thresholds['in_span']}):           {n_in:3d} / {len(errors)}")
        print(f"  Approximately in span (< {thresholds['approx_in_span']}):   {n_approx:3d} / {len(errors)}")
        print(f"  Partially in span (< {thresholds['partial_in_span']}):      {n_partial:3d} / {len(errors)}")
        print(f"  Not in span (≥ {thresholds['partial_in_span']}):            {n_not:3d} / {len(errors)}")
        print(f"\n  Mean error: {np.mean(errors):.4f}")
        print(f"  Mean cosine: {np.mean(cosines):.4f}")

    log_entry(config, {
        "phase": "3",
        "task": "3.1-3.9",
        "description": "Phase 3 span analysis complete",
        "full_sweep_run": args.full_sweep,
        "baselines_run": args.baselines,
        "svd_run": args.svd,
    })

    print("\n✓ Phase 3 complete.")


if __name__ == "__main__":
    main()
