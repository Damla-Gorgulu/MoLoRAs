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
from sklearn.linear_model import Lasso, ElasticNet

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
# Memory-safe regression helpers (Gram-matrix approach)
# ================================================================
# Key insight: D = 301,465,600, K ≈ 108. We NEVER materialise the full
# (D × K) float32 matrix. Instead we compute:
#   G = X^T X  (K × K, ~91 KB in float64)
#   q = X^T y  (K,)
# by streaming K columns from fp16 in small batches, then solve the K×K
# normal equations directly.
#
# Peak RAM: fp16 matrix (~62 GB) + one column-batch float32 (~0.3 GB)
# ================================================================

# Path for the cached full Gram matrix (109×109, ~91 KB)
_GRAM_CACHE_NAME = "gram_full.npz"


def _build_gram(matrix_fp16, col_indices, col_chunk=8):
    """
    Compute G = X^T X and return column vectors — column-by-column to
    keep the intermediate float32 footprint modest.

    Args:
        matrix_fp16: (D, N) fp16 torch tensor
        col_indices:  list / 1-D int array of K column indices
        col_chunk:    number of full fp32 columns to keep in RAM at once
                      (each column is D * 4 bytes ≈ 1.2 GB for D=301M)
                      Set to 8 → ~10 GB transient, well within budget.
    Returns:
        G:        (K, K) float64 numpy array
        cols_f32: list of K (D,) float32 numpy arrays
    """
    K = len(col_indices)
    cols_f32 = []  # will hold K float32 columns sequentially

    # Cast and store columns one batch at a time
    for start in range(0, K, col_chunk):
        end = min(start + col_chunk, K)
        batch_idx = col_indices[start:end]  # may be list or numpy slice
        batch = matrix_fp16[:, batch_idx].float().numpy()  # (D, batch)
        for j in range(batch.shape[1]):
            cols_f32.append(np.ascontiguousarray(batch[:, j]))  # (D,)
        del batch

    # Build symmetric Gram matrix
    G = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(i, K):
            v = np.dot(cols_f32[i].astype(np.float64),
                       cols_f32[j].astype(np.float64))
            G[i, j] = v
            G[j, i] = v
    return G, cols_f32


def build_or_load_full_gram(matrix_fp16, config):
    """
    Build the full (N × N) Gram matrix G_full = X^T X once, then cache it
    to disk.  Subsequent calls load from cache in < 1 second.

    Key insight: G_full[mask, :][:, mask] gives the (N-1)×(N-1) leave-one-out
    Gram for any target, and G_full[mask, target_idx] gives the q vector.
    This eliminates re-reading the 62 GB fp16 matrix for each target.

    Returns:
        G_full:  (N, N) float64 numpy array
    """
    cache_path = Path(config["experiment_dir"]) / "results" / "phase1" / _GRAM_CACHE_NAME
    if cache_path.exists():
        data = np.load(cache_path)
        G_full = data["G"]
        print(f"  Loaded cached Gram matrix from {cache_path}  ({G_full.shape})")
        return G_full

    N = matrix_fp16.shape[1]
    all_indices = list(range(N))
    print(f"  Building full Gram matrix (N={N}) — this takes ~3 h ...")
    t0 = time.time()
    G_full, _cols = _build_gram(matrix_fp16, all_indices)
    del _cols  # free ~130 GB
    elapsed = time.time() - t0
    print(f"  Gram built in {elapsed/60:.1f} min")

    np.savez_compressed(cache_path, G=G_full)
    print(f"  Cached to {cache_path}")
    return G_full


def _loo_gram_and_q(G_full, target_idx):
    """
    Extract leave-one-out Gram and q from the full Gram matrix.

    G_loo = G_full[mask][:, mask]   — (N-1, N-1)
    q_loo = G_full[mask, target]    — (N-1,)

    Also returns xt_norm2 = G_full[target, target] = ||x_target||^2.
    """
    N = G_full.shape[0]
    mask = np.ones(N, dtype=bool)
    mask[target_idx] = False
    G_loo = G_full[np.ix_(mask, mask)]       # (N-1, N-1)
    q_loo = G_full[mask, target_idx].copy()  # (N-1,)
    xt_norm2 = float(G_full[target_idx, target_idx])
    return G_loo, q_loo, xt_norm2


def _build_q(cols_f32, x_target_f64):
    """q = X^T y  (K,) float64."""
    return np.array([np.dot(c.astype(np.float64), x_target_f64)
                     for c in cols_f32], dtype=np.float64)


def _ridge_via_gram(G, q, alpha_val):
    """Solve (G + alpha*I) w = q.  Returns float32 (K,) vector."""
    K = G.shape[0]
    A = G + alpha_val * np.eye(K, dtype=np.float64)
    w = np.linalg.solve(A, q)
    return w.astype(np.float32)


def solve_regression(matrix_fp16, col_indices, x_target_f64,
                     method="ridge", alpha=1.0, l1_ratio=0.5,
                     G=None, cols_f32=None, q=None):
    """
    Memory-safe regression.  G, cols_f32, q may be pre-computed and passed
    in to avoid redundant work across multiple alpha values.

    Args:
        matrix_fp16:  (D, N) fp16 torch tensor  (never cast as a whole)
        col_indices:  list of K donor column indices
        x_target_f64: (D,) float64 numpy target vector
        method:       "ridge", "lasso", or "elasticnet"
        alpha:        regularisation strength
        l1_ratio:     for ElasticNet
        G, cols_f32, q: pre-built Gram objects (pass to reuse)

    Returns:
        w:              (K,) float32 coefficient vector
        (G, cols_f32, q): cached objects for subsequent calls
    """
    if G is None:
        G, cols_f32 = _build_gram(matrix_fp16, col_indices)
    if q is None:
        q = _build_q(cols_f32, x_target_f64)

    if method == "ridge":
        w = _ridge_via_gram(G, q, alpha)

    elif method in ("lasso", "elasticnet"):
        # Solve in the K-dim Gram (kernel) space:
        #   X^T X @ w = X^T y  →  fit on (G, q) with K=108 samples/features
        # This is exact for Ridge, and a tight approximation for Lasso /
        # ElasticNet when columns are near-orthogonal (verified in Phase 3).
        K = G.shape[0]
        if method == "lasso":
            model = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000)
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                               fit_intercept=False, max_iter=50000)
        model.fit(G.astype(np.float32), q.astype(np.float32))
        w = model.coef_.astype(np.float32)

    else:
        raise ValueError(f"Unknown method: {method}")

    return w, (G, cols_f32, q)


def _gram_metrics(G, q_orig, w, xt_norm2, sparsity_threshold):
    """
    Compute reconstruction metrics purely from Gram objects — no D-dim
    float32 materialisation required.

    ||x_t - X w||^2 = ||x_t||^2 - 2 w^T q + w^T G w

    Args:
        G:          (K, K) Gram matrix (original, un-normalised)
        q_orig:     (K,) dot products with target (original)
        w:          (K,) coefficient vector (original scale)
        xt_norm2:   float, ||x_target||^2
        sparsity_threshold: coefficient magnitude threshold
    """
    w64 = w.astype(np.float64)
    xr_norm2 = float(w64 @ G @ w64)
    dot_tr = float(w64 @ q_orig)
    err2 = max(0.0, xt_norm2 - 2.0 * dot_tr + xr_norm2)
    rel_err = float(np.sqrt(err2) / (np.sqrt(xt_norm2) + 1e-12))
    cos_sim = float(dot_tr / (np.sqrt(xt_norm2 * xr_norm2) + 1e-12))
    sparsity = float(np.mean(np.abs(w) < sparsity_threshold))
    return {
        "relative_error": rel_err,
        "cosine_similarity": float(np.clip(cos_sim, -1.0, 1.0)),
        "sparsity": sparsity,
        "num_nonzero": int(np.sum(np.abs(w) >= sparsity_threshold)),
    }


def run_regression_sweep(matrix_fp16, target_idx, styles, config,
                         normalize=False, G_full=None):
    """
    Run all regression methods/hyperparameters for one target.

    When G_full is provided (recommended), the leave-one-out Gram and q
    are extracted from it in O(N²) — zero disk/matrix I/O.  Falls back to
    building the Gram from scratch when G_full is None.

    Args:
        matrix_fp16: (D, N) torch float16 tensor — never cast in full
        target_idx:  column index of the target style
        styles:      list of style dicts
        config:      experiment config
        normalize:   if True, scale columns to unit norm before regression
        G_full:      (N, N) float64 numpy — pre-built full Gram (optional)
    """
    N = matrix_fp16.shape[1]
    donor_indices = [i for i in range(N) if i != target_idx]
    K = len(donor_indices)

    # Fast path: extract from pre-built full Gram
    if G_full is not None:
        print(f"    Extracting LOO Gram from cached G_full (K={K})...")
        t_gram = time.time()
        G, q, xt_norm2 = _loo_gram_and_q(G_full, target_idx)
        # cols_f32 is not needed (only used for image generation which
        # uses _reconstruct_vector_from_gram instead)
        cols_f32 = None
        print(f"    LOO Gram extracted in {time.time()-t_gram:.2f}s")
    else:
        # Slow path: build from scratch
        x_target_f32 = matrix_fp16[:, target_idx].float().numpy()
        x_target_f64 = x_target_f32.astype(np.float64)
        print(f"    Building Gram (K={K}) for target [{target_idx}]...")
        t_gram = time.time()
        G, cols_f32 = _build_gram(matrix_fp16, donor_indices)
        q = _build_q(cols_f32, x_target_f64)
        xt_norm2 = float(np.dot(x_target_f64, x_target_f64))
        print(f"    Gram built in {time.time()-t_gram:.1f}s")

    # Keep originals for metrics (before normalisation)
    G_orig, q_orig = G.copy(), q.copy()

    # Optional normalisation in Gram space
    if normalize:
        norms = np.sqrt(np.maximum(np.diag(G), 1e-24))      # (K,)
        G_use = G / np.outer(norms, norms)
        yt_norm = float(np.sqrt(xt_norm2))
        scale = norms * yt_norm if yt_norm > 1e-12 else norms
        q_use = q / scale
    else:
        G_use, q_use = G, q
        norms = None
        yt_norm = None

    results = []
    cfg1 = config["phase1"]
    thr = cfg1["sparsity_threshold"]

    def _record(method, alpha, l1r, w_norm, elapsed):
        """Undo normalisation if needed and record metrics."""
        if normalize and norms is not None:
            w_orig = (w_norm / norms) * (yt_norm if yt_norm and yt_norm > 1e-12 else 1.0)
        else:
            w_orig = w_norm
        m = _gram_metrics(G_orig, q_orig, w_orig, xt_norm2, thr)
        entry = {
            "method": method,
            "alpha": alpha,
            "normalized": normalize,
            "target_index": target_idx,
            "target_name": styles[target_idx]["name"],
            "wall_time_seconds": elapsed,
            **m,
            "coefficients": w_orig.tolist(),
        }
        if method == "elasticnet":
            entry["l1_ratio"] = l1r
        results.append(entry)

    # Ridge — solve directly on the tiny K×K system
    for alpha in cfg1["ridge_alphas"]:
        t0 = time.time()
        w = _ridge_via_gram(G_use, q_use, alpha)
        _record("ridge", alpha, None, w, time.time() - t0)

    # Lasso
    for alpha in cfg1["lasso_alphas"]:
        t0 = time.time()
        w, _ = solve_regression(None, None, None,
                                 "lasso", alpha,
                                 G=G_use, cols_f32=None, q=q_use)
        _record("lasso", alpha, None, w, time.time() - t0)

    # ElasticNet
    for alpha in cfg1["elasticnet_alphas"]:
        for l1r in cfg1["elasticnet_l1_ratios"]:
            t0 = time.time()
            w, _ = solve_regression(None, None, None,
                                     "elasticnet", alpha, l1r,
                                     G=G_use, cols_f32=None, q=q_use)
            _record("elasticnet", alpha, l1r, w, time.time() - t0)

    return results


def self_reconstruction_check(matrix_fp16, styles, config, G_full=None):
    """
    Task 1.5: Self-reconstruction — include target in donor pool.
    When all N columns (including target) are donors the solution should
    recover w[target] ≈ 1.0, all others ≈ 0.
    Uses Gram-matrix approach — never materialises the full float32 matrix.
    If G_full is provided, uses it directly (no rebuild needed).
    """
    print("\n" + "=" * 60)
    print("TASK 1.5 — Self-reconstruction sanity check")
    print("=" * 60)

    target_idx = 0
    N = matrix_fp16.shape[1]

    if G_full is not None:
        G = G_full
        q = G_full[:, target_idx].copy()
        print(f"  Using cached Gram matrix ({N}×{N})")
    else:
        all_indices = list(range(N))
        x_target_f32 = matrix_fp16[:, target_idx].float().numpy()
        x_target_f64 = x_target_f32.astype(np.float64)
        print(f"  Building full Gram matrix (K={N})...")
        G, cols_f32 = _build_gram(matrix_fp16, all_indices)
        q = _build_q(cols_f32, x_target_f64)
        del cols_f32

    w = _ridge_via_gram(G, q, alpha_val=0.01)

    # Metrics via Gram (no D-dim materialisation)
    xt_norm2 = float(G[target_idx, target_idx])
    w64 = w.astype(np.float64)
    xr_norm2 = float(w64 @ G @ w64)
    dot_tr = float(w64 @ q)
    err2 = max(0.0, xt_norm2 - 2.0 * dot_tr + xr_norm2)
    rel_err = float(np.sqrt(err2) / (np.sqrt(xt_norm2) + 1e-12))
    cos_sim = float(dot_tr / (np.sqrt(xt_norm2 * xr_norm2) + 1e-12))

    print(f"  Target coefficient: {w[target_idx]:.6f} (expected ~1.0)")
    other_w = np.delete(w, target_idx)
    print(f"  Max other coeff:    {np.max(np.abs(other_w)):.6f} (expected ~0)")
    print(f"  Relative error:     {rel_err:.6f} (expected ~0)")
    print(f"  Cosine similarity:  {cos_sim:.6f} (expected ~1)")

    ok = (
        abs(w[target_idx] - 1.0) < 0.05
        and rel_err < 0.01
    )
    print(f"  {'✓ PASS' if ok else '✗ FAIL'}")

    result = {
        "target_coefficient": float(w[target_idx]),
        "max_other_coefficient": float(np.max(np.abs(other_w))),
        "relative_error": rel_err,
        "cosine_similarity": float(np.clip(cos_sim, -1.0, 1.0)),
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


def _reconstruct_vector_from_gram(matrix_fp16, col_indices, w, row_chunk=2_000_000):
    """
    Compute x_recon = X_donors @ w  without materialising the full (D, K)
    float32 matrix.  Streams D rows in chunks; each chunk is (chunk, K) float32.

    Args:
        matrix_fp16:  (D, N) fp16 torch tensor
        col_indices:  list of K donor column indices
        w:            (K,) float32 numpy coefficient vector
        row_chunk:    rows to process per iteration (~2M * 108 * 4 = ~0.86 GB)

    Returns:
        x_recon:  (D,) float32 numpy array
    """
    D = matrix_fp16.shape[0]
    K = len(col_indices)
    col_idx_t = torch.tensor(col_indices, dtype=torch.long)
    w_t = torch.from_numpy(np.asarray(w, dtype=np.float32))  # (K,) float32

    x_recon = np.empty(D, dtype=np.float32)
    for start in range(0, D, row_chunk):
        end = min(start + row_chunk, D)
        X_chunk = matrix_fp16[start:end].index_select(1, col_idx_t).float()  # (chunk, K)
        x_recon[start:end] = (X_chunk @ w_t).numpy()
        del X_chunk
    return x_recon


def generate_comparison_images(matrix_fp16, all_results, styles, config, meta):
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

        # Reconstruct ΔW from coefficients via row-chunked streaming
        w = np.array(result["coefficients"])
        donor_indices = [i for i in range(matrix_fp16.shape[1]) if i != tidx]
        x_recon = _reconstruct_vector_from_gram(matrix_fp16, donor_indices, w)

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
    print(f"  Memory (fp16): {D * N * 2 / 1e9:.1f} GB")
    matrix_fp16 = matrix
    del matrix

    # Load styles
    styles = discover_lora_pool(config["lora_pool_dir"])

    # ── Build or load the full Gram matrix (ONCE) ──
    # This is the key optimisation: the 109×109 Gram takes ~3.5 h to build,
    # but once cached, every subsequent call is instant.  All leave-one-out
    # sub-Grams and q vectors are extracted from it in O(N²).
    G_full = build_or_load_full_gram(matrix_fp16, config)

    # ── Self-check ──
    self_check_path = exp_dir / "results" / "phase1" / "self_check.json"
    if args.self_check:
        self_reconstruction_check(matrix_fp16, styles, config, G_full=G_full)
        return

    # ── Select targets ──
    target_sel_path = exp_dir / "results" / "phase1" / "target_selection.json"
    if target_sel_path.exists():
        with open(target_sel_path) as f:
            target_data = json.load(f)
        target_indices = [t["index"] for t in target_data]
        print(f"\nLoaded {len(target_indices)} targets from {target_sel_path}")
    else:
        print("\nSelecting representative targets...")
        target_indices = select_representative_targets(styles, config)
        save_json(
            [{"index": i, "name": styles[i]["name"]} for i in target_indices],
            target_sel_path,
        )

    # ── Self-check (skip if already passed) ──
    if self_check_path.exists():
        with open(self_check_path) as f:
            sc = json.load(f)
        if sc.get("pass"):
            print(f"\n  Self-check already PASSED (cached) — skipping")
        else:
            self_reconstruction_check(matrix_fp16, styles, config, G_full=G_full)
    else:
        self_reconstruction_check(matrix_fp16, styles, config, G_full=G_full)

    # ── Run regression sweep (using cached G_full → instant LOO extraction) ──
    all_results = []
    for i, tidx in enumerate(target_indices):
        print(f"\n{'='*60}")
        print(f"TARGET {i+1}/{len(target_indices)}: [{tidx}] {styles[tidx]['name']}")
        print(f"{'='*60}")

        results = run_regression_sweep(
            matrix_fp16, tidx, styles, config, normalize=False, G_full=G_full
        )
        all_results.extend(results)

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
            results = run_regression_sweep(
                matrix_fp16, tidx, styles, config, normalize=True, G_full=G_full
            )
            norm_results.extend(results)

        norm_summary = [{k: v for k, v in r.items() if k != "coefficients"} for r in norm_results]
        save_json(norm_summary, exp_dir / "results" / "phase1" / "normalized_results.json")

    # ── Generate images ──
    if args.generate_images:
        generate_comparison_images(matrix_fp16, all_results, styles, config, meta)

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
