# ROADMAP — LoRA Linear Composition Experiment

---

## 1. Motivation & Scientific Question

Given a pool of 109 B-LoRA style adapters (rank 64, trained on WikiArt), we ask:
**Can the ΔW of any single target style be expressed as a linear combination of the remaining 108 ΔWs?**

This is a pure linear algebra question about whether the style LoRA space is linearly redundant.

Two hypotheses are tested:

- **H1 (Global):** A single scalar weight per donor LoRA suffices across all tensors.
- **H2 (Layer-wise):** Different scalar weights per tensor group yield better reconstruction.

The null hypothesis is that each style occupies a unique direction not spanned by the others.

**EXTREMELY IMPORTANT CONSTRAINT:**
Do NOT treat LoRA as A and B separately. Always work with:

```
ΔW = B @ A
```

because A and B are not unique, but ΔW is unique. All linear combinations must be done on ΔW.

---

## 2. Integration with Existing Mo-LoRAs System

The master document `lora_attention/roadmap.md` (2013 lines) establishes conventions that this experiment **must** respect:

| Aspect | Convention from roadmap.md | How this experiment uses it |
|--------|---------------------------|----------------------------|
| **Expert pool location** | `repos/B-LoRA/blora_zoo/bloras/style_NNNN_StyleName/` | Load all 109 LoRAs from this path. Do NOT retrain or relocate. |
| **ΔW computation** | `W_up @ W_down` (confirmed in §22–23 of roadmap as the product-space fix) | Always compute `ΔW = B @ A` before any operation. Never operate on A, B separately. |
| **Tensor naming** | `unet.up_blocks.0.attentions.1.transformer_blocks.{0-9}.{attn1,attn2}.to_{q,k,v,out.0}` for style block | Use identical key names. Parse block index, attention type, and projection type from these keys. |
| **Rank** | 64 for the expert pool | All 109 LoRAs share rank 64. ΔW shapes are (1280×1280) for attn1 and (1280×2048) for attn2. |
| **Alpha scaling** | α = 2.0 is the validated sweet spot (§26 of roadmap) | Use α = 2.0 for all image generation sanity checks. |
| **Base model** | `stabilityai/stable-diffusion-xl-base-1.0` | Same base model for all image generation. |
| **SVD stability** | fp32 accumulation + CPU LAPACK fallback (§27–30) | If SVD is needed for re-decomposition, follow this convention. |
| **Injection method** | `pipe.load_lora_weights()` or direct weight merge `W += α·ΔW` | Use direct weight merge for reconstructed LoRA injection (most transparent). |
| **Style block** | `unet.up_blocks.0.attentions.1` only (B-LoRA convention) | Only these tensors are present in the safetensors files. 80 adapter pairs per LoRA. |

Additionally, `ip-lora/compute_projectors_from_loras.py` already contains a function that computes `delta_w = B @ A` and flattens all tensors into a single vector. This utility will be **reused or adapted** rather than reimplemented.

---

## 3. Experiment Phases

### PHASE 0 — Sanity Check (Validation of Pipeline)

**Goal:** Confirm that LoRA extraction, ΔW computation, and injection produce correct, visually distinct results before any reconstruction is attempted.

**Data flow:**
```
safetensors file → extract A, B per tensor key → compute ΔW = B @ A → verify norms → inject into base model → generate image
```

**Steps:**

1. **Load one target LoRA** (e.g., `style_0001_*`) from `repos/B-LoRA/blora_zoo/bloras/`.
2. **Extract all tensor keys.** Confirm exactly 80 adapter pairs exist per LoRA (10 transformer blocks × 2 attention types × 4 projections). Log all key names.
3. **Compute ΔW = B @ A** for every tensor. Log the Frobenius norm of each ΔW. Verify all norms are non-zero.
4. **Flatten and concatenate** all 80 ΔW tensors into one vector. Log total dimensionality D (expected: ~157M parameters).
5. **Generate 3 images** using the same prompt and same seed:
   - **Image 1:** Base SDXL, no LoRA.
   - **Image 2:** Base SDXL + target LoRA injected at α = 2.0 via direct weight merge.
   - **Image 3:** Base SDXL + target LoRA injected via `pipe.load_lora_weights()` at scale = 2.0.
6. **Compare Images 2 and 3** — they must be pixel-identical (validates injection method equivalence). Compare both against Image 1 — they must be visually different (validates LoRA has effect).

**Success criteria:**
- All 80 ΔW norms > 0.
- Images 2 and 3 are identical (pixel MSE < 1e-4).
- Images 2/3 are visibly different from Image 1.

**Failure condition:** If any ΔW is zero or images don't differ, the LoRA file is corrupt or injection is wrong. Debug before proceeding.

---

### PHASE 1 — Global Linear Reconstruction

**Goal:** Test whether a target LoRA's full flattened ΔW vector can be approximated as a weighted sum of 108 donor LoRA vectors using a single coefficient vector **w** ∈ ℝ¹⁰⁸.

**Mathematical formulation:**

Let `x_A ∈ ℝ^D` be the target LoRA's flattened ΔW vector (all 80 tensors concatenated).
Let `X ∈ ℝ^(D×108)` be the donor matrix where each column is a donor LoRA's flattened ΔW.
Solve:

```
min_w  ‖x_A − X·w‖² + λ·‖w‖
```

**Data flow:**
```
109 safetensors → 109 ΔW vectors (each ~157M dim) → remove target → build donor matrix X → solve regression → reconstruct → evaluate
```

**Steps:**

1. **Load all 109 LoRAs.** For each, compute ΔW = B @ A for all tensors, flatten and concatenate into one vector. Store as a matrix of shape (D × 109).
2. **For each target style** (leave-one-out, starting with a representative subset of ~10):
   a. Remove target column from matrix → donor matrix X (D × 108).
   b. Target vector = removed column.
3. **Solve regression** using three methods:
   - **Ridge (L2):** `sklearn.linear_model.Ridge` with α ∈ {0.01, 0.1, 1.0, 10.0}
   - **Lasso (L1):** `sklearn.linear_model.Lasso` with α ∈ {0.001, 0.01, 0.1}
   - **ElasticNet:** `sklearn.linear_model.ElasticNet` with l1_ratio ∈ {0.1, 0.5, 0.9}
4. **Compute metrics** for each solution:
   - **Relative reconstruction error:** `ε = ‖x_A − x̂_A‖ / ‖x_A‖`
   - **Cosine similarity:** `cos(x_A, x̂_A)`
   - **Sparsity:** number of coefficients with |w_i| > threshold
   - **Top-k energy:** fraction of total weight magnitude in top-k donors
5. **Reconstruct ΔW** from the best solution. Unflatten back into per-tensor ΔW matrices.
6. **Inject reconstructed LoRA** into base model. Generate one image with same prompt/seed as Phase 0.
7. **Visual comparison:** Place side by side: Base | Target LoRA | Reconstructed LoRA.

**Normalization strategy:**
- **Option A (default):** No normalization. Preserve natural scale differences between tensors.
- **Option B (ablation):** Normalize each LoRA's full vector to unit norm before regression, then rescale result.
- Run both and compare. Report which performs better.

**Memory consideration:**
- D ≈ 157M parameters × 109 LoRAs × 4 bytes = ~68 GB for full matrix in fp32.
- **Mitigation:** Use fp16 for storage, cast to fp32 only during regression. Or solve via the Gram matrix approach: compute `G = X^T·X` (108×108) and `b = X^T·x_target` (108×1) via streaming dot products without holding the full matrix in memory. This is the default if a machine with < 64 GB RAM is used.

**Success criteria:**

| Scenario | Relative Error | Cosine Sim | Interpretation |
|----------|---------------|------------|----------------|
| Strong span membership | < 0.10 | > 0.95 | Target is essentially in the span. |
| Partial span membership | 0.10 – 0.30 | 0.85 – 0.95 | Composition works but loses fine detail. |
| Weak span membership | 0.30 – 0.50 | 0.70 – 0.85 | Target has significant unique components. |
| Not in span | > 0.50 | < 0.70 | Linear composition fails. |

**Failure condition:** OOM during matrix construction → switch to Gram matrix approach. All methods give error > 0.5 → global reconstruction not feasible, proceed to Phase 2.

---

### PHASE 2 — Layer-wise Linear Reconstruction

**Goal:** Allow different coefficient vectors per tensor group, testing whether layer-specific weighting improves reconstruction.

**Mathematical formulation:**

Partition the 80 tensors into G groups. For each group g, solve independently:

```
min_{w^(g)}  ‖x_A^(g) − X^(g)·w^(g)‖² + λ·‖w^(g)‖
```

**Tensor grouping strategy** (derived from `B-LoRA_files/blora_block_definition.py`):

| Group ID | Description | Tensor keys | Count |
|----------|-------------|-------------|-------|
| G1 | Self-attention (attn1), all blocks | `transformer_blocks.*.attn1.to_{q,k,v,out.0}` | 40 |
| G2 | Cross-attention (attn2), all blocks | `transformer_blocks.*.attn2.to_{q,k,v,out.0}` | 40 |
| G3 | Early blocks, self-attn | `transformer_blocks.{0-3}.attn1.*` | 16 |
| G4 | Early blocks, cross-attn | `transformer_blocks.{0-3}.attn2.*` | 16 |
| G5 | Mid blocks, self-attn | `transformer_blocks.{4-6}.attn1.*` | 12 |
| G6 | Mid blocks, cross-attn | `transformer_blocks.{4-6}.attn2.*` | 12 |
| G7 | Late blocks, self-attn | `transformer_blocks.{7-9}.attn1.*` | 12 |
| G8 | Late blocks, cross-attn | `transformer_blocks.{7-9}.attn2.*` | 12 |
| G9 | Value + Output projections only | `*.to_v`, `*.to_out.0` | 40 |
| G10 | Query + Key projections only | `*.to_q`, `*.to_k` | 40 |

**Steps:**

1. **Define grouping schemes.** Start with coarse (G1, G2), then refine to (G3–G8), then test projection-specific (G9, G10).
2. **For each grouping scheme and each target style:**
   a. Extract relevant tensor subset for target and all donors.
   b. Flatten and concatenate within each group.
   c. Solve regression independently per group.
3. **Reconstruct full LoRA** by combining per-group solutions.
4. **Compute metrics:** per-group relative error and cosine similarity, plus overall error after combining all groups. Compare against Phase 1 global result.
5. **Generate one image** with reconstructed LoRA for visual comparison.
6. **Analyze which groups reconstruct well vs poorly.** This reveals which aspects of style (texture, color, composition) are more/less linearly decomposable.

**Expected outcome:** Layer-wise should perform ≥ global. If equal, the global weight vector is already layer-consistent. If much better, different style aspects rely on different donor combinations.

---

### PHASE 3 — Span Membership Interpretation

**Goal:** Synthesize results from Phases 1–2 into a definitive answer about linear span membership.

**Steps:**

1. **Expand to all 109 targets** using the best method identified in Phases 1–2. Run full leave-one-out reconstruction for all 109 styles.
2. **Compute distribution statistics:** mean, std, min, max, quartiles of reconstruction error across all 109 targets.
3. **Baseline comparisons:**
   - **Random donor selection:** Pick k ∈ {5, 10, 20, 50} random donors and solve regression. Repeat 5 times. Plot error vs k.
   - **Random tensor baseline:** Replace all donor ΔWs with random Gaussian vectors of matching norm. Solve regression. Validates whether structure matters.
   - **Self-reconstruction check:** Include the target in the donor pool. Coefficient for target should be ≈ 1.0, all others ≈ 0. Error should be ≈ 0. Validates the regression pipeline.
4. **Sparsity analysis:** For each target, sort |w_i| descending. Plot cumulative weight energy vs number of donors. Find natural k where 90% energy is captured.
5. **Subspace dimensionality analysis:** Compute SVD of the full (D × 109) matrix. Plot singular value spectrum. How many singular values capture 95% of total variance?
6. **Classify each of the 109 styles:**
   - "In span" (error < 0.10)
   - "Approximately in span" (0.10 – 0.30)
   - "Partially in span" (0.30 – 0.50)
   - "Not in span" (> 0.50)
7. **Write final report.** Answer the scientific question definitively. Include all plots, tables, and interpretations.

---

## 4. Tensor Extraction Strategy

All 109 LoRAs are stored as `.safetensors` files. Each file contains keys in PEFT format:

```
base_model.model.unet.up_blocks.0.attentions.1.transformer_blocks.{i}.attn{j}.to_{proj}.lora_{A|B}.weight
```

Where:
- `i` ∈ {0..9} — transformer block index
- `j` ∈ {1, 2} — attn1 (self-attention) or attn2 (cross-attention)
- `proj` ∈ {q, k, v, out.0} — projection type
- `A` = down projection, shape (rank, input_dim) = (64, input_dim)
- `B` = up projection, shape (output_dim, rank) = (1280, 64)

The extraction procedure:
1. Open safetensors with `safetensors.torch.load_file()`.
2. Group keys by adapter name (strip `lora_A` / `lora_B` suffix).
3. For each pair: `ΔW = B @ A` → shape (1280, input_dim).
4. Flatten each ΔW and concatenate all 80 into one vector.

This is consistent with how `ip-lora/compute_projectors_from_loras.py` computes `delta_w = B @ A`.

---

## 5. Reproducibility Strategy

| Aspect | Convention |
|--------|-----------|
| **Random seed** | Fixed seed = 42 for all image generation. Fixed seed = 0 for all regression. |
| **LoRA ordering** | Alphabetical sort of style directory names. Target index is deterministic. |
| **Regression library** | `scikit-learn`. Pin version in requirements. |
| **Precision** | ΔW computed in fp32. Storage in fp16. Regression in fp64. |
| **Logging** | JSON log per experiment with all hyperparameters, metrics, timestamps. |
| **Versioning** | All experiment code lives in `experiments/linear_composition/`. Does NOT modify existing Mo-LoRAs code. |

---

## 6. Directory Structure

```
mo-lora/MoLoRAs/
└── experiments/
    └── linear_composition/
        ├── ROADMAP.md
        ├── TODO.md
        ├── README.md
        ├── requirements.txt
        ├── config.yaml
        ├── extract_deltaw.py          # Phase 0: ΔW extraction & validation
        ├── sanity_check.py            # Phase 0: Image generation comparison
        ├── global_reconstruction.py   # Phase 1: Global linear regression
        ├── layerwise_reconstruction.py # Phase 2: Layer-wise regression
        ├── span_analysis.py           # Phase 3: Interpretation & baselines
        ├── utils.py                   # Shared utilities
        ├── results/
        │   ├── phase0/
        │   │   ├── tensor_norms.json
        │   │   └── images/
        │   ├── phase1/
        │   │   ├── ridge_results.json
        │   │   ├── lasso_results.json
        │   │   ├── elasticnet_results.json
        │   │   ├── best_methods.json
        │   │   ├── normalized_results.json
        │   │   ├── summary.md
        │   │   ├── coefficients/
        │   │   └── images/
        │   ├── phase2/
        │   │   ├── groupA_results.json
        │   │   ├── groupB_results.json
        │   │   ├── groupC_results.json
        │   │   ├── per_tensor_results.json
        │   │   ├── summary.md
        │   │   └── images/
        │   └── phase3/
        │       ├── all_targets_results.json
        │       ├── random_donor_baseline.json
        │       ├── random_tensor_baseline.json
        │       ├── sparsity_analysis.json
        │       ├── span_classification.json
        │       ├── final_report.md
        │       └── plots/
        └── logs/
            └── experiment_log.json
```

---

## 7. Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Target style | `target_{NNNN}_{StyleName}` | `target_0042_Impressionism` |
| Donor set | `donors_excl_{NNNN}` | `donors_excl_0042` |
| Coefficient file | `coeffs_{method}_{target}.npy` | `coeffs_ridge_0042.npy` |
| Result image | `{phase}_{target}_{type}.png` | `p1_0042_reconstructed.png` |
| Log entry key | `{ISO_timestamp}_{phase}_{target}` | `2026-02-25T14:30:00_phase1_0042` |

---

## 8. Logging Strategy

Each phase writes a JSON log entry containing:
- Phase ID, target style ID, method name
- All hyperparameters (λ, l1_ratio, normalization, grouping scheme)
- All metrics (relative error, cosine similarity, sparsity, top-k energy)
- Wall-clock time
- Peak memory usage (RSS)
- File paths to all outputs

A master `experiment_log.json` aggregates all entries.

---

## 9. Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| OOM when building D×109 matrix | Use Gram matrix approach (108×108) via streaming dot products |
| SVD numerical instability | fp32 accumulation + CPU LAPACK fallback (already validated in roadmap §27–30) |
| Rank-deficiency / ill-conditioned Gram matrix | Ridge regularization is mandatory; monitor condition number |
| Alpha scaling makes reconstructed LoRA invisible | Use α = 2.0, validated in roadmap §26 |
| CLIP scores misleading (cluster by content, not style) | Do NOT use CLIP for style evaluation; use visual inspection + parameter-space metrics only |
| Cross-term cancellation if A/B operated separately | Always compute ΔW = B @ A first; never mix A and B from different LoRAs |
