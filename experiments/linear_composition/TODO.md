# TODO LIST — LoRA Linear Composition Experiment

All tasks are listed in execution order with dependencies, expected outputs, and checkpoints.
A task may not begin until all tasks it depends on are marked complete.

---

## Infrastructure Tasks (do first, in parallel)

| # | Task | Depends on | Expected Output | Checkpoint |
|---|------|------------|-----------------|------------|
| I.1 | Create directory structure as specified in ROADMAP.md section 6. Create all `results/`, `logs/`, and `plots/` subdirectories. | — | All directories exist | `ls` confirms full tree |
| I.2 | Write `config.yaml` with all hyperparameters: expert pool path, base model ID, prompt, seed, alpha, Ridge/Lasso/ElasticNet alpha grids, l1_ratio grid, grouping definitions, precision settings, output paths. | — | `config.yaml` parseable with PyYAML | `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` succeeds |
| I.3 | Write `requirements.txt` with pinned versions: torch, safetensors, diffusers==0.25.0, scikit-learn, numpy, matplotlib, pyyaml, tqdm. | — | `requirements.txt` | `pip install -r requirements.txt` succeeds in B-LoRA env |
| I.4 | Verify LoRA pool path. Confirm `repos/B-LoRA/blora_zoo/bloras/` exists and contains exactly 109 subdirectories, each with a `pytorch_lora_weights.safetensors` file. Log any missing or malformed entries. | — | Log file confirming count and path | Count = 109, no missing safetensors |

---

## PHASE 0 — Sanity Check

| # | Task | Depends on | Expected Output | Checkpoint |
|---|------|------------|-----------------|------------|
| 0.1 | List all 109 LoRA directories. Sort alphabetically. Assign integer index 0–108. Save as `style_index.json` mapping index → directory name → full path. | I.4 | `style_index.json` | File has exactly 109 entries |
| 0.2 | Load one LoRA safetensors file (index 0). Print all tensor key names to log. Count unique adapter names (should be 80). Count lora_A and lora_B keys separately (should each be 80). | 0.1 | Printed key list in `logs/experiment_log.json` | Exactly 80 adapter pairs, 160 total keys |
| 0.3 | Parse tensor keys into structured format per adapter: `(block_idx, attn_type, projection)`. Confirm 10 blocks × 2 attn types × 4 projections = 80 unique adapters. Save parsed map as `tensor_key_map.json`. | 0.2 | `tensor_key_map.json` with 80 entries | All 80 entries have valid (block_idx, attn_type, proj) fields |
| 0.4 | For the test LoRA (index 0): compute `ΔW = B @ A` for all 80 adapters. Record the shape and Frobenius norm of each ΔW. Save as `results/phase0/tensor_norms.json`. | 0.3 | `tensor_norms.json` with 80 entries | All 80 norms > 0; shapes match expected (1280×1280 for attn1, 1280×2048 for attn2) |
| 0.5 | Flatten all 80 ΔW tensors and concatenate into one vector. Log total dimensionality D. Confirm D matches expected value based on known tensor shapes. | 0.4 | Log entry with D value | D is consistent across repeated loads of the same LoRA |
| 0.6 | Write `utils.py` with the following functions: `load_lora_deltaw(path) → dict[str, Tensor]`, `flatten_deltaw(deltaw_dict, key_order) → Tensor`, `unflatten_deltaw(flat_vec, key_order, shapes) → dict[str, Tensor]`, `inject_deltaw(pipe, deltaw_dict, alpha) → None`. Test each function on the test LoRA. | 0.3, 0.4, 0.5 | `utils.py` | All 4 functions return correct types and shapes; round-trip flatten→unflatten is lossless |
| 0.7 | Generate **Image 1** (base): SDXL base model, no LoRA, prompt = "A painting of a riverside village at sunset", seed = 42, steps = 30, guidance = 7.5. Save as `results/phase0/images/p0_base.png`. | I.2 | `p0_base.png` | Image saved, pipeline runs without error |
| 0.8 | Generate **Image 2** (direct merge): Same prompt/seed + target LoRA (index 0) injected via direct weight merge `W += α·ΔW` at α = 2.0. Save as `results/phase0/images/p0_target_merge.png`. | 0.6, 0.7 | `p0_target_merge.png` | Image saved |
| 0.9 | Generate **Image 3** (API): Same prompt/seed + same LoRA injected via `pipe.load_lora_weights()` at scale = 2.0. Save as `results/phase0/images/p0_target_api.png`. | 0.7 | `p0_target_api.png` | Image saved |
| 0.10 | Compare Images 2 and 3. Compute pixel-wise MSE. Log result. Compare both visually against Image 1. | 0.8, 0.9 | MSE value in log | MSE between Image 2 and Image 3 < 1e-4; Images 2/3 are visibly different from Image 1 |
| 0.11 | Write Phase 0 log entry with: tensor count, norm statistics (min, max, mean, std), D value, injection equivalence result. Append to `logs/experiment_log.json`. | 0.4–0.10 | Log entry in `experiment_log.json` | Entry is valid JSON with all required fields |

---

## PHASE 1 — Global Linear Reconstruction

| # | Task | Depends on | Expected Output | Checkpoint |
|---|------|------------|-----------------|------------|
| 1.1 | Load all 109 LoRAs. For each: compute ΔW = B @ A for all 80 adapters, flatten using the same key order established in task 0.3, concatenate into one vector. Store all 109 vectors as columns in a matrix of shape (D × 109) saved to `all_deltaw_matrix.pt` (fp16). Log peak memory usage and wall time. | 0.6, 0.1 | `all_deltaw_matrix.pt` | Shape is (D, 109); file size ~34 GB in fp16 |
| 1.2 | Estimate available RAM and VRAM. If total RAM < 64 GB: switch to Gram matrix approach. Document decision in log. If >= 64 GB: direct approach is available. Log: "direct" or "gram". | 1.1 | Decision logged | Choice documented in `logs/experiment_log.json` |
| 1.3 | **[If Gram matrix approach]** For each of the 10 target styles: compute `G = X^T·X` (108×108) and `b = X^T·x_target` (108×1) via streaming dot products without holding full D×108 matrix in memory simultaneously. Save `G` and `b` per target. | 1.2, 1.4 | `gram_{target_id}.npy` and `b_{target_id}.npy` for each target | G is symmetric (verify `‖G - G^T‖ < 1e-6`), shape (108, 108) |
| 1.4 | Select 10 representative target styles spanning different artistic movements (e.g., Impressionism, Cubism, Abstraction, Realism, Baroque, Romanticism, Expressionism, Surrealism, Pointillism, Art Nouveau). Record chosen indices in `target_selection.json`. | 0.1 | `target_selection.json` | 10 distinct indices, one per movement |
| 1.5 | **Self-reconstruction sanity check.** For one target (index 0), include it in the donor pool (all 109 LoRAs). Solve Ridge regression (α = 0.01). Inspect: coefficient for target LoRA should be ≈ 1.0, all others ≈ 0. Relative error should be ≈ 0. Log results. | 1.1 or 1.3, 0.6 | Log entry with coefficients and error | Target coefficient ∈ [0.95, 1.05]; all others < 0.05; relative error < 0.01 |
| 1.6 | **Run Ridge regression** for all 10 targets × 4 alpha values {0.01, 0.1, 1.0, 10.0} = 40 experiments. For each: record relative error, cosine similarity, coefficient vector, sparsity, wall time. Save all results to `results/phase1/ridge_results.json`. | 1.1 or 1.3, 1.4 | `ridge_results.json` with 40 entries | All entries have complete metric fields |
| 1.7 | **Run Lasso regression** for all 10 targets × 3 alpha values {0.001, 0.01, 0.1} = 30 experiments. For each: record same metrics plus number of nonzero coefficients (|w_i| > 1e-4). Save to `results/phase1/lasso_results.json`. | 1.1 or 1.3, 1.4 | `lasso_results.json` with 30 entries | All entries complete |
| 1.8 | **Run ElasticNet** for all 10 targets × l1_ratio ∈ {0.1, 0.5, 0.9} × alpha ∈ {0.01, 0.1} = 60 experiments. Record same metrics. Save to `results/phase1/elasticnet_results.json`. | 1.1 or 1.3, 1.4 | `elasticnet_results.json` with 60 entries | All entries complete |
| 1.9 | For each of the 10 target styles: select the best (method, hyperparameter) combination based on lowest relative reconstruction error. Record in `results/phase1/best_methods.json`. | 1.6, 1.7, 1.8 | `best_methods.json` with 10 entries | One best method per target |
| 1.10 | **Normalization ablation.** Normalize each LoRA's full flat vector to unit norm before regression. Repeat tasks 1.6–1.8 with normalized vectors. Record metrics. Compare normalized vs unnormalized results. Save to `results/phase1/normalized_results.json`. | 1.1 or 1.3, 1.4 | `normalized_results.json` | Comparison table produced |
| 1.11 | **Reconstruct ΔW** for 3 targets: best-performing, median-performing, worst-performing (by relative error). Use best method from task 1.9. Unflatten flat coefficient-weighted vector back to per-tensor ΔW dict using `unflatten_deltaw`. | 1.9, 0.6 | 3 reconstructed ΔW dicts in memory / on disk | Shapes of reconstructed tensors match originals |
| 1.12 | **Generate comparison images** for the 3 targets from task 1.11. For each target: generate base image, target LoRA image, reconstructed LoRA image. Same prompt/seed as Phase 0. Save as `results/phase1/images/p1_{id}_base.png`, `p1_{id}_target.png`, `p1_{id}_reconstructed.png`. | 1.11, 0.6 | 9 images total (3 targets × 3 types) | All 9 images saved |
| 1.13 | Write Phase 1 summary. Include: table of all results sorted by error, best method per target, error distribution statistics, preliminary conclusion on global span membership. Save to `results/phase1/summary.md`. | 1.6–1.12 | `results/phase1/summary.md` | File exists and contains all tables |

---

## PHASE 2 — Layer-wise Reconstruction

| # | Task | Depends on | Expected Output | Checkpoint |
|---|------|------------|-----------------|------------|
| 2.1 | Define 3 tensor grouping schemes in `config.yaml` (if not already done): (A) attn1 vs attn2, (B) early/mid/late × attn1/attn2 (6 groups), (C) by projection type: q, k, v, out. Record group sizes (number of tensors per group). Save `tensor_groups.json`. | 0.3, I.2 | `tensor_groups.json` with 3 schemes, each listing group IDs and member tensor keys | All tensor keys are covered in each scheme with no overlap |
| 2.2 | **Grouping A — extract sub-matrices.** For all 109 LoRAs: extract and flatten ΔW tensors belonging to G1 (attn1) and G2 (attn2) separately. Build 2 donor sub-matrices: shape (D_G1 × 109) and (D_G2 × 109). | 1.1, 2.1 | Two sub-matrices stored (fp16) | Shapes correct: D_G1 + D_G2 = D |
| 2.3 | **Grouping A — solve per-group Ridge regression** for all 10 targets using best alpha from Phase 1. For each target: solve independently for G1 and G2. Record per-group relative error, cosine similarity, and coefficient vectors. Save to `results/phase2/groupA_results.json`. | 2.2, 1.9, 1.4 | `groupA_results.json` | 10 targets × 2 groups = 20 result entries |
| 2.4 | **Grouping A — reconstruct full ΔW** by combining G1 and G2 solutions. Compute overall relative error and cosine similarity. Compare vs Phase 1 global result for each target. Save comparison to `results/phase2/groupA_comparison.json`. | 2.3 | `groupA_comparison.json` | Improvement (or not) vs Phase 1 quantified per target |
| 2.5 | **Grouping B — extract sub-matrices.** For all 109 LoRAs: extract ΔW tensors for each of the 6 groups (early/mid/late × attn1/attn2). Build 6 donor sub-matrices. | 1.1, 2.1 | 6 sub-matrices stored | Shapes sum to D |
| 2.6 | **Grouping B — solve per-group regression** for all 10 targets using best alpha. Record per-group and combined metrics. Save to `results/phase2/groupB_results.json`. | 2.5, 1.9, 1.4 | `groupB_results.json` | 10 targets × 6 groups = 60 result entries |
| 2.7 | **Grouping B — reconstruct full ΔW** and compare vs Phase 1 and Grouping A. Save `results/phase2/groupB_comparison.json`. | 2.6 | `groupB_comparison.json` | Comparison table shows trend |
| 2.8 | **Grouping C — extract sub-matrices** by projection type (to_q, to_k, to_v, to_out.0) across all blocks and attention types. Build 4 sub-matrices. | 1.1, 2.1 | 4 sub-matrices stored | Shapes sum to D |
| 2.9 | **Grouping C — solve per-group regression** for all 10 targets. Record per-group and combined metrics. Save to `results/phase2/groupC_results.json`. | 2.8, 1.9, 1.4 | `groupC_results.json` | 10 targets × 4 groups = 40 result entries |
| 2.10 | **Grouping C — reconstruct full ΔW** and compare vs all prior schemes. Save `results/phase2/groupC_comparison.json`. | 2.9 | `groupC_comparison.json` | Comparison table complete |
| 2.11 | **Per-tensor regression (upper bound).** Solve independently for each of the 80 individual tensors across all 10 targets. This gives the upper bound on layer-wise decomposability. Record per-tensor error. Save to `results/phase2/per_tensor_results.json`. | 1.1, 1.4 | `per_tensor_results.json` (10 targets × 80 tensors = 800 entries) | All 800 entries complete |
| 2.12 | **Generate comparison images** for the best and worst targets under the best grouping scheme. Same prompt/seed. Save to `results/phase2/images/`. | 2.4, 2.7, 2.10, 0.6 | At least 6 images (2 targets × 3 types) | All images saved |
| 2.13 | **Analyze per-group error patterns.** Which groups (attn1 vs attn2, early vs late, q/k vs v/out) show consistently low vs high reconstruction error across targets? Write interpretation of findings. | 2.3–2.11 | Analysis section in summary | Interpretation relates groups to known B-LoRA semantics |
| 2.14 | Write Phase 2 summary. Include: comparison table across all grouping schemes vs global, per-group error heatmap description, best grouping scheme, interpretation. Save to `results/phase2/summary.md`. | 2.3–2.13 | `results/phase2/summary.md` | File exists and contains full comparison |

---

## PHASE 3 — Span Membership Interpretation

| # | Task | Depends on | Expected Output | Checkpoint |
|---|------|------------|-----------------|------------|
| 3.1 | **Full leave-one-out sweep.** Using the best method and best grouping scheme from Phases 1–2: run reconstruction for all 109 target styles. For each: record relative error, cosine similarity, coefficient vector, sparsity. Save to `results/phase3/all_targets_results.json`. | 1.9, 2.14, 0.6 | `all_targets_results.json` with 109 entries | All 109 entries complete with no missing values |
| 3.2 | **Distribution statistics.** From the 109 results: compute mean, std, min, max, 25th, 50th, 75th percentiles of relative error. Compute same for cosine similarity. Generate histogram plots of both distributions. Save plots to `results/phase3/plots/`. | 3.1 | 2 histogram plots + statistics in JSON | Plots saved as PNG; statistics in `span_analysis.json` |
| 3.3 | **Random donor baseline.** For 10 targets: solve regression using k ∈ {5, 10, 20, 50, 108} randomly selected donors. Repeat 5 independent random draws per k. Record mean ± std error per k. Save to `results/phase3/random_donor_baseline.json`. Generate error-vs-k plot. | 1.1, 1.4 | `random_donor_baseline.json` + plot | Plot shows error trend as k increases |
| 3.4 | **Random tensor baseline.** For 10 targets: replace all 108 donor ΔW vectors with random Gaussian vectors scaled to the same per-LoRA norm. Solve Ridge regression. Record errors. Save to `results/phase3/random_tensor_baseline.json`. | 1.1, 1.4 | `random_tensor_baseline.json` | Random baseline error is substantially higher than real donor error (validates that LoRA structure matters) |
| 3.5 | **Sparsity analysis.** For each of 109 targets: sort coefficient magnitudes |w_i| descending. Compute cumulative weight energy. Find k* = minimum k such that top-k donors capture ≥ 90% of total |w| mass. Record k* per target. Save to `results/phase3/sparsity_analysis.json`. Generate boxplot of k* distribution. | 3.1 | `sparsity_analysis.json` + boxplot | k* values computed for all 109 targets |
| 3.6 | **Hub donor analysis.** For each of the 109 leave-one-out experiments: identify which donors had |w_i| > 0.05. Count how many times each of the 109 LoRAs appears as a significant donor. Build co-occurrence matrix (109×109). Generate heatmap. Identify top-10 "hub" styles. | 3.1 | Co-occurrence heatmap + `hub_styles.json` | Heatmap saved; hub styles listed with frequency counts |
| 3.7 | **Subspace dimensionality analysis.** Compute SVD of the full (D × 109) ΔW matrix. Plot singular value spectrum (all 109 values). Compute cumulative explained variance. Find rank r* such that top-r* singular values capture 95% of total variance (Frobenius energy). Save plot and r* value. | 1.1 | Singular value plot + `subspace_rank.json` with r* | Plot saved; r* is a well-defined integer < 109 |
| 3.8 | **Span classification.** Using the 109 relative errors from task 3.1: classify each style as: "In span" (< 0.10), "Approximately in span" (0.10–0.30), "Partially in span" (0.30–0.50), "Not in span" (> 0.50). Record count per class. Save to `results/phase3/span_classification.json`. | 3.1 | `span_classification.json` with 109 entries and a summary count | 4 classes, all 109 entries assigned |
| 3.9 | **Write final report.** Consolidate all findings. Sections: (1) Executive summary answering the scientific question, (2) Phase 0 results, (3) Phase 1 results with method comparison, (4) Phase 2 results with grouping comparison, (5) Phase 3 results with all baselines, (6) Conclusion: is linear composition feasible for these 109 style LoRAs? Save to `results/phase3/final_report.md`. | 3.1–3.8, 1.13, 2.14 | `results/phase3/final_report.md` | File contains all 6 sections; references all result files by path |
