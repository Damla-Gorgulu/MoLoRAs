# MoELoRA Inference Experiments — Full Reference

This document is the single source of truth for every inference experiment run in
this project.  It explains **what model was used**, **what prompt was given**,
**what the style input was**, **what settings were applied**, and **what question
each experiment was trying to answer**.

For the live operational log of what is currently working, what failed, and what
to try next, see [experiment_kb.md](./experiment_kb.md).

---

## Glossary

| Term | Meaning |
|------|---------|
| **B-LoRA** | Backbone-LoRA: a fine-tuned LoRA adapter that encodes a single art style into SDXL's style (UNet) pathway.  Each expert in the pool is one B-LoRA. |
| **Expert / LoRA Expert** | One trained B-LoRA file, representing a specific art style (e.g. Baroque).  The full pool has **109 experts**. |
| **Pool** | The set of all active experts that the router samples from at inference time.  During a given run the pool may be the full 109, or a randomly sampled subset. |
| **Router / Routing MLP (v1)** | Small neural network that takes a CLIP image embedding and outputs a weight (`attention`) for each expert in the pool.  Used in v1.0. |
| **LoRARankEncoder (v2)** | Lighter neural network (2.2M params vs 17.3M in v1) that outputs a per-tensor, per-rank weight matrix `A ∈ ℝ^{N × T × r}`.  Used in v2.0 and later. |
| **N** | Number of experts in the current pool (varies per run). |
| **T** | Number of tensor groups — how many distinct weight matrices in the UNet are being modified simultaneously (T = 80 in this project). |
| **r** | LoRA rank.  Each B-LoRA adapter has rank r = 64. |
| **Temperature (τ)** | Controls how "sharp" the routing distribution is.  Low τ (e.g. 0.005) → almost all weight on one expert.  High τ (e.g. 1.0) → weights spread across many experts. |
| **Top-k** | After applying temperature, zero out all but the k highest-weight experts.  `top_k=1` is the oracle (single-best expert).  `top_k=none` uses all experts. |
| **style_alpha (α)** | Linear scaling of the synthesised LoRA delta before injection into SDXL.  `α=1.0` is the default.  `α=0.0` turns off LoRA injection entirely (vanilla SDXL). |
| **Synthesis (legacy / normal mode)** | Compute `ΔW = (Σ A_i W_up_i)(Σ A_j W_down_j)`.  **Buggy** — produces O(N²) cross-terms that cancel the style signal when N is large. |
| **Product-Space Synthesis (`--product_synth`)** | Compute `W_avg = Σ A_i (W_up_i @ W_down_i)`, then decompose with SVD back to rank-r factors.  **Correct** — preserves style signal even with large N. |
| **SDXL** | Stable Diffusion XL — the base generative model.  All images in this project are generated at 1024 × 1024 unless noted. |
| **CLIP** | Contrastive Language-Image Pretraining model (`openai/clip-vit-base-patch32`).  Used to encode the style query image into a 512-d embedding. |
| **WikiArt** | Public dataset of \~80 k paintings organised by style/movement.  Used as training data for the router and as unseen query images in generalization tests. |
| **LDM loss** | Latent Diffusion Model denoising loss — the standard SDXL training objective used in Stage 2. |
| **Entropy regulariser** | Added to Stage 2 loss to prevent routing collapse.  `L_ent = −λ · H(A)`.  Maximises routing entropy so multiple experts stay active. |

---

## Models Trained

### Stage 1 v1.0  (`/scratch/eyavuz21/lora_attention/stage1/`)

| Property | Value |
|----------|-------|
| Architecture | RoutingMLP |
| Parameters | 17,320,896 |
| Input | CLIP ViT-B/32 image embedding (512-d) + pre-computed LoRA feature PCA (480-d) |
| Output | `A ∈ ℝ^{N}` — a single weight per expert (no per-tensor or per-rank distinction) |
| Training dataset | 109 zoo-style reference images (one per expert), not WikiArt |
| Training steps | 10,000 |
| Learning rate | 1e-4 (constant after 500-step warmup) |
| Batch size | 8 |
| Pool size per step | 3–20 randomly sampled experts |
| Loss | MSE between predicted attention and one-hot target (GT expert = 1, others = 0) |
| Final loss | ~0.001 (very low — near-perfect fit on the small 109-image training set) |
| Known limitation | Trained on the same 109 images that define the experts → severe overfitting risk; routing may rely on memorised pixel patterns rather than learned style features |

---

### Stage 2 v1.0  (`/scratch/eyavuz21/lora_attention/stage2/`)

| Property | Value |
|----------|-------|
| Architecture | RoutingMLP (same as Stage 1, frozen encoder) |
| Training steps | 5,000 |
| Learning rate | 5e-5 (constant, 250-step warmup) |
| Mixed precision | fp16 |
| Loss | LDM denoising loss only (no entropy term) |
| Final loss | ~0.547 |
| Purpose | Fine-tune attention weights using the full SDXL diffusion signal so that the synthesised LoRA actually improves image quality, not just routing accuracy |

---

### Stage 1 v2.0  (`/scratch/eyavuz21/lora_attention/stage1_v2/`)

| Property | Value |
|----------|-------|
| Architecture | LoRARankEncoder |
| Parameters | 2,232,320 |
| Input | CLIP ViT-B/32 image embedding (512-d) |
| Output | `A ∈ ℝ^{N × T × r}` — per-tensor per-rank attention (T=80, r=64) |
| Training dataset | WikiArt (~32k samples after per-style cap of 500 images) |
| Training steps | 15,000 |
| Learning rate | 3e-4 with linear warmup (500 steps) then cosine decay |
| Batch size | 8 |
| Pool size per step | 5–20 randomly sampled experts |
| Loss | KL divergence between predicted attention and **soft CLIP-similarity targets** |
| Soft target construction | For each GT expert, look up its row in the 109×109 CLIP similarity matrix; apply softmax with τ=0.3 to make a distribution over the pool |
| Entropy monitoring | Logged only — routing entropy ~2.42 bits (high: routing is almost uniform) |
| Final KL loss | ~0.303 |
| Known limitation | CLIP similarity clusters images by **subject matter** (portraits, landscapes), not artistic style.  A Baroque portrait and a Realism portrait are CLIP-similar, so the loss incorrectly teaches the encoder to treat them as related styles.  This was replaced by one-hot CE in v2.1. |

---

### Stage 2 v2.0  (`/scratch/eyavuz21/lora_attention/stage2_v2/`)

| Property | Value |
|----------|-------|
| Architecture | LoRARankEncoder (Stage 1 v2.0 checkpoint, encoder fine-tuned) |
| Training steps | 8,000 |
| Learning rate | 5e-5 (constant, 200-step warmup) |
| Mixed precision | fp16 |
| Loss | LDM denoising + entropy regulariser |
| Entropy weight λ | Linearly annealed from 0.10 → 0.01 over 8000 steps |
| Final LDM loss | ~0.589 |
| Final entropy term | ~−0.025 |
| Purpose | Same as v1.0 Stage 2 but with the LoRARankEncoder and per-tensor synthesis |

> **Critical note:** Both Stage 1 and Stage 2 v2.0 training used **legacy (buggy) synthesis** by default (`product_space=False`).  The O(N²) cross-term cancellation bug was discovered after training completed.  The resulting checkpoints are therefore used with `--product_synth` at inference time to apply the correct synthesis.

---

### Stage 1 v2.1  (`/scratch/eyavuz21/lora_attention/stage1_v21/`)  — *currently training*

| Property | Value |
|----------|-------|
| Architecture | LoRARankEncoder (same as v2.0) |
| Loss | **One-hot cross-entropy** on the GT style label |
| Soft target construction | None — no CLIP similarity file needed |
| Motivation | Replaces the flawed CLIP-similarity KL targets; enforces that the encoder learns style-discriminative features, not subject-matter features |
| Default synthesis | `product_space=True` — correct product-space SVD synthesis is now the default in code |
| SLURM job | 762449 (submitted 2026-02-23, currently PENDING/RUNNING) |

#### Mini canary: exact-instance routing

We are also running a smaller exact-exemplar Stage 1 canary under
`/scratch/eyavuz21/lora_attention/mini_exact_v1/` to test the real retrieval
problem more directly:

| Property | Value |
|----------|-------|
| Supervision | Exact exemplar image for each LoRA, not just the WikiArt category tag |
| Positives | Style-preserving augmentations of the source exemplar |
| Negatives | Mixed 4-expert pool, including same-category and cross-category distractors |
| Styles | `Baroque`, `Cubism`, `Impressionism`, `Expressionism` |
| Goal | Check whether routing becomes sharper when the label is tied to the actual LoRA source image |
| SLURM job | 1007514 (training), 1007515 (validation, chained after training) |
| Observed result | Validation top-1 = `1.000`, mean GT rank = `1.0`, entropy = `1.3753` |

#### Mini neutral generalization replay

This is the next retrospective benchmark after the exact mini canary.  The
prompt stays neutral so we can check whether the style signal still appears on
held-out singleton styles and on zero-shot styles that do not have a matching
expert in the pool.

| Property | Value |
|----------|-------|
| Checkpoints | `stage1_v21/latest.pt`, `stage2_v22/latest.pt`, `stage2_v23/latest.pt` |
| Prompt | `A detailed painting` |
| Query groups | Held-out singleton styles + zero-shot styles |
| Routing modes | Soft routing and top-1 routing |
| Goal | Check whether the neutral-prompt style effect survives beyond the exact mini canary |
| Submission | `slurm/mini_generalization/submit_neutral_generalization_mini_v1.sh` |

---

## Experiment Index

| # | Folder | Model | Synthesis | Question |
|---|--------|-------|-----------|----------|
| 1 | `inference_s1/` | Stage 1 v1.0 | Legacy | Does v1 routing produce correct-looking outputs? |
| 2 | `inference_s2/` | Stage 2 v1.0 | Legacy | Does Stage 2 fine-tuning improve image quality? |
| 3 | `inference_sweep/` | Stage 1 v1.0 | Legacy | What temperature / top-k / alpha settings work best? |
| 4 | `generalization/` | Stage 1 v1.0 | Legacy | Does the router generalise to unseen images and styles? |
| 5 | `generalization_v2/` | Stage 1 v2.0 | Legacy | Same question repeated with the new LoRARankEncoder |
| 6 | `s1v2_sweep/` | Stage 1 v2.0 | Legacy (buggy) | Sanity sweep before bug was discovered — invalidated |
| 7 | `s2v2_sweep/` | Stage 2 v2.0 | Legacy (buggy) | Same sweep with Stage 2 checkpoint — invalidated |
| 8 | `s1v2_ps_sweep/` | Stage 1 v2.0 | **Product-space ✓** | First correct synthesis: does style transfer now work visually? |
| 9 | `s2v2_ps_sweep/` | Stage 2 v2.0 | **Product-space ✓** | Same with Stage 2 checkpoint |
| 10 | `mini_stage1_category/` | Stage 1 v2.1 mini | Routing-only | Coarse 4-style proxy to test whether the router learns at all |
| 11 | `mini_stage1_exact/` | Stage 1 v2.1 mini | Routing-only | Exact-instance retrieval canary with exemplar-tied supervision |
| 12 | `neutral_generalization_mini/` | v2.1/v2.2/v2.3 replay | Neutral | Does a neutral prompt still surface style transfer on held-out singleton and zero-shot queries? |
| 13 | `neutral_alpha_sweep/` | v2.1/v2.2/v2.3 replay | Neutral + alpha sweep | Does increasing LoRA magnitude recover visible style transfer on old checkpoints? |
| 14 | `mini_generalization_train_eval/` | Stage 1 v2.1 mini | CE routing + neutral replay | Can a compact WikiArt subset train a fresh checkpoint that generalizes under the neutral benchmark? |

---

## Experiment 1 — `inference_s1/` — Stage 1 v1.0 Basic Inference

**Purpose:** Sanity-check that the Stage 1 v1.0 router can synthesise a visually
recognisable stylised image when given a style query image.

**Model:** Stage 1 v1.0 checkpoint (`stage1/latest.pt`) — RoutingMLP.

**Synthesis:** Legacy (normal mode) — `ΔW = (Σ A_i W_up_i)(Σ A_j W_down_j)`.

**Query images:** The canonical zoo reference image for each style
(`/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images/style_XXXX_*/`).
These are the same images the experts were originally trained on → the router
has likely memorised these, so this is an **in-distribution** test.

**Styles tested:**

| Style dir | Generation prompt |
|-----------|------------------|
| `style_0000_Baroque` | `A cat in baroque style [v]` |
| `style_0002_Abstract_Expressionism` | `A landscape in abstract expressionism style [v]` |
| `style_0003_Cubism` | `A portrait in cubism style [v]` |
| `style_0010_Expressionism` | `A forest in expressionism style [v]` |
| `style_0020_Minimalism` | `A room in minimalism style [v]` |

**Common inference settings (all experiments unless listed otherwise):**

| Setting | Value |
|---------|-------|
| Inference steps | 30 |
| Guidance scale | 7.5 |
| Seed | 42 |
| Images per run | 4 |
| Temperature | 0.1 (unless swept) |
| style_alpha | 1.0 (unless swept) |
| Top-k | None (unless swept) |
| Resolution | 1024 × 1024 (SDXL default) |

**Output:** 4 images per style → 20 images total in `inference_s1/`.

**Note on `[v]`:** This is a Textual Inversion placeholder token present in the
original B-LoRA codebase.  When a B-LoRA is active, it binds `[v]` to the style
concept.  When using MoE synthesis, `[v]` is not bound to any single LoRA token
but the synthesised LoRA still activates the style pathway.

---

## Experiment 2 — `inference_s2/` — Stage 2 v1.0 Basic Inference

**Purpose:** Repeat Experiment 1 using the Stage 2 checkpoint to see whether
end-to-end LDM fine-tuning improves visual quality and style fidelity.

**Model:** Stage 2 v1.0 checkpoint (`stage2/latest.pt`) — RoutingMLP fine-tuned
with LDM loss.

**Everything else (styles, prompts, query images, settings) is identical to
Experiment 1.**

**What to look for:** Compared to Experiment 1, Stage 2 images should ideally
show stronger style transfer and fewer artefacts because the routing weights have
been nudged by actual diffusion gradients.

---

## Experiment 3 — `inference_sweep/` — Temperature / Top-k / Alpha Sweep

**Purpose:** Find the best inference hyperparameters for the v1.0 model.  This
is a grid search over temperature, top-k, and style_alpha.  It also establishes
two baselines for comparison: reference B-LoRA (direct single-expert injection)
and vanilla SDXL (no LoRA at all).

**Model:** Stage 1 v1.0 checkpoint (`stage1/latest.pt`).

**Synthesis:** Legacy mode.

**Styles tested (3, for speed):**

| Style dir | Prompt |
|-----------|--------|
| `style_0000_Baroque` | `A cat in baroque style [v]` |
| `style_0003_Cubism` | `A portrait in cubism style [v]` |
| `style_0010_Expressionism` | `A forest in expressionism style [v]` |

**Configurations tested:**

| Folder label | Temperature τ | Top-k | style_alpha α | Notes |
|-------------|---------------|-------|---------------|-------|
| `tau1.0_noTopK` | 1.0 | none | 1.0 | Baseline soft routing — weights spread across all 109 experts |
| `tau0.5_noTopK` | 0.5 | none | 1.0 | Moderate sharpening |
| `tau0.1_noTopK` | 0.1 | none | 1.0 | Sharp routing — most weight on top few experts |
| `tau0.01_noTopK` | 0.01 | none | 1.0 | Very sharp — near-argmax |
| `tau0.1_top1` | 0.1 | 1 | 1.0 | Oracle: only the single best expert is used |
| `tau0.1_top3` | 0.1 | 3 | 1.0 | Top-3 experts only |
| `tau0.1_top5` | 0.1 | 5 | 1.0 | Top-5 experts only |
| `tau1.0_alpha2.0` | 1.0 | none | 2.0 | Higher alpha to compensate for soft-routing dilution |
| `tau0.1_alpha1.5` | 0.1 | none | 1.5 | Sharp routing + boosted alpha |
| `reference_blora` | N/A | N/A | 1.0 | **Reference baseline:** the actual B-LoRA for that style injected directly (no routing at all) |
| `vanilla_sdxl` | any | none | **0.0** | **No-LoRA baseline:** SDXL generates with the text prompt alone — no style adapter injected |

**Output:** 4 images per (style, config) → ~108 images total.

**Interpretation key:**
- If `tau0.01_noTopK` and `tau0.1_top1` produce similar results: the router
  is effectively routing to one expert already — sharpening does not hurt.
- If `reference_blora` looks dramatically better than all MoE configs: the
  synthesis is destroying style information (foreshadowing the cross-term bug).
- `vanilla_sdxl` shows what SDXL "knows" about the style from text alone —
  any MoE result should at minimum beat this in terms of visual style coherence.

---

## Experiment 4 — `generalization/` — v1.0 Routing Generalisation

**Purpose:** Test whether the v1.0 RoutingMLP generalises beyond its 109 training
images.  Three distinct experimental axes:

**Model:** Stage 1 v1.0 checkpoint (`stage1/latest.pt`).

**Synthesis:** Legacy mode.

**Common settings:** τ=0.1, α=1.0, top-k=none, 4 images per run, 30 steps,
guidance 7.5, seed 42.

**Shared prompt for all runs:** `A painting in [v] style`

**Query images:** Fresh WikiArt images (not the zoo reference images).

---

### Exp A — Singleton Held-Out

**Question:** For styles that have exactly one expert in the pool, can the router
reconstruct correct style if that single expert is removed from the pool?

Two conditions:
- **A1 (in-pool):** GT expert is present in the pool.  Expected: router selects
  it confidently.  This is the easy baseline.
- **A2 (held-out):** GT expert is excluded from the pool.  Sibling experts from
  nearby styles must compensate.  This tests knowledge transfer / interpolation.

**Styles and query images:**

| Style label | WikiArt query image |
|-------------|---------------------|
| Baroque | `adriaen-brouwer_a-boor-asleep.jpg` |
| Cubism | `adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg` |
| Fauvism | `abraham-manievich_artist-s-wife-1937.jpg` |
| Northern_Ren (Northern Renaissance) | `albrecht-altdorfer_alpine-landscape-with-church-1522.jpg` |
| Early_Ren (Early Renaissance) | `andrea-del-castagno_crucifixion-1.jpg` |
| High_Ren (High Renaissance) | `andrea-del-sarto_archangel-raphael-…1512.jpg` |
| Color_Field (Color Field Painting) | `ad-reinhardt_abstract-painting-1963.jpg` |

---

### Exp B — Sibling Reconstruction

**Question:** For styles that have **multiple** experts in the pool (e.g. 3
different Impressionism B-LoRAs), does the router cluster its attention onto
all experts of the correct category, or does it scatter randomly?

| Style label | WikiArt query image |
|-------------|---------------------|
| Impressionism | `abdullah-suriosubroto_air-terjun.jpg` |
| Expressionism | `abidin-dino_drawing-pain-1968.jpg` |
| Post_Impressionism | `a.y.-jackson_barns-1926.jpg` |
| Romanticism | `adolphe-joseph-thomas-monticelli_an-evening-at-the-paiva.jpg` |
| Abstract_Expr (Abstract Expressionism) | `aaron-siskind_acolman-1-1955.jpg` |

---

### Exp C — Novel / Zero-Shot Styles

**Question:** For styles that have **no expert** in the pool at all, which experts
does the router recruit?  Do they make intuitive visual sense?

Two sub-groups:

**C1 — Stylistically close to pool styles:**

| Style | WikiArt query image | Nearest pool analogues expected |
|-------|---------------------|--------------------------------|
| Pointillism | `andre-derain_boats-at-collioure-1905.jpg` | Impressionism (both post-Impressionist brush-based) |
| Analytical_Cubism | `albert-gleizes_acrobats-1916.jpg` | Cubism |
| Synthetic_Cubism | `ad-reinhardt_collage-1940.jpg` | Cubism |

**C2 — Stylistically distant from pool:**

| Style | WikiArt query image | Challenge |
|-------|---------------------|-----------|
| Ukiyo-e | `hiroshige_a-bridge-across-a-deep-gorge.jpg` | Japanese woodblock print — no close analogue in Western styles |
| Rococo | `allan-ramsay_charlotte-sophia-…1762.jpg` | Decorative 18th-century French — closest is Baroque |
| Mannerism | `agnolo-bronzino_a-portrait-of-giuliano-…jpg` | Late Renaissance distortion style |
| Action painting | `antonio-palolo_untitled-1992.jpg` | Very gestural — Abstract Expressionism closest |

**Output:** 4 images per sub-condition per style.  Attention `.pt` files also
saved so routing behaviour can be analysed numerically (which experts
received how much weight).

---

## Experiment 5 — `generalization_v2/` — v2.0 Routing Generalisation

**Purpose:** Identical experimental design to Experiment 4, but using the
**Stage 1 v2.0** model (LoRARankEncoder, per-tensor routing) to enable a direct
comparison between v1.0 and v2.0 routing quality.

**Model:** Stage 1 v2.0 checkpoint (`stage1_v2/latest.pt`).

**Synthesis:** Legacy mode (note: the cross-term bug is present here, but since
synthesis quality is not the focus — only routing attention `.pt` files are
analysed — this does not invalidate the experiment).

**Styles, query images, prompts, and settings:** Identical to Experiment 4 (see
above).  The four experiment groups are:

| Folder | Corresponds to |
|--------|---------------|
| `expA_inpool/` | Exp A1 — GT expert present |
| `expA_holdout/` | Exp A2 — GT expert excluded |
| `expB/` | Exp B — Multi-expert sibling routing |
| `expC/` | Exp C — Zero-shot novel styles (C1+C2 combined) |

---

## Experiment 6 — `s1v2_sweep/` — v2.0 Synthesis Sweep (LEGACY / INVALIDATED)

**Purpose:** Sweep temperature and top-k for Stage 1 v2.0 using the v2 inference
script (`inference_v2.py`).

**Model:** Stage 1 v2.0 checkpoint (`stage1_v2/latest.pt`).

**Synthesis:** Legacy (buggy) — `ΔW = (Σ A_i W_up_i)(Σ A_j W_down_j)`.

> **⚠️ This experiment is INVALIDATED.** The O(N²) cross-term cancellation
> bug causes the synthesised LoRA to carry effectively zero style signal when
> N > ~5.  Since the pool size used during inference is 109 (all experts), the
> 108 cross-terms per diagonal term reduce the cosine similarity between the
> synthesised LoRA and the oracle LoRA to ~0.10.  All generated images look
> nearly identical to vanilla SDXL regardless of the style input.

**Configs run (per style, per query source):**

| Folder label | Temperature τ | Top-k |
|-------------|---------------|-------|
| `nm_t0.005_k1` | 0.005 | 1 |
| `nm_t0.005_knone` | 0.005 | none |
| `nm_t0.05_k1` | 0.05 | 1 |
| `nm_t0.05_knone` | 0.05 | none |
| `nm_t0.5_k1` | 0.5 | 1 |
| `nm_t0.5_knone` | 0.5 | none |

Additional configs in `baroque/wikiart/` only (early sweep):

| Folder label | Temperature τ | Top-k | style_alpha |
|-------------|---------------|-------|-------------|
| `t0.005_k1_a1.0` | 0.005 | 1 | 1.0 |
| `t0.005_k5_a1.0` | 0.005 | 5 | 1.0 |
| `t0.005_knone_a1.0` | 0.005 | none | 1.0 |
| `t0.01_*` | 0.01 | various | 1.0 |
| … | … | … | … |

**Styles tested:** Baroque, Cubism, Impressionism, Expressionism.

**Query sources tested:**
- `wikiart/` — fresh WikiArt image (unseen during training)
- `pool/` — canonical zoo reference image for the style

**Why the output folders have both `wikiart/` and `pool/` sub-directories:**
The "pool" image is the image the B-LoRA expert was trained on — using it is an
in-distribution test.  The WikiArt image is out-of-distribution.

---

## Experiment 7 — `s2v2_sweep/` — v2.0 Stage 2 Sweep (LEGACY / INVALIDATED)

**Purpose:** Same sweep as Experiment 6 but with the Stage 2 v2.0 checkpoint.

**Model:** Stage 2 v2.0 checkpoint (`stage2_v2/latest.pt`).

**Synthesis:** Legacy (buggy).

> **⚠️ This experiment is also INVALIDATED** for the same reason as Experiment 6.

**Configs, styles, and query sources:** Identical to Experiment 6.

---

## Experiment 8 — `s1v2_ps_sweep/` — v2.0 Stage 1 Product-Space Sweep ✓

**Purpose:** The first experiment using **correct** product-space synthesis.
Systematically answers four questions:
1. Does the fixed synthesis actually change the output images? (expected: yes)
2. Does sharpening temperature / using top-k improve style fidelity?
3. What does the "average" of all 109 experts look like under a neutral prompt?
4. How does oracle routing (top-k=1) compare to soft routing?

**Model:** Stage 1 v2.0 checkpoint (`stage1_v2/latest.pt`).

**Synthesis:** Product-space (`--product_synth`) — `W_avg = Σ A_i (W_up_i @ W_down_i)`, decomposed via SVD.

**Styles:** Baroque, Cubism, Impressionism, Expressionism.

**Query sources per style:**
- `wikiart/` — fresh WikiArt image (unseen)
- `pool/` — canonical in-pool reference image

**Style-specific prompts and query images:**

| Style | Style prompt | Neutral prompt | WikiArt query | Pool image |
|-------|-------------|----------------|---------------|------------|
| Baroque | `A Baroque painting` | `A painting of a village scene` | `adriaen-brouwer_a-boor-asleep.jpg` | `style_0000_Baroque.jpg` |
| Cubism | `A Cubism painting` | `A painting of a still life` | `adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg` | `style_0003_Cubism.jpg` |
| Impressionism | `An Impressionism painting` | `A painting of a landscape` | `abdullah-suriosubroto_air-terjun.jpg` | `style_0005_Impressionism.jpg` |
| Expressionism | `An Expressionism painting` | `A painting of a figure` | `abidin-dino_drawing-pain-1968.jpg` | `style_0010_Expressionism.jpg` |

**Sub-sweeps:**

### Sweep 1 — Style prompt × τ × top-k  (4 styles × 2 sources × 3 temps × 2 topks = 48 runs)

| Folder pattern | Temperature τ | Top-k | Prompt type |
|----------------|---------------|-------|-------------|
| `ps_t0.005_k1` | 0.005 | 1 | Style prompt (e.g. "A Baroque painting") |
| `ps_t0.005_knone` | 0.005 | none | Style prompt |
| `ps_t0.05_k1` | 0.05 | 1 | Style prompt |
| `ps_t0.05_knone` | 0.05 | none | Style prompt |
| `ps_t0.5_k1` | 0.5 | 1 | Style prompt |
| `ps_t0.5_knone` | 0.5 | none | Style prompt |

`top_k=1` forces single-expert oracle routing.  This should produce the
**cleanest possible** style transfer since only one B-LoRA is active and there
are no cross-terms at all.  If `top_k=1` looks good but `knone` (all experts)
still looks bad, it suggests the routing is mostly correct but the synthesis of
multiple experts is still imperfect.

### Sweep 2 — Neutral prompt × τ   (4 styles × 2 sources × 2 temps = 16 runs)

| Folder pattern | Temperature τ | Prompt type |
|----------------|---------------|-------------|
| `ps_neutral_t0.005` | 0.005 | Neutral (content only, no style word in prompt) |
| `ps_neutral_t0.5` | 0.5 | Neutral |

**Motivation:** When the prompt contains a style word (e.g. "A Baroque
painting"), SDXL's text conditioning alone can produce Baroque-looking results
without any LoRA.  Using a neutral prompt (e.g. "A painting of a village scene")
removes this confound — only the synthesised LoRA can introduce style.

### Sweep 3 — Reference B-LoRA baselines  (4 styles × 2 prompts = 8 runs)

| Folder pattern | What it does |
|----------------|-------------|
| `ref_style_prompt` | Inject the **real** B-LoRA for that style directly; use style prompt — upper bound on style transfer quality |
| `ref_neutral_prompt` | Inject real B-LoRA; use neutral prompt — shows pure LoRA style with no text assist |

### Sweep 4 — Vanilla SDXL baselines  (4 styles × 2 prompts = 8 runs)

| Folder pattern | What it does |
|----------------|-------------|
| `vanilla_style` | No LoRA (α=0.0); use style prompt — shows what SDXL knows from text alone |
| `vanilla_neutral` | No LoRA (α=0.0); use neutral prompt — pure unguided generation |

**Common inference settings for all sub-sweeps:**

| Setting | Value |
|---------|-------|
| style_alpha | 1.0 (except vanilla: 0.0) |
| Inference steps | 30 |
| Guidance scale | 7.5 |
| Seed | 42 |
| Images per run | **1** (reduced for speed — 80 total runs) |

---

## Experiment 9 — `s2v2_ps_sweep/` — v2.0 Stage 2 Product-Space Sweep ✓

**Purpose:** Repeat Experiment 8 using the Stage 2 v2.0 checkpoint to see
whether the LDM-fine-tuned routing produces better style fidelity than Stage 1 alone.

**Model:** Stage 2 v2.0 checkpoint (`stage2_v2/latest.pt`).

**Synthesis:** Product-space (`--product_synth`).

**Everything else (styles, prompts, query images, sweep structure, settings) is
identical to Experiment 8.**

The only difference is the checkpoint used.  Comparing Experiment 8 vs 9
directly isolates the effect of Stage 2 LDM training.

---

## Cross-Experiment Comparison Guide

To compare experiments fairly, use matching sub-conditions:

| To compare | Use |
|-----------|-----|
| v1.0 vs v2.0 routing quality | Exp 4 `generalization/` vs Exp 5 `generalization_v2/` (same images, same settings) |
| v1.0 sweep quality | Exp 3 `inference_sweep/` (9 configs × 3 styles) |
| Stage 1 vs Stage 2 benefit | Exp 8 vs Exp 9 (product-space; same styles+prompts) |
| Correct vs buggy synthesis | Exp 8/9 (`s1v2_ps_sweep/`) vs Exp 6/7 (`s1v2_sweep/`) |
| MoE vs real B-LoRA | `ps_t0.005_k1` vs `ref_style_prompt` within Exp 8 or 9 |
| Style prompt vs neutral prompt | Any `ps_t*` vs `ps_neutral_t*` in Exp 8 or 9 |
| Oracle (single expert) vs soft routing | `ps_t0.005_k1` vs `ps_t0.005_knone` in Exp 8 or 9 |
| LoRA vs no LoRA | Any `ps_t*` vs `vanilla_style` in Exp 8 or 9 |

---

## Known Bugs and Their Impact on Experiments

### Bug: O(N²) Cross-Term Cancellation in Legacy Synthesis

**Affected experiments:** 1, 2, 3, 4, 5, 6, 7.

**What it is:** The legacy synthesis formula
`ΔW = (Σᵢ Aᵢ Wᵤₚᵢ)(Σⱼ Aⱼ Wᵈₒ wₙⱼ )`
expands into N diagonal terms (the intended style signal) and N²−N cross-terms.
With N=109: 109 signal terms vs 11,772 cross-terms.  Each cross-term is
`Aᵢ Aⱼ Wᵤₚᵢ Wᵈₒwₙⱼ` for i≠j — a product of LoRA weights from two different
styles that carries no coherent style information.

**Measured impact (v2.0, uniform routing, N=109):**
- cosine similarity between oracle (correct) LoRA and synthesised LoRA: **0.10**
- The synthesised LoRA norm is only **6.3%** of the oracle magnitude

**Why v1.0 (exp 1–4) partially worked:** v1.0 routing tends to assign sharp
weights (near one-hot) because it was trained on only 109 images with a direct
MSE target.  When almost all weight is on one expert (near top-k=1), there are
no meaningful cross-terms and the synthesis is approximately correct.

**Why v2.0 (exp 5–7) failed completely:** v2.0 routing was trained with entropy
regularisation and KL-soft targets, causing high-entropy (near-uniform) routing
across all 109 experts.  With uniform routing, every cross-term is equally
weighted and the cancellation is maximal.

**Fix:** Product-space synthesis (Exp 8, 9) — see §8 and §9 above.

---

## v2.3: Three-Stage Training Approach

To address the training instability observed in v2.2 (flat LDM loss with no improvement), 
v2.3 introduces a three-stage training approach:

**Stage 1 - Warmup (steps 0-2000): LDM Loss Only**
- Pure latent diffusion loss to learn meaningful routing for image quality
- No load-balancing loss to avoid conflicting objectives early
- Temperature: τ=1.0 → 2.0 (increase exploration/diversity)

**Stage 2 - Balance (steps 2000-6000): LDM + LB Loss**
- Add Switch-style load-balancing loss to encourage expert utilization
- Temperature: τ=2.0 → 2.0 (maintain diversity)
- Lambda: ramps up from 0 → 0.1

**Stage 3 - Refine (steps 6000-8000): LDM + smaller LB Loss**
- Reduce LB pressure for final refinement
- Temperature: τ=2.0 → 0.3 (sharpen routing for exploitation)
- Lambda: ramps down from 0.1 → 0.01

**Key Improvements:**
1. **SVD on GPU** - Fixed `_synthesise_product_space` to maintain gradient flow
2. **Separated objectives** - LDM learns routing quality first, LB adds balance later
3. **Smart temperature scheduling** - Exploration → diversity → exploitation
4. **Reduced final lambda** - 0.01 instead of 0.05 to prevent over-regularization

This approach allows the encoder to first learn how to route for image quality 
(Stage 1), then balance expert usage (Stage 2), and finally refine to sharp 
routing (Stage 3).

---

## Experiment 14 — `expA_inpool_next` Follow-Up and Identity Verdict

**Run date:** 2026-04-26  
**Job:** `1031909`  
**Output:** `/scratch/eyavuz21/lora_attention/diagnostics/expA_inpool_next_20260426_162640`

### Purpose

After the first `mini_generalization_v2` replay looked partly encouraging on
top-1 routing, this follow-up asked a stricter question:

**Do the generated outputs actually match the query image's style identity, or
do they only show a generic style/category effect?**

### Setup

For each query style (`Baroque`, `Cubism`, `Fauvism`), generate a side-by-side grid:

- query image
- base SDXL (`A detailed painting`)
- direct reference B-LoRA
- synthesized top-1 MoE LoRA
- synthesized top-1 MoE LoRA with norm matching

Checkpoint used:
- `mini_generalization_v2/stage1_train/latest.pt`

### Visual Verdict

| Style | Verdict |
|------|---------|
| Baroque | A visible Baroque-like effect appears, but it does not faithfully match the query painting. |
| Cubism | Does not match the expected top-1/query style; outputs drift to a different look. |
| Fauvism | Norm-matched synthesis is somewhat closer, but still not exact; other outputs are black-and-white or unrelated. |

### Interpretation

This experiment rules out the comforting version of the story.

- The system can change the image, so synthesis/injection is not completely dead.
- Direct reference B-LoRA remains stronger than synthesized MoE outputs.
- But the current pipeline does **not** preserve query-style identity reliably.

So the present MoE setup should be treated as, at best, **style-category
transfer**, not faithful query-specific style matching.

### Consequence

Future benchmarks should not stop at:

- top-1 expert rank
- visible stylization

They must answer:

- does the output match the query image's style identity?

That is why the next benchmark is a tiny direct-vs-MoE identity check under a
shared dog prompt family, using the same three query styles and comparing:

- base SDXL
- direct reference B-LoRA
- MoE top-1
- MoE norm-matched top-1
