# MoELoRA v2 — Speaker Notes
> Bullet-point reference. ~25–30 min total. Bold = key point to land.

---

## Slide 1 — Title
- Introduce yourselves and the project name: **MoELoRA v2**
- Frame it: per-tensor mixture-of-experts routing over a B-LoRA style zoo
- Progress report, February 2026, VALAR Lab

---

## Slide 2 — Outline
- Walk through the 7 sections briefly
- Mention appendix is there for extra detail
- Ask audience to hold questions until end

---

## Slide 3 — Motivation & B-LoRA Background
- **The problem**: given any image, zero-shot style transfer via SDXL
- We have 109 B-LoRA experts, each specialised in one WikiArt style
- Real queries have no clean style labels → can't just pick one expert
- **The routing question**: which experts? how much? blended how?
- B-LoRA targets a single attention block: `up_blocks.0.attentions.1` (the SDXL style block)
- Rank 64, 80 adapter pairs (60 × 1280-dim, 20 × 2048-dim inputs)
- Injection: W' = W + α·W_up·W_down (frozen UNet, only LoRA changes)

---

## Slide 4 — System Overview
- Walk left to right through the pipeline
- Query → frozen CLIP → 512-d embedding q
- q + LoRA weight pool → **LoRARankEncoder** (only trainable part, 2.2M params)
- Encoder produces per-rank attention keys K for every expert and adapter
- **Per-tensor cross-expert attention** A: shape N×80×64
- **Product-space synthesis**: weighted sum in full weight space → tSVD → inject into SDXL
- Everything except the encoder is frozen

---

## Slide 5 — LoRARankEncoder φ(·)
- Key idea: **point-wise encoding over rank positions** (not the full adapter at once)
- Each row of W_down → small MLP (Linear → LayerNorm → GELU → Linear + L2-norm) → 512-d key
- Two separate heads for the two adapter input sizes (1280-d, 2048-d)
- Gives **per-rank routing**: rank position j can prefer different expert than j+1
- Batching: implemented as Conv1d → only 2 kernel launches for all N×T tensors
- **2.23M params** vs 17.3M in v1.0 — 7× more efficient

---

## Slide 6 — Per-Tensor Cross-Expert Attention
- Dot-product: H = q · K / √512, then softmax over expert dimension
- Result A: N × 80 × 64 — each adapter pair × each rank position has its own expert distribution
- **Per-tensor**: adapter pair t can route differently from t+1 (important!)
- **Temperature τ**: 1.0 at training (smooth gradients), 0.005 at inference (sharp, decisive)
- Optional top-k: set k=1 for hard single-expert routing, None for soft blend

---

## Slide 7 — Cross-Term Problem (Naïve LoRA Averaging)
- **Slow down here — most important insight of the project**
- v1.0 bug: average W_up and W_down separately, then multiply
- Expanding the product: diagonal terms (the style signal you want) + **O(N²) cross-terms**
- N=109 → over 10,000 cross-terms, all unrelated style-component products → cancel each other
- Effect: **all outputs look identical regardless of routing** (routing mathematically irrelevant)
- Fix (v2.0): compute full W_bar = Σ A_i · W_up_i · W_down_i, then tSVD to recover LoRA factors
- Validated: oracle cosine similarity = **0.9998**

---

## Slide 8 — Product-Space Synthesis: Engineering Details
- The math is correct; the **fp16 implementation hit three separate crash modes**
- Bug 1: **fp16 accumulation overflow** summing 109 matrices → fix: accumulate in fp32
- Bug 2: **cuSOLVER silent hang** on ill-conditioned matrices (no exception raised!) → fix: CPU LAPACK SVD
- Bug 3: **fp16 softmax NaN** poisons the entire accumulation → fix: nan_to_num before SVD
- Runtime: 80 SVDs/step ≈ 400 ms ≈ 8% overhead on a 5–8 s SDXL forward pass
- Mention: 5 separate Stage 2 training attempts before all bugs resolved
- **See next slide for SVD diagram and explanation**

---

## Slide 9 — Why SVD? From Full Matrix to LoRA Format
- **This slide answers: why can't we just inject $\bar{W}_t$ directly?**
- $\bar{W}_t$ is a full $(1280 \times 1280)$ matrix (full-rank); SDXL's LoRA injection only accepts two small factors $(W^\text{up}, W^\text{down})$ at rank 64
- SVD decomposes $\bar{W}_t = U \cdot S \cdot V^T$ — singular values $S$ are sorted descending (most energy first)
- **Truncation**: keep only top 64 singular triplets — best rank-64 approximation (matches B-LoRA's own rank, so no extra approximation loss)
- **Split $\sqrt{S_r}$** symmetrically: gives $W^\text{up} = U_r \sqrt{S_r}$, $W^\text{down} = \sqrt{S_r} V_r^T$ — balanced norms
- Fidelity: oracle test gives cosine similarity = **0.9998** → essentially lossless for style-relevant directions
- Cost: ~400 ms total for 80 adapter pairs per step (~8% overhead), runs on **CPU LAPACK** to avoid GPU cuSOLVER hang

---

## Slide 10 — Dataset
- WikiArt: 31,949 images, 20 styles, capped at 500/style for balance
- Expert pool: 109 B-LoRA checkpoints; LoRA features pre-cached (5.8 GB) for speed
- Stage 2 sampling subtlety: sample N experts **excluding the GT expert** → encoder must find style signal from negatives

---

## Slide 10 — Two-Stage Training Pipeline
- **Stage 1**: ~2h, no SDXL, CrossEntropy loss only → teach encoder which expert to activate
- **Stage 2**: ~24h, full SDXL forward every step → LDM loss + entropy regulariser
- Load Stage 1 checkpoint, only encoder trains; everything else frozen
- The two stages are ordered: S1 first, S2 continues from that checkpoint

---

## Slide 11 — Stage 1: Cross-Entropy Routing Loss
- Average A over rank dimension → per-adapter routing distribution
- CE loss: for each of 80 adapter pairs, classify which of N experts is correct
- **v2.0 KL vs v2.1 CE**:
  - v2.0: soft targets from CLIP similarity — CLIP clusters by subject, not style → noisy
  - v2.1: hard one-hot GT from WikiArt labels → unambiguous, cleaner gradients
- Training curves show both converge; CE gives more decisive routing

---

## Slide 12 — Stage 2: LDM + Entropy Regularisation
- LDM loss: predict noise added to latent, conditioned on synthesised LoRA in UNet
- Entropy regulariser: -λ·H(A) — prevents routing collapse to always-same-expert
- λ decays 0.1 → 0.01 over training (explore early, consolidate later)
- Loss oscillates: normal (stochastic batches + diffusion timestep randomness)

---

## Slide 14 — Alpha Calibration Problem
- **All early inference results were identical** even after fixing cross-terms!
- Root cause: synthesised LoRA norm ≈ 16 vs real B-LoRA norm ≈ 50 → at α=1.0 only 32% signal
- SDXL ignores the LoRA injection because it's too small relative to the pretrained weights
- Slide shows 6 individual images: vanilla → α=1.0 → **α=2.0 (sweet spot)** → α=3.0 oracle (content leakage) → α=5.0 (distortion) → Ref B-LoRA
- **α=3.0 oracle image**: GT expert with high alpha → query content starts bleeding into generation (content leakage, not just noise)
- Distinguish clearly: **content leakage** (query image content appears in output) vs **distortion** (artifacts from excessive activation)
- **Fix: always use α = 2.0 at inference**

---

## Slide 15 — Routing Analysis: Non-Uniform Heatmap (Baroque)
- Heatmap rows = 80 adapter pairs, cols = N pooled experts
- Non-uniform colour means **routing is working** (not ignoring the query)
- Top-5 for Baroque: Northern Renaissance (1), Romanticism (2), Baroque (3)
- All three share dark palette, chiaroscuro — semantically adjacent, not wrong routing
- Shows encoder learned style-space geometry without explicit supervision on expert distances

---

## Slide 15 — Inference Sweep Setup
- 4 styles × 2 query modes × 8 routing configs × 5 images = 80 images per checkpoint
- Prompts: "A Baroque painting", "A Cubism painting", "An Impressionism painting", "An Expressionism painting"
- Note: we used free-text prompts (not "A [v]") — this is the **prompt mismatch issue** (later slide)
- 3 of 4 checkpoints fully swept; S2 v2.1 still training

---

## Slide 16 — Inference Results: S1 v2.1 — Baroque
- Left: WikiArt Baroque query. Middle: MoE output. Right: Top-1 expert (Northern Renaissance)
- Top-1 is adjacent, not exact — but shares visual character (dark dramatic tone)
- Baroque appears at rank 3
- Stage 1 only (no denoising training) — already shows style injection working

---

## Slide 17 — Inference Results: S2 v2.0 — Baroque
- Same style, Stage 2 (LDM fine-tuned) checkpoint
- Top-1 shifts to **Early Renaissance** — different adjacent cluster than S1
- LDM training shifts the routing distribution
- We can't do full S1 vs S2 quality comparison yet (waiting for S2 v2.1)

---

## Slide 19 — Inference Results: S1 v2.1 — Cubism
- Prompt: "A Cubism painting"
- Top-1: **Naïve Art / Primitivism** — shares geometric flatness, bold simplified forms with Cubism
- Cubism itself at rank 5 (not top-1, but in top-5)
- Another semantically adjacent result — not random

---

## Slide 20 — Inference Results: S1 v2.1 — Impressionism
- Prompt: "An Impressionism painting"
- Top-1: **Impressionism** — direct hit ✓ (strongest routing success)
- Post-Impressionism at ranks 3–4 (close but distinct style cluster)
- Shows the encoder can be accurate, not just adjacent

---

## Slide 21 — Inference Results: S1 v2.1 — Expressionism
- Prompt: "An Expressionism painting"
- Top-1: **Symbolism** — shares dark emotional intensity, expressive brushwork with Expressionism
- Expressionism itself not in top-5 for S1
- Symbolism is semantically adjacent — not random, but routing gap exists

---

## Slide 22 — Inference Results: S2 v2.0 — Expressionism (S1 vs S2 Routing Shift)
- **Key comparison slide** — shows S2 LDM training directly improves routing
- S1 top-1: Symbolism → S2 top-1: **Abstract Expressionism** (same visual family as Expressionism!)
- This is the clearest evidence that Stage 2 fine-tuning closes the routing gap
- Slide also includes a mini summary table: Baroque, Expressionism, Impressionism S1 vs S2 top-1
- **Point to land**: Stage 2 is doing something useful beyond Stage 1 for at least one style

---

## Slide 23 — Routing Pattern vs Generated Image (4 Styles)
- **New visual**: heatmap + generated image for all 4 styles side-by-side
- Top row: routing attention heatmaps (rows = 80 adapter pairs, cols = N experts)
- Bottom row: generated images at α=2.0
- Key observation: **each style has a distinct routing fingerprint** — not all the same pattern
- Impressionism: dense concentrated column (direct hit) → clean Impressionism output
- Baroque/Cubism/Expressionism: dispersed multi-column patterns → stylistically adjacent outputs
- Shows the encoder is truly discriminating between styles, not ignoring the query

---

## Slide 24 — Temperature Effect
- Quick slide — shows 3 Baroque outputs at different τ and k settings
- Left (hard k=1): strong single-expert aesthetic
- Middle (sharp soft k=None): our default — all experts contribute but top dominates
- Right (uniform τ=0.5): washed-out averaged look
- **Default inference: τ=0.005, k=None**

---

## Slide 25 — Routing Heatmaps: All 4 Styles
- 4 heatmaps in one view — critical evidence of style-specific routing
- Each style has a **distinctly different pattern** — not the same routing for all queries
- Impressionism: top-1 is direct Impressionism match (cleanest case)
- Baroque/Cubism/Expressionism: semantically adjacent but distinct from each other
- Non-uniform in all four confirms the encoder is not ignoring the image

---

## Slide 26 — Multi-Style Results Gallery
- 3 grid images: Baroque, Cubism, Expressionism
- Each grid: rows = routing configs, cols = different query images, bottom row = GT reference B-LoRA
- Visual evidence of variety across queries and routing settings
- Audience can scan the grids visually — no need to explain every cell

---

## Slide 27 — Model Variants Summary
- 4 variants: S1/S2 × v2.0/v2.1 (KL vs CE loss)
- v1.0 obsolete (naïve averaging = broken)
- 3 swept; S2 v2.1 training now
- Once S2 v2.1 is done: **4-way comparison across all checkpoints**

---

## Slide 28 — Open Issue: Prompt Mismatch
- B-LoRA experts were trained with trigger token **"A [v]"** (a learned special embedding)
- Our sweep used free text: "A Baroque painting", etc.
- These give completely different text embeddings → LoRA weights may not activate optimally
- Effect: style fidelity may be degraded even with correct routing
- **Current results are a lower bound**
- Proposed fix: re-run sweep with "A [v]" and compare directly

---

## Slide 29 — Engineering Challenges Overcome
- **Bug 1 — Cross-term**: all outputs identical → product-space synthesis + tSVD
- **Bug 2 — Alpha threshold**: still identical after bug 1 → α=2.0 calibration
- **Bug 3 — SVD instability (5 attempts)**:
  - fp16 overflow → fp32 accumulation
  - cuSOLVER hang → CPU LAPACK
  - fp16 NaN → nan_to_num
- S2 v2.1 now at step 875/8000, LDM≈0.55 decreasing, no crashes
- Note entropy=0: near-uniform softmax has zero gradient on -H; should self-correct as routing sharpens

---

## Slide 26 — Next Steps
- **Days**: S2 v2.1 complete → sweep → 4-way comparison
- **This week**: "A [v]" prompt experiment, CLIP-style similarity metric, ablations
- **Medium**: FID/LPIPS, multi-style interpolation, OOD testing, possible joint fine-tuning
- Core open question: **does S2 LDM fine-tuning meaningfully improve over S1 CE routing alone?**

---

## Slide 27 — Summary
- Built: 2.23M param encoder, per-tensor per-rank routing over 109 experts, product-space synthesis
- Works: non-uniform routing heatmaps, semantically adjacent top-k, α=2.0 sweet spot
- Pending: S2 v2.1, prompt fix, quantitative eval
- 3 sweeps done (240 images + alpha diagnostic)

---

## Thank You
- Code: `MoLoRAs/lora_attention/`
- Results: `/scratch/eyavuz21/lora_attention/`
- Appendix has architecture detail, oracle test, config tables

---

## Appendix Notes

**A1 — Architecture diagram**
- Labels show exact tensor shapes flowing through encoder → attention → synthesis → SDXL

**A2 — Oracle test**
- Set A = one-hot on GT expert → synthesis should recover exact GT LoRA
- cosine similarity = 0.9998 → synthesis lossless

**A3 — Training config**
- Stage 1: 15k steps, LR 3e-4, cosine, batch 8, pool 5–20, grad clip 1.0
- Stage 2: 8k steps, LR 5e-5, fp16, λ_ent 0.1→0.01, pool 5–20 (no GT)

---

## Q&A Cheatsheet

| Question | Key answer |
|----------|------------|
| Why not CLIP-nearest-expert? | CLIP clusters subject, not style |
| Entropy = 0? | near-uniform softmax → zero gradient; should sharpen with training |
| Why not joint Pool training? | Cost: 109 B-LoRAs + SDXL every step; frozen pool = stable target |
| Does prompt mismatch break results? | No — routing + synthesis validated (0.9998). Prompt affects generation quality, not routing. Lower bound. |
| Inference speed? | Standard SDXL + ~400 ms SVD overhead (~8%) |
