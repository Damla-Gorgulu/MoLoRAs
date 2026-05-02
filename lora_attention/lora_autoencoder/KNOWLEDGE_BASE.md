# LoRA Autoencoder — Experiment Knowledge Base

Living record of every training run, diagnostic, and finding for the
style-LoRA reconstructive autoencoder pipeline.

---

## Architecture Evolution

### Input

Style-only B-LoRA weights from SDXL `unet.up_blocks.0.attentions.1`.
- **80 up/down pairs** (4 matrices × 2 attn blocks × multiple transformer blocks)
- **Rank 64** per pair
- Packed into fixed-shape slots: `down1280 [B,80,64,1280]`, `down2048 [B,80,64,2048]`, `up [B,80,64,1280]`
- `d_in` mask per pair: 1280 or 2048
- Total: **14.09M params** style-only

### v1 (overfit16_v1) — 2026-05-02

```
Encoder: rank-token transformer, 5120 tokens, single CLS → latent 65536
Decoder: single memory token (ultra-bottleneck)
Latent: z:[B, 65536], from CLS token only
Loss: tensor_mse + absolute delta_mse + cosine
Heads: default Kaiming init
```

| Setting | Value |
|---------|-------|
| lr | 1e-4 |
| clip | 1.0 |
| max_steps | 2000 |
| limit | 16 |
| Key bug | Decoder: all 5120 output tokens attend to 1 single memory token |

**Result FAIL**: cos≈0.058, rel≈1.056. Loss went down but model learned near-zero outputs. Delta_mse became tiny but LoRA direction was random.

**SLURM**: 1041434
**Output**: `/scratch/eyavuz21/lora_autoencoder/overfit16_v1`

---

### v2 (overfit16_v2_loss) — 2026-05-02

```
Loss change: replace absolute delta_mse with relative delta_rel + norm_loss + rel_loss
Heavier cos_weight (2.0 vs 0.01)
```

**Result FAIL**: Worse than v1. cos≈0.001, norm_ratio collapsed to 0.33. Model learned to produce tiny outputs instead of correct direction. Relative loss terms dominated and taught the model to shrink predictions.

**SLURM**: 1041480 (failed WANDB key), 1041483
**Output**: `/scratch/eyavuz21/lora_autoencoder/overfit16_v2_loss`

---

### v3 (overfit16_v3_pair_tokens) — 2026-05-02

```
Decoder: 80 pair memory tokens instead of 1 global (critical fix)
Latent: z:[B, 80, 512] = 40960 effective
Encoder: CLS discarded, pair tokens pooled from rank tokens
Loss: delta_rel damped with log1p
```

**Result FAIL**: cos≈0.005, rel≈1.2. Still not learning direction. Decoder bottleneck fixed but loss/optimization still wrong.

**SLURM**: 1041501, 1041506
**Output**: `/scratch/eyavuz21/lora_autoencoder/overfit16_v3_pair_tokens`

---

### v4 (overfit16_v4_global_pair_fuse) — 2026-05-02

```
Encoder: concat+MLP fuse instead of h_down + h_up
Decoder: CLS token kept as global memory token (81 tokens total)
Loss: masked flat cosine/relative/norm (fixes unmasked loss bug)
Rank denominator fix for down1280/down2048 tensor MSE
```

**Result FAIL**: cos≈0.024, rel≈1.17, norm_ratio≈0.64. Slight improvement over v3 but still not learning direction.

**SLURM**: 1041529
**Output**: `/scratch/eyavuz21/lora_autoencoder/overfit16_v4_global_pair_fuse`

---

### v5 (overfit1_v5_head_init) — 2026-05-02

```
Key fix: near-zero output head init (std=1e-5, zero bias)
lr=3e-3, tensor-only loss, 1 LoRA
```

**Result FAIL**: Near-zero start but exploded after 1 Adam step. lr=3e-3 way too high for transformer.

**SLURM**: 1041552
**Output**: `/scratch/eyavuz21/lora_autoencoder/overfit1_v5_head_init`

---

### v6 (full1_v6_stable) — 2026-05-02

```
Full AE with all architecture fixes
lr=1e-4, clip=0.1, near-zero heads, tensor-only loss, 1 LoRA
```

**Result PARTIAL**: cos=0.784, rel=0.621, mse=0.000229. First run where model actually learned direction! But plateaued at cos≈0.78-0.80.

**SLURM**: 1041588
**Output**: `/scratch/eyavuz21/lora_autoencoder/full1_v6_stable`

---

## Key Diagnostic Baselines

All on style_0000_Baroque, 1 LoRA, tensor-only loss unless noted.

### Identity Check (1041544)

Sanity: `pred = target`. Result: `cos=1.0, rel=0, mse=0`. Confirms loss function is correct.

### Zero Baseline

Result: `cos=0, rel=1.0, mse=0.000654`. Lower bound on MSE.

### Shuffled Baseline (posa_0001 as prediction for style_0000)

Result: `cos=0.877, rel=0.484, mse=0.000125`. **Critical**: another LoRA is better than any trained model! Shows strong common structure across styles.

### Direct Parameter Baseline (1041545)

Learnable tensors directly matching target (no model). Adam + tensor MSE.

Result: `cos=1.0, rel=0, mse=0` in ~30 steps. Confirms loss+optimizer work perfectly.

### Metadata-Only Decoder (1041546, 1041551)

Decoder-only: metadata tokens → TransformerDecoder (16 mem tokens) → output heads. No encoder.

Initial (lr=3e-3): FAIL — cos≈0.01. Exploded after first step.
Fixed (lr=1e-4, clip=0.1, near-zero heads): **PASS** — cos=0.786. Decoder architecture validated.

| Run | Job | LR | Final cos |
|-----|-----|-----|-----------|
| meta_decoder_lr3e3 | 1041546 | 3e-3 | 0.01 |
| meta_decoder_lr1e4 | 1041573 | 1e-4 | **0.786** |
| meta_decoder_lr3e5 | 1041574 | 3e-5 | 0.726 |

**SLURM**: 1041546, 1041573, 1041574

### Metadata MLP Baseline (1041575)

Simple per-token MLP (metadata embeddings → 3-layer MLP → output slice). No transformer.

Result: `cos=0.803`. Slightly outperforms transformer decoder (0.786).

**SLURM**: 1041575

### Lookup Table Baseline (1041747)

One unique embedding per `(pair, rank)` token → MLP → output slice. No shared structure.

Result: `cos=0.707`. Worse than both transformer and MLP. Shared parameters help.

**SLURM**: 1041747

### Residual Learning — 2-LoRA Mean (1041748)

Full AE predicting `residual = W_target − mean(W_0, W_1)`. Mean from 2 LoRAs.

Result: **full_cos=0.985**. Near-perfect memorization! The common LoRA component was blocking directional learning.

**SLURM**: 1041748

### Residual Learning — 16-LoRA Mean (1041759)

Same but mean from all 16 training LoRAs. Mean template already gives cos=0.931 baseline.

Result: **FAIL**. fcos oscillates 0.93-0.95 around the baseline. Residual cos dead at zero. The 16-LoRA residual is ~7% of full weight vs ~50% for 2-LoRA. Too tiny for tensor-only MSE.

**SLURM**: 1041759
**Output**: `/scratch/eyavuz21/lora_autoencoder/residual16_v1`

### Residual — 1 LoRA on 16-LoRA Mean (1041766) — RUNNING

Diagnostic: 1-LoRA residual with 16-LoRA mean template. Isolates whether the 16-style failure is due to tiny residual signal or multi-style capacity.

**SLURM**: 1041766
**Output**: `/scratch/eyavuz21/lora_autoencoder/residual1_on16mean`

---

## Loss Function Evolution

### v1 Loss
```
tensor_mse (masked A/B MSE) + delta_mse (absolute ΔW MSE) + cos_loss
```
Problem: Absolute delta_mse on tiny weight values collapses to zero without learning direction.

### v2 Loss
```
tensor_mse + delta_rel (relative ΔW) + cos + rel + norm
```
Problem: Relative terms dominate, model collapses to near-zero outputs. cos_weight=5.0 was too aggressive.

### Current Loss (v4+)
```
tensor_mse (rank-corrected denominators)
+ log1p(delta_rel) (damped relative ΔW)
+ cos_loss (1 - cos)
+ rel (relative L2)
+ log(norm_ratio)² (norm preservation, clamped at 1e-8)
```
Default weights: tensor=1.0, delta=0.0 (debug), cos=0.0, rel=0.0, norm=0.0

### Key Bug Fixes
- **Rank denominator**: Down1280/Down2048 tensor MSE was missing `shape[-2]` (rank dim), making down loss 64× too large.
- **Masked flat losses**: Cosine/relative/norm now mask invalid `down1280/down2048` entries.
- **norm_loss NaN**: Added `.clamp_min(1e-8)` before `.log()`.

---

## Optimization Rules (Discovered)

1. **lr ≤ 1e-4** for transformer AE with rank-64 tensors. Higher LRs cause Adam to overshoot near-zero head init by 1000× on first step.
2. **clip_grad_norm = 0.1** is effective. 1.0 was too loose.
3. **Near-zero output head init** (`std=1e-5`, zero bias) is essential. Default Kaiming init produces initial outputs 40× too large.
4. **Tensor-only MSE works** for the 1-LoRA case once optimization is stable (reaches cos=0.78).
5. **cos_weight > 0.5** dominates the loss and prevents directional learning when residual is small. Debug with cos_weight=0 first.
6. **`concat + MLP fuse`** (`down + up → MLP → token`) is better than simple addition (`h_down + h_up`).

---

## Key Findings

1. **The loss function and data pipeline are correct** (identity baseline passes, direct param baseline memorizes instantly).

2. **The TransformerDecoder architecture works** — it just needed stable optimization (low LR, strong clipping, near-zero heads). It reaches cos=0.786 vs 0.80 plateau shared by all architectures.

3. **A strong common LoRA component exists** — shuffled LoRA gives cos=0.877. This dominates the MSE objective and prevents the model from learning style-specific fine structure. The model learns the common component first, then plateaus.

4. **Residual learning works** — predicting `W − mean_template` with a 2-LoRA mean reaches cos=0.985. This confirms the architecture CAN memorize when the common component is removed.

5. **The 16-LoRA residual is too small for tensor-only MSE** — with 16 LoRAs in the mean, the residual per style is ~7% of full weight, and the model collapses to near-zero residual.

6. **Shared parameter architectures outperform independent lookup** — Transformer (0.786) and MLP (0.803) both beat the lookup table (0.707). Cross-token parameter sharing helps.

7. **The full AE matches decoder-only performance** — encoder does not degrade reconstruction quality. The bottleneck is in the decoder/head path, not the encoder.

---

## Comparison with Oğuzkağan's Architecture

| aspect | Theirs | Ours |
|--------|--------|-----|
| Latent | Per-module: `[L, 512]` | Per-module + global: `[B, 81, 512]` |
| Encoder | Linear A/B projections + mean over rank + MLP | Transformer: rank tokens + self-attention |
| Decoder | MLP: latent + module_emb → A_head + B_head | TransformerDecoder: metadata queries ×-attn to pair latents |
| Cross-module | None | Full self/cross-attention across all tokens |
| Rank | 4 | 64 |
| Loss target | ΔW only (never A/B) | A/B tensor + ΔW + cosine + norm |
| Spectral loss | Yes (dominant, weight=2.0) | No |
| VAE | Yes (KL=1e-6, nearly AE) | Deterministic AE |
| lr | 5e-4 | 1e-4 |
| Optimization | Stable from start (MLP-based) | Required careful tuning (low LR, clipping, head init) |
| Stage 2 | Built (latent diffusion + CLIP) | Not started |

---

## Current State (2026-05-02)

The architecture is validated. The 1-LoRA optimization regime is known (`lr=1e-4, clip=0.1, near-zero heads`). The remaining challenge is scaling to 16+ LoRAs with residual learning.

**Immediate question**: Can the model learn a single residual from the 16-LoRA mean template? Run 1041766 answers this.

**Next steps after 1041766**:
- If residual cos rises → add cosine loss on residual, scale to 16
- If residual cos stays zero → the residual signal is too small; normalize/rescale residuals or add directional loss
- Then: mean-template baselines, full comparison table, 1000-LoRA train/val/test

---

## Files Index

| File | Purpose |
|------|---------|
| `lora_autoencoder/model.py` | StyleLoRAAutoencoder + reconstruction_losses |
| `lora_autoencoder/dataset.py` | StyleLoRAAutoencoderDataset, PairSpec, metadata |
| `train_lora_autoencoder.py` | Main training script |
| `train_lora_autoencoder_residual.py` | Residual training (target − mean_template) |
| `diagnostics/eval_lora_autoencoder_overfit.py` | Reconstruction test with SDXL inference |
| `diagnostics/ae_loss_sanity.py` | Identity/zero/shuffled/metric checks |
| `diagnostics/direct_param_baseline.py` | Learnable tensor baseline |
| `diagnostics/meta_decoder_baseline.py` | Decoder-only baseline |
| `diagnostics/meta_mlp_baseline.py` | MLP-only baseline |
| `diagnostics/lookup_baseline.py` | Per-token lookup table baseline |
| `diagnostics/residual_baseline.py` | 2-LoRA residual learning |
| `diagnostics/residual1_on_n_mean.py` | 1-LoRA residual on N-LoRA mean template |
| `slurm/lora_autoencoder/` | All SLURM wrappers |
