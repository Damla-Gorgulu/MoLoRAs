# Project Roadmap: Rank-Level Attention MoE for LoRA Style Transfer

## 1. Architectural Blueprint
* **Objective**: Implement a Mixture-of-Experts (MoE) routing mechanism at the rank level for LoRA weights in a Stable Diffusion XL (SDXL) pipeline.
* **Query ($Q$)**: Extract the incoming style image embedding using a frozen CLIP ViT encoder. The resulting shape is $1 \times d$.
* **Routing MLP**: Design a Multi-Layer Perceptron to project the raw weights of each available expert LoRA into a Key matrix.
* **Key ($K_i$)**: Generate the Key matrix for the $i$-th LoRA using the Routing MLP. The shape must be $r \times d$ (where $r$ is the rank, e.g., 64, and $d$ is the embedding dimension, e.g., 512).
* **Rank-Level Attention ($H_i$)**: Compute the raw attention scores for each rank by multiplying the Query and Key matrix: $H_i = Q K_i^T$. The resulting shape is $1 \times r$.
* **Cross-Expert Softmax ($A_i$)**: Stack the $H_i$ vectors for all $N$ available LoRAs to form an $N \times r$ matrix. Apply Softmax across the $N$ dimension (depth) so that the sum of probabilities for each rank equals 1 across all experts.
* **Weight Synthesis**: Synthesize the new LoRA weights using the Hadamard product: $W_{new\_down} = \sum_{i=1}^{N} (W_{i\_down} \odot A_i)$. Apply the same operation for the up-projection matrices.

## 2. Stage 1 Training: Ground Truth Mapping
* **Goal**: Train the Routing MLP to perfectly map a style image to its corresponding LoRA without involving the diffusion process.
* **Input Data**: A style image and its pre-trained ground truth LoRA ($LoRA_{GT}$).
* **Pool Configuration**: The routing pool contains $LoRA_{GT}$ and $N-1$ other random LoRAs.
* **Forward Pass**: Compute the $N \times r$ attention mask matrix ($A$).
* **Loss Function**: Apply a rank-level Mean Squared Error (MSE) or Cross-Entropy loss. The target for $LoRA_{GT}$ is 1 across all $r$ ranks. The target for all other $N-1$ LoRAs is 0 across all ranks.
* **Optimization**: Update only the parameters of the Routing MLP using backpropagation.

## 3. Stage 2 Training: Hold-out Reconstruction
* **Goal**: Force the Routing MLP to learn style combinations by removing the exact match from the pool.
* **Input Data**: A style image and its corresponding text prompt.
* **Pool Configuration**: The routing pool contains $N$ LoRAs, explicitly excluding the ground truth LoRA for the current style image.
* **Forward Pass**: Synthesize a composite LoRA using the rank-level attention mechanism.
* **Diffusion Injection**: Inject the synthesized LoRA weights into the frozen SDXL base model dynamically.
* **Loss Function**: Calculate the standard Latent Diffusion Model (LDM) noise prediction loss ($\mathcal{L}_{LDM}$).
* **Optimization**: Backpropagate the diffusion loss to update only the Routing MLP. SDXL base weights and the expert LoRA weights remain strictly frozen.

## 4. Bottleneck Resolution & Constraints
* **Input Dimensionality Issue**: The Routing MLP cannot feasibly take flattened $W_{down}$ and $W_{up}$ matrices (tens of thousands of parameters) directly as inputs due to memory and computation constraints.
* **Action Item**: Implement a statistical summarization step (e.g., mean, variance, or max pooling along the rank dimension) to extract features from the expert LoRA weights before feeding them into the Routing MLP to generate $K_i$.

---

## 5. Environment & Conventions

| Item | Detail |
|---|---|
| **Expert LoRA pool** | 109 active styles (1 skipped), rank-64, `.safetensors` at `repos/B-LoRA/blora_zoo/bloras/` |
| **Style block** | `unet.up_blocks.0.attentions.1` (160 tensors per LoRA) |
| **Tensor shapes** | `down`: `(64, 1280)` self-attn / `(64, 2048)` cross-attn; `up`: `(1280, 64)` always |
| **feature_dim** | 480 = 160 keys × 3 stats (mean/std/max per tensor) |
| **CLIP model** | `openai/clip-vit-base-patch32` via **`CLIPModel.get_image_features()`** (NOT `CLIPVisionModel`) |
| **CLIP dim** | 512 — `CLIPVisionModel.pooler_output` is **768** (raw ViT hidden size, no projection); `CLIPModel.get_image_features()` applies the visual projection and returns **512-dim** L2-normalised features |
| **RoutingMLP params** | 17,320,896 |
| **Base code** | `repos/B-LoRA-fresh/B-LoRA/` (stable) |
| **Conda env** | `B-LoRA_2` |
| **SLURM** | `--partition=ai --gres=gpu:tesla_v100:1 --qos=ai` |
| **Weight output** | `/scratch/eyavuz21/lora_attention/` |
| **Code root** | `repos/MoLoRAs/lora_attention/` |
| **Style images** | `repos/B-LoRA/blora_zoo/style_images/` (pass via `--image_dirs`) |
| **sys.path fix** | entry-point scripts use `parents[1]` (MoLoRAs/) not `parents[2]` (repos/) |

---

## 6. File Structure

```
repos/MoLoRAs/lora_attention/
├── roadmap.md
├── models/
│   ├── __init__.py
│   ├── lora_pool.py        # Load & cache all expert LoRAs; statistical summarization
│   ├── routing_mlp.py      # RoutingMLP: LoRA features → Key K_i (r × d)
│   └── moe_lora.py         # Full pipeline: CLIP query → rank-attention → weight synthesis
├── data/
│   ├── __init__.py
│   └── dataset.py          # Stage1Dataset and Stage2Dataset
├── utils/
│   ├── __init__.py
│   └── lora_inject.py      # Reconstruct & inject synthesized LoRA into frozen SDXL
├── train_stage1.py         # Stage 1: MSE loss on attention mask vs one-hot GT
├── train_stage2.py         # Stage 2: hold-out pool + LDM diffusion loss
├── inference.py            # Run synthesized LoRA on SDXL for a query style image
└── slurm/
    ├── train_stage1.sh
    └── train_stage2.sh
```

---

## 7. Implementation Plan

### Step 1 — `models/lora_pool.py`
Load all 110 style LoRAs; filter to style block only (`unet.up_blocks.0.attentions.1`);
compute per-LoRA summary vector via `[mean, std, max]` concatenated along rank dim
per tensor → fixed-size feature vector per expert.
Cache all features and raw tensors to `/scratch/eyavuz21/lora_attention/lora_features_cache.pt`.

### Step 2 — `models/routing_mlp.py`
`RoutingMLP(feature_dim, rank, clip_dim)`:
- Input: LoRA summary vector (from Step 1)
- Output: $K_i \in \mathbb{R}^{r \times d}$ (Key matrix for rank-level attention)
- Architecture: Linear → LayerNorm → GELU → Linear → reshape to `(rank, clip_dim)`

### Step 3 — `models/moe_lora.py`
`MoELoRA` full forward:
1. Extract CLIP ViT embedding $Q \in \mathbb{R}^{1 \times d}$ from style image (frozen)
2. For each expert $i$: `RoutingMLP(features_i)` → $K_i \in \mathbb{R}^{r \times d}$
3. $H_i = Q K_i^T \in \mathbb{R}^{1 \times r}$; stack → $[N \times r]$; softmax over $N$ → $A \in \mathbb{R}^{N \times r}$
4. Hadamard-sum: $W_{new\_down} = \sum_i (W_{i\_down} \odot A_i)$, same for $W_{up}$

### Step 4 — `data/dataset.py`
- `Stage1Dataset`: `(image, gt_lora_idx, pool_indices)` — pool size $N \sim U[3,20]$, GT always in pool
- `Stage2Dataset`: same but GT explicitly excluded from pool (forces combination)

### Step 5 — `utils/lora_inject.py`
Reconstruct a full style-block state dict from synthesized $(W_{down}, W_{up})$ tensors,
then call `pipeline.load_lora_into_unet()` from `B-LoRA-fresh`.

### Step 6 — `train_stage1.py`
- Loss: rank-level MSE of $A$ vs one-hot GT target `[N × r]`
- Updates: only `RoutingMLP` parameters
- Variable $N$ per batch (collate handles padding/masking)
- Saves checkpoint to `/scratch/eyavuz21/lora_attention/stage1/`

### Step 7 — `train_stage2.py`
- Pool: GT LoRA excluded; synthesize composite LoRA → inject into frozen SDXL
- Loss: $\mathcal{L}_{LDM}$ noise prediction
- Updates: only `RoutingMLP` parameters
- Saves checkpoint to `/scratch/eyavuz21/lora_attention/stage2/`

### Step 8 — `inference.py`
Load trained `RoutingMLP`; given a query style image, synthesize LoRA and generate images.

### Step 9 — SLURM scripts
Following cluster conventions from `experiment_3/scripts/run_baseline_training.sh`.

---

## 10. Implementation Status & Bug Log

### Completed ✅
All files created and syntax-verified. The pipeline runs end-to-end on a SLURM GPU node.

| File | Status |
|---|---|
| `models/lora_pool.py` | Done. Cache at `/scratch/eyavuz21/lora_attention/lora_features_cache.pt` |
| `models/routing_mlp.py` | Done. 17,320,896 params. |
| `models/moe_lora.py` | Done. Uses `CLIPModel.get_image_features()` (512-dim). |
| `data/dataset.py` | Done. 109 samples found (1 jpg per style). Returns PIL images (no transform). |
| `utils/lora_inject.py` | Done. Includes gradient-compatible hooks for Stage 2. |
| `train_stage1.py` | Done. `--num_workers` default 0; SLURM uses 2. |
| `train_stage2.py` | Done. Images kept as PIL; VAE transform applied in loop; CLIP gets raw PIL. |
| `inference.py` | Done. |
| `slurm/train_stage1.sh` | Done. `--cpus-per-task=4`, `--image_dirs`, `--num_workers 2`. |
| `slurm/train_stage2.sh` | Done. `--cpus-per-task=4`, `--image_dirs`. |
| `slurm/inference.sh` | Done. Tests 5 styles × 2 checkpoints (S1+S2), 4 images each. |

### Bugs Fixed

**Bug 1 — `sys.path` import error** (`ModuleNotFoundError: No module named 'lora_attention'`)
- Root cause: `parents[2]` from `MoLoRAs/lora_attention/train_stage1.py` resolves to `repos/` (grandparent), but the `lora_attention` package lives inside `MoLoRAs/`, requiring `parents[1]`.
- Fix: Changed to `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` in `train_stage1.py`, `train_stage2.py`, `inference.py`.

**Bug 2 — CLIP shape mismatch** (`RuntimeError: shape '[1, 512]' is invalid for input of size 768`)
- Root cause: Code used `CLIPVisionModel` whose `.pooler_output` is the **raw ViT [CLS] token** = **768-dim** (the ViT-B/32 hidden size). The 512-dim projection to the shared CLIP embedding space is only applied inside `CLIPModel.get_image_features()`.
- Fix: In `moe_lora.py`, switched from `CLIPVisionModel` + `.pooler_output` to `CLIPModel` + `.get_image_features()`, which applies the visual projection (768 → 512) and returns L2-normalised 512-dim features.
- **KEY RULE for future sessions**: Always use `CLIPModel.get_image_features()` for vision embeddings, never `CLIPVisionModel` alone. `clip_dim=512` is correct.

**Bug 3 — DataLoader worker OOM crash** (`DataLoader worker exited unexpectedly` / `Killed`)
- Root cause: `num_workers=2` causes each worker to pickle a full copy of `LoRAPool` (109 experts × 160 tensors ≈ 7 GB RAM). The software node cannot accommodate this.
- Fix 1: Added `--num_workers` CLI arg to `train_stage1.py`, default `0` (in-process, no workers). SLURM script uses `NUM_WORKERS=2`.
- Fix 2: **Do not run training on the software node** (`ssh software`). The software node is for installation only. Training must run via `sbatch`.

**Bug 4 — SLURM too few CPUs for DataLoader workers** (warning: "suggested max number of worker is 1")
- Root cause: `#SBATCH --ntasks-per-node=1` allocates only 1 CPU core, but `num_workers=4` requested 4 worker processes.
- Fix: Added `#SBATCH --cpus-per-task=4` to both SLURM scripts. Reduced default `NUM_WORKERS` to 2.

**Bug 5 — Stage 2 CLIP received Tensor instead of PIL** (latent bug, would crash at Stage 2 runtime)
- Root cause: `image_transform=_IMG_TRANSFORM` was applied in the dataset, converting PIL → Tensor. Then `model.encode_image()` passed this Tensor to CLIP processor which expects PIL.
- Fix: Removed `image_transform` from `Stage2Dataset` constructor. Images returned as PIL. VAE transform (`_IMG_TRANSFORM`) is applied in the training loop. CLIP gets raw PIL directly.

**Bug 6 — Stage 2 SLURM script missing `--image_dirs`**
- Same issue as Stage 1 but wasn't caught until review.
- Fix: Added `IMAGE_DIRS` variable and `--image_dirs` flag to `train_stage2.sh`.

**Bug 7 — `RemovableHook` typo in `lora_inject.py`** (`AttributeError: module 'torch.utils.hooks' has no attribute 'RemovableHook'`)
- Root cause: Type annotation used `torch.utils.hooks.RemovableHook` at module load time (lines 232, 264, 293). The correct type is `torch.utils.hooks.RemovableHandle`.
- Effect: Stage 2 job 759039 crashed at import time, before any training began.
- Fix: `sed -i 's/RemovableHook/RemovableHandle/g'` on `utils/lora_inject.py` (3 occurrences). Job 759044 resubmitted successfully.

### Image Discovery
- Style images are at `repos/B-LoRA/blora_zoo/style_images/{style_name}/{style_name}.jpg` — **1 image per style**, 109 usable styles (1 style dir missing safetensors → skipped).
- Default `image_dirs` in the dataset resolves to `zoo_dir/../style_images` = `blora_zoo/style_images` — **correct by default**.
- Both SLURM scripts explicitly pass `--image_dirs /home/eyavuz21/repos/B-LoRA/blora_zoo/style_images` for robustness.

---

## 11. Training History

### Stage 1

| Job ID | Status | Steps | Loss (start→end) | Duration | Notes |
|---|---|---|---|---|---|
| `757525` | Cancelled | ~200 | ~0.097 | — | Bugs 4–6 not yet fixed. Cancelled, replaced by 757529. |
| `757529` | **Completed ✅** | 10,000/10,000 | 0.099→0.001580 | ~5h 22min | All 6 bugs fixed. Strong convergence. |

**Job 757529 — Completed**
- Started: 2026-02-21 20:01:41 → Finished: 2026-02-22 01:23:37 (node `ai12`, V100)
- Config: 10,000 steps, lr=1e-4, batch=8, pool∈[3,20], `cpus-per-task=4`, `num_workers=2`
- Final loss: **0.001580** (started at ~0.099)
- Loss milestones: step 500→0.089, 2000→0.071, 5000→0.012, 7500→0.003, 10000→0.0016
- Checkpoints: every 500 steps → 20 total under `/scratch/eyavuz21/lora_attention/stage1/`
- **Weights for ablation**: all checkpoints preserved — use `checkpoint-{N}/checkpoint.pt` at any horizon
- SLURM log: `lora_attention/logs/MoELoRA-Stage1-757529.log`

**Performance note**: Stage 1 was CPU-bound — 8 sequential CLIP encodes per step.
Actual time: ~1.9 sec/step (faster than the ~3–4 sec/step estimate).
Future optimization: batch CLIP encoding across the full batch in one call.

### Stage 2

| Job ID | Status | Notes |
|---|---|---|
| `759039` | Failed immediately | Bug 7: `RemovableHook` typo in `lora_inject.py`, crashed at import. Fixed. |
| `759044` | **Completed ✅** (2026-02-22 03:57→05:11, node `ai12`) | Bug 7 fixed. Loads `stage1/latest.pt`, 5000 steps, lr=5e-5, batch=1, fp16. |

**Job 759044 — Completed**
- Started: 2026-02-22 03:58:15 → Finished: 2026-02-22 05:11:37 (~1h 13min)
- Config: 5000 steps, lr=5e-5, mixed_precision=fp16, GT excluded from pool
- Loss: ~0.584→~0.547 (high variance, no strong downward trend)
- Checkpoints: every 500 steps → 10 total under `/scratch/eyavuz21/lora_attention/stage2/`
- SLURM log: `lora_attention/logs/MoELoRA-Stage2-759044.log`

**Stage 2 loss note**: LDM diffusion loss (~0.55) does not decrease significantly — this is expected. The noise-prediction objective is dominated by random timestep sampling and the stochastic noise, making per-step loss highly variable. The RoutingMLP's contribution is a small perturbation to SDXL. Visual quality assessment via inference is the correct evaluation, not the training loss.

---

## 12. Initial Inference Results (pre-fix)

### Inference (job 760723 — completed)

SLURM script `slurm/inference.sh` tested 5 styles × 2 checkpoints = 10 runs, 4 images each.

**Result**: Images showed only pre-trained SDXL styles — no visible LoRA style transfer.

---

## 13. Bug 8: LoRA Injection Appears Inactive — Diagnosis & Fix

### Diagnosis (job 760729)

Created comprehensive diagnostic script `diagnose_injection.py` with 4 tests:

| Test | Result | Conclusion |
|------|--------|------------|
| **Key format** | Raw safetensors keys ≡ `pipeline.lora_state_dict()` keys | ✅ Key format is correct |
| **UNet modification** | Real B-LoRA: diff=0.008, Our inject: diff=0.008 (identical) | ✅ Injection API works correctly |
| **Synth magnitude** | Cosine sim synth↔real = 0.957; synth std=0.019 vs real std=0.021 | ✅ Synth weights are close to real |
| **S2 attention** | S2 entropy=4.42 / max=4.69 (near-uniform); S1 entropy=2.04 | ❌ S2 routing collapsed to uniform |

**Key finding — attention dilution is the root cause**:
- With 109 experts and max S1 attention of 0.44, the synthesised LoRA is a diluted blend of many styles
- Synth UNet diff = 0.003 (35% of real LoRA's 0.008)
- The partial cancellation from blending 109 styles reduces the net style effect below visual threshold
- S2 is even worse: near-uniform → no useful routing, diff=0.001

### Fix: Temperature Scaling & Top-k Routing

**Changes implemented:**

1. **`models/moe_lora.py`**: Added `temperature` and `top_k` parameters to `forward()` and `route()`
   - `temperature < 1.0` → sharper softmax → more weight on correct expert
   - `top_k` → keep only top-k experts per rank, zero rest, re-normalise
   
2. **`inference.py`**:
   - Added `--temperature`, `--top_k`, `--reference_blora` CLI arguments
   - Added `cross_attention_kwargs={"scale": 1.0}` to pipeline call
   - Added attention entropy logging
   - Added `_save_attention_heatmap()` → saves per-run heatmap PNG
   - `--reference_blora` mode bypasses MoE, injects real B-LoRA for comparison

3. **`slurm/inference_sweep.sh`**: Comprehensive sweep over:
   - Temperature: 1.0, 0.5, 0.1, 0.01
   - Top-k: none, 1, 3, 5 (combined with τ=0.1)
   - Style alpha: 1.0, 1.5, 2.0
   - Reference: real B-LoRA direct injection
   - Baseline: vanilla SDXL (alpha=0)

### Sweep Job (760741 — submitted)

Tests 3 styles × (9 configs + reference + vanilla) = 33 runs, 4 images each = 132 images.

```bash
# Monitor:
squeue -u eyavuz21
tail -f /home/eyavuz21/repos/MoLoRAs/lora_attention/logs/MoELoRA-Sweep-760741.log

# After completion — compare:
ls /scratch/eyavuz21/lora_attention/inference_sweep/
# Layout:
#   tau1.0_noTopK/style_XXXX/      ← original (no temp/topk)
#   tau0.1_noTopK/style_XXXX/      ← sharp temperature
#   tau0.1_top3/style_XXXX/        ← sharp + top-3
#   reference_blora/style_XXXX/    ← real B-LoRA
#   vanilla_sdxl/style_XXXX/       ← no LoRA baseline
```

### Expected outcomes
- **τ=0.01, top_k=1**: Should approximate direct expert injection → strongest style
- **τ=0.1, top_k=3**: Good balance between style strength and blending
- **reference_blora**: Ground truth for maximum style transfer possible
- **vanilla_sdxl**: Confirms baseline has no style artifacts

### S2 Status
Stage 2 checkpoint produces near-uniform attention (entropy 4.42/4.69 max) — effectively useless for routing. The LDM loss did not provide sufficient gradient signal to learn meaningful attention patterns beyond what Stage 1 already captured. **S2 is deprioritised; focus on S1 + temperature tuning.**

---

## 14. Temperature Sweep Results (job 760747 — completed)

The sweep tested 11 configs × 3 styles = 33 runs, 4 images each = 132 images.

### Routing entropy at different temperatures

| Config | Baroque entropy | Cubism entropy | Expressionism entropy |
|--------|-----------------|----------------|-----------------------|
| τ=1.0  | 2.04            | 1.08           | 2.20                  |
| τ=0.5  | 0.74            | 0.12           | 0.69                  |
| τ=0.1  | ~0.0002         | ~0.0           | ~0.0001               |
| τ=0.01 | ~0.0            | ~0.0           | ~0.0                  |

**Best config: τ=0.1, top_k=None, α=1.0** — essentially all weight on the single most-similar expert, while remaining differentiable (unlike top_k=1 hard selection).

---

## 15. Generalization Experiments (job 760807 — completed 2026-02-22 19:15)

### Background: What Is Being Tested? (And Why the Sweep Results Are Not Comparable)

#### Why the temperature sweep "worked" but this experiment fails

The temperature sweep (§14) used the **exact training images** as query inputs:
```
Sweep query:  blora_zoo/style_images/style_0000_Baroque/style_0000_Baroque.jpg
              ↑ This IS the image the Baroque expert was trained on.
```
At τ=0.1, the router computes CLIP similarity between the query and all 109 expert keys. When the query IS the training image, its CLIP embedding is identical to the Baroque key → similarity = 1.0 → 100% weight on Baroque. **This is trivial nearest-neighbour self-retrieval, not routing.**

The generalization experiments use a **completely different painting** from WikiArt as the query:
```
A1 Baroque query:  wikiart/Baroque/adriaen-brouwer_a-boor-asleep.jpg
                   ↑ Different artist, different composition, different CLIP embedding.
```
The router then finds whichever of the 109 training images happens to be closest in CLIP space — which turns out to be an Impressionism image (0.97 weight), not the Baroque training image. **The router never learned "Baroque style"; it memorised one Baroque painting's CLIP embedding.**

This is why A1 Baroque routes to Impressionism even though the Baroque expert IS in the pool — "in-pool" means the GT expert is available, not that the query is the training image.

---

This section asks a more meaningful question: **given a fresh, never-seen painting from a style category, can the router identify the right expert(s)?**

#### Key concepts

- **Pool expert**: A single LoRA adapter (out of 109) trained on one specific painting. Its "key" is the CLIP embedding of that one training image.
  - Example: `style_0000_Baroque` was trained on one Baroque painting. Its CLIP key encodes that exact painting.
  - Training images live in: `/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images/{expert_name}/{expert_name}.jpg`

- **Query image**: A fresh painting from WikiArt — same art style, but a **different artwork** that the router has never seen.
  - Query images live in: `/home/eyavuz21/datasets/wikiart/{StyleCategory}/`

- **Ground-truth (GT) expert**: The pool expert whose style label matches the query image's style label (e.g., the Baroque expert for a Baroque query).

- **Router decision**: At τ=0.1 (very sharp), the router puts ~100% weight on the single expert whose CLIP key is most similar to the query. We measure whether that expert is the GT.

### Experiment Structure

Three increasingly difficult scenarios, 26 runs total:

```
Exp A — "Recognition" test (7 queries)
  Each style has exactly ONE expert in the pool (singleton).
  Run A1: GT expert IS in pool  → did the router find it?
  Run A2: GT expert REMOVED     → did routing change at all?

Exp B — "Specialisation" test (5 queries)  
  Style has MULTIPLE experts in pool (e.g., 7 different Romanticism LoRAs).
  All GT experts remain in pool.
  → Do the matching experts share the weight, or does an unrelated expert win?

Exp C — "Transfer" test (7 queries)
  Style has ZERO experts in the pool (e.g., Ukiyo-e).
  → Which experts does the router use as a proxy?
  → Are the proxies art-historically sensible?
```

**Settings**: τ=0.1 (near one-hot routing), α=1.0, 4 generated images per run  
**Script**: `lora_attention/slurm/generalization.sh`  
**Output**: `/scratch/eyavuz21/lora_attention/generalization/`  
**Results CSV**: `/scratch/eyavuz21/lora_attention/generalization/report.csv`

---

### Experiment A — Recognition Test (Singleton Pool Experts)

Each style below has exactly one LoRA expert. We ask: given a fresh painting from the same style, does the router assign top weight to that expert?

**Run A1**: GT expert in pool. **Run A2**: GT expert removed — does routing change?

| Style | Pool training image (1 image, what the expert "knows") | Query image (fresh WikiArt painting) |
|-------|-------------------------------------------------------|--------------------------------------|
| Baroque | `.../style_0000_Baroque/style_0000_Baroque.jpg` | `wikiart/Baroque/adriaen-brouwer_a-boor-asleep.jpg` |
| Cubism | `.../style_0003_Cubism/style_0003_Cubism.jpg` | `wikiart/Cubism/adolf-fleischmann_hommage-delaunay-et-gleizes-1938.jpg` |
| Fauvism | `.../style_0148_Fauvism/style_0148_Fauvism.jpg` | `wikiart/Fauvism/abraham-manievich_artist-s-wife-1937.jpg` |
| Northern Renaissance | `.../style_0014_Northern_Renaissance/style_0014_Northern_Renaissance.jpg` | `wikiart/Northern_Renaissance/albrecht-altdorfer_alpine-landscape-with-church-1522.jpg` |
| Early Renaissance | `.../style_0084_Early_Renaissance/style_0084_Early_Renaissance.jpg` | `wikiart/Early_Renaissance/andrea-del-castagno_crucifixion-1.jpg` |
| High Renaissance | `.../style_0172_High_Renaissance/style_0172_High_Renaissance.jpg` | `wikiart/High_Renaissance/andrea-del-sarto_archangel-raphael-with-tobias-st-lawrence-and-the-donor-leonardo-di-lorenzo-morelli-1512.jpg` |
| Color Field | `.../style_0189_Color_Field_Painting/style_0189_Color_Field_Painting.jpg` | `wikiart/Color_Field_Painting/ad-reinhardt_abstract-painiting-1963.jpg` |

> All pool image paths are relative to `/home/eyavuz21/repos/B-LoRA/blora_zoo/style_images/`  
> All query image paths are relative to `/home/eyavuz21/datasets/`

#### Results

| Style | A1: Router's top pick (GT *in* pool) | GT rank | A2: Router's top pick (GT *removed*) | Did routing change? | Art-historical sanity |
|-------|--------------------------------------|---------|--------------------------------------|--------------------|-----------------------|
| Baroque | Impressionism (0.97) | 2 | Impressionism (0.99) | ❌ **No** | ✗ Impressionism is unrelated to Baroque |
| Color Field | Minimalism (0.99) | 30 | Minimalism (0.99) | ❌ **No** | ✓ Both geometric/flat abstract — plausible |
| Cubism | Color_Field_Painting (1.00) | 2 | Color_Field_Painting (1.00) | ❌ **No** | ✗ Color Field is unrelated to Cubism |
| Early Renaissance | Baroque (0.89) | 59 | Baroque (0.89) | ❌ **No** | ✓ Adjacent historical period |
| Fauvism | Impressionism (1.00) | 43 | Impressionism (1.00) | ❌ **No** | ✓ Fauvism grew from Impressionism |
| High Renaissance | Baroque (0.49) + Realism (0.45) | 7 | (identical) | ❌ **No** | ✓ Adjacent period |
| Northern Renaissance | Early_Renaissance (0.58) | 3 | Early_Renaissance (0.58) | ❌ **No** | ✓ Same era, northern vs. Italian school |

**Critical finding — A1 ≡ A2 for every single style.** Removing the GT expert has zero effect because at τ=0.1, the GT expert was *already* receiving near-zero weight before removal. The router was ignoring the GT expert entirely from the start. It simply doesn't recognise the query as belonging to the GT's style.

---

### Experiment B — Specialisation Test (Multiple Experts Per Style)

Here several experts share the same style label. Given a fresh query from that style, do the matching experts collectively attract the routing weight?

| Style | # experts in pool | Pool expert names | Query image |
|-------|-------------------|-------------------|-------------|
| Post-Impressionism | 3 | `style_0022_Post_Impressionism`, `style_0077_Post_Impressionism`, `style_0102_Post_Impressionism` | `wikiart/Post_Impressionism/abraham-manievich_autumn-day.jpg` |
| Impressionism | 4 | `style_0005_Impressionism`, `style_0041_Impressionism`, `style_0073_Impressionism`, `style_0095_Impressionism` | `wikiart/Impressionism/abdullah-suriosubroto_air-terjun.jpg` |
| Expressionism | 6 | `style_0010_Expressionism`, `style_0038_Expressionism`, … (6 total) | `wikiart/Expressionism/abidin-dino_drawing-pain-1968.jpg` |
| Romanticism | 7 | `style_0007_Romanticism`, `style_0024_Romanticism`, … (7 total) | `wikiart/Romanticism/adolphe-joseph-thomas-monticelli_an-evening-at-the-paiva.jpg` |
| Abstract Expressionism | 3 | `style_0002_Abstract_Expressionism`, `style_0029_Abstract_Expressionism`, `style_0089_Abstract_Expressionism` | `wikiart/Abstract_Expressionism/aaron-siskind_acolman-1-1955.jpg` |

#### Results

| Style | Router top-1 | GT rank (best-ranked GT expert) | Same style? |
|-------|-------------|--------------------------------|-------------|
| **Post-Impressionism** | Post_Impressionism (1.00) | **1** ✅ | ✓ **Perfect** — router nailed it |
| Impressionism | Post_Impressionism (0.96) | 11 | ✓ Related (both pointerly, close CLIP embedding space) |
| Expressionism | Northern_Renaissance (0.87) | 2 (Abstract_Expr.) | ✗ Best GT is only at rank 2; wrong style wins |
| Abstract Expressionism | Minimalism (0.98) | 11 | ✓ Both abstract/minimal — embedding confusion |
| Romanticism | Baroque (1.00) | 42 | ✗ Baroque wins; all 7 Romanticism experts ranked low |

2 of 5 styles route correctly (Post-Impressionism perfect; Impressionism partially correct). The remaining 3 are captured by different-style experts whose single training images happened to be closer in CLIP space to the query.

---

### Experiment C — Transfer Test (Style Not In Pool)

These styles have **no expert** in the pool at all. The router must compose from whatever experts are available as proxies. This tests whether the routing makes art-historically sensible choices.

#### C1 — Expected close analogues exist in pool

| Query style | Query image | Expected proxy | Router's actual top pick | Correct? |
|-------------|-------------|----------------|--------------------------|----------|
| Pointillism | `wikiart/Pointillism/andre-derain_boats-at-collioure-1905.jpg` | Post-Impressionism (Pointillism evolved from it) | Art_Nouveau_Modern (1.00) | ✗ Wrong |
| Analytical Cubism | `wikiart/Analytical_Cubism/albert-gleizes_acrobats-1916.jpg` | Cubism singleton | Art_Nouveau_Modern (0.91) + Cubism (0.09) | ~ Partial (Cubism in top-2) |
| Synthetic Cubism | `wikiart/Synthetic_Cubism/georges-braque_aria-de-bach-1913.jpg` | Cubism singleton | Naive_Art_Primitivism (0.96) | ✗ Wrong |

#### C2 — No obvious pool analogue

| Query style | Query image | Expectation | Router's actual top pick | Plausible? |
|-------------|-------------|-------------|--------------------------|------------|
| Action Painting | `wikiart/Action_painting/franz-kline_accent-grave-1955.jpg` | Abstract Expressionism (same gestural energy) | Color_Field_Painting (0.99) | ✓ Both expressive abstract |
| Mannerism | `wikiart/Mannerism_Late_Renaissance/agnolo-bronzino_adoration-of-the-cross-with-the-brazen-serpent.jpg` | Renaissance singletons | Romanticism (1.00) | ~ Plausible (both European figurative) |
| Rococo | `wikiart/Rococo/allan-ramsay_charlotte-sophia-of-mecklenburg-strelitz-1762.jpg` | Baroque or Romanticism | Romanticism (0.56) + Romanticism (0.38) | ✓ European decorative-narrative |
| Ukiyo-e | `wikiart/Ukiyo_e/hiroshige_a-bridge-across-a-deep-gorge.jpg` | Minimalism or pattern-heavy experts | Realism (0.94) | ✗ Realism has nothing in common with Japanese woodblock |

3 of 7 novel styles show art-historically plausible proxies.

---

### Overall Results Summary

| Metric | Value |
|--------|-------|
| Experiments with GT in pool (Exp A + B) | 12 of 19 applicable runs |
| **Top-1 accuracy** (GT is highest-weighted expert) | **1 / 12 = 8.3%** |
| **Top-3 accuracy** (GT in top 3 experts) | **5 / 12 = 41.7%** |
| GT mean rank (lower = better, best=1) | **17.75 / 109** |
| Exp C proxies art-historically plausible | 3 / 7 = 43% |

---

### Root Cause: Single-Image Overfitting

The router failed because **it was trained on exactly one image per expert**. Here is what that means concretely:

```
Training setup (Stage 1):
  109 experts × 1 training image each = 109 total training images.
  The router learned: "this CLIP embedding → use this expert."

At inference (τ=0.1):
  Router does near-exact 1-nearest-neighbour lookup against those 109 embeddings.
  A fresh WikiArt painting of the same style → different artist, composition, colours
                                             → different CLIP embedding
                                             → nearest neighbour is some OTHER expert's training image
                                             → wrong expert is selected
```

Concrete evidence:
- **A1 ≡ A2**: Removing the GT expert changes nothing because the GT's training image was already so far from the query that it had ~0 weight.
- **Only Post-Impressionism works** (rank=1): Its one training image happened to be CLIP-close to the WikiArt query — a lucky coincidence, not generalisation.
- **8.3% top-1 accuracy** out of 109 experts: random chance would be ~0.9%. The router is actively routing to the *wrong* expert because a different style's training image is closer in CLIP space.

This is precisely what v2.0 (§18) addresses: training on **thousands of WikiArt images per style** forces the encoder to learn a style *centroid* rather than memorise one image's embedding.

---

## 16. Follow-on Study: Full WikiArt Stage-2 Retraining (planned)

### Motivation

The current Stage-2 training used only the **109 B-LoRA zoo style images** (one image per expert) as training queries. The LDM loss made no meaningful progress beyond Stage-1 routing (§13). A key confound is severe **data starvation**: 109 unique training images is far too few to learn robust style embeddings from an LDM signal.

WikiArt provides **≥28 style categories × hundreds of images each** (~80k images total at `/home/eyavuz21/datasets/wikiart/`). Rerunning Stage-2 with WikiArt as the image source — while keeping the same 109 frozen expert LoRAs as the pool — tests the hypothesis that **the LDM objective can learn to compose styles if given enough stylistically diverse training images**.

### Experimental question

> Does training Stage-2 on thousands of WikiArt images improve (or fail to affect) the routing MLP's ability to generalise to unseen style queries, compared to the 109-image baseline?

### Plan

#### Step 1 — WikiArt ↔ LoRA label mapping

Each WikiArt image carries a style label (e.g., `Impressionism`). For supervised LDM training we need to either:
- **(Option A — matched pool)** restrict training to the 21 WikiArt categories that overlap with pool expert categories; assign GT expert = matching pool expert for the loss.
- **(Option B — unsupervised Stage-2)** use all WikiArt images with GT *always excluded* and let the LDM gradient teach composition; no label matching needed.

**Recommended approach**: Option A for a clean comparison with the previous 109-image Stage-2 run; Option B as an ablation.

WikiArt categories that overlap with pool experts:
`Impressionism`, `Expressionism`, `Romanticism`, `Post_Impressionism`, `Abstract_Expressionism`, `Minimalism`, `Symbolism`, `Naive_Art_Primitivism`, `Realism`, `Baroque`, `Cubism`, `Fauvism`, `Northern_Renaissance`, `Early_Renaissance`, `High_Renaissance`, `Color_Field_Painting`, `New_Realism`, `Art_Nouveau_Modern`, `Pop_Art`

#### Step 2 — Dataset changes (`data/dataset.py`)

Add a `WikiArtStage2Dataset` class:
```python
class WikiArtStage2Dataset(Stage2Dataset):
    """
    Like Stage2Dataset but draws image queries from WikiArt instead of
    blora_zoo/style_images.  Supports label-matched GT exclusion (Option A)
    and label-free exclusion (Option B / all-excluded).
    """
    def __init__(self, wikiart_root, zoo_dir, cache_dir,
                 label_map,          # dict: wikiart_category → list[pool_idx]
                 pool_size=15, mode="option_a", ...):
        ...
```

The `label_map` is a JSON file produced once from the pool style names.

#### Step 3 — Training script (`train_stage2_wikiart.py`)

Minimal diff from `train_stage2.py`:
- Replace dataset class with `WikiArtStage2Dataset`
- Add `--wikiart_root` and `--label_map` CLI args
- Keep all other hyperparams identical to the original Stage-2 run for fair comparison

#### Step 4 — SLURM script (`slurm/train_stage2_wikiart.sh`)

```bash
#SBATCH --time=12:00:00   # WikiArt is ~16× larger; budget extra time
WIKIART="/home/eyavuz21/datasets/wikiart"
LABEL_MAP="$REPO_ROOT/lora_attention/data/wikiart_label_map.json"
STEPS=20000               # 4× Stage-2 steps to expose model to full dataset
LR=5e-5
BATCH=1
```

Start from the **Stage-1 latest checkpoint** (`stage1/latest.pt`), not the failed Stage-2, to have a clean baseline.

#### Step 5 — Evaluation

After training, run the same generalization suite (§15) with the new checkpoint and compare:

| Metric | Original Stage-2 (109 imgs) | WikiArt Stage-2 (~80k imgs) |
|--------|----------------------------|----------------------------|
| A1 top-1 acc (GT in pool) | TBD | TBD |
| A2 top-3 acc (GT held-out) | TBD | TBD |
| Exp B intra-category % | TBD | TBD |
| Exp C proxy quality (visual) | TBD | TBD |
| Stage-2 final LDM loss | ~0.55 | TBD |

### Prerequisites

- [ ] Generalization results from §15 (Exp A/B/C) available as baseline
- [ ] `WikiArtStage2Dataset` implemented and tested locally (software node, small subset)
- [ ] `wikiart_label_map.json` generated from pool style names × WikiArt categories
- [ ] Estimate: ~12h on a single V100 for 20k steps at 1 sample/step

### Status: **Subsumed into Architecture v2.0 (§18)** — WikiArt training is now integral to the redesigned pipeline, not a standalone follow-on.

The generalization analysis (§15) confirmed single-image overfitting as root cause. The post-hoc architectural review (§18) further identified two additional bottlenecks (attention collapse from one-hot targets, information loss from statistical summarisation). WikiArt training addresses the data starvation problem but must be combined with the new LoRARankEncoder and soft targets to be effective.

---

## 17. Next Steps

| Priority | Task | Status |
|----------|------|--------|
| ✅ Done | Generalization Exp A/B/C (job 760807) | Complete |
| ✅ Done | `analyse_generalization.py` + CSV report | Complete |
| 🔴 High | Implement Architecture v2.0 (see §18) | Ready |
| 🔴 High | Implement `LoRARankEncoder` + per-tensor routing | Blocked by §18 design |
| 🔴 High | Implement soft-target Stage-1 + CLIP similarity matrix | Blocked by §18 design |
| 🔴 High | Build `WikiArtStage2Dataset` + label map | Ready |
| 🔴 High | Create v2.0 training scripts + SLURM | Blocked by above |
| 🔴 High | Submit v2.0 Stage-1 + Stage-2 training | Blocked by above |
| 🟡 Medium | Re-run generalization suite on v2.0 checkpoints | After training |
| 🟡 Medium | A/B comparison: v1.0 (global attention) vs v2.0 (per-layer attention) | After above |
| 🟢 Low | Quantitative eval: FID / LPIPS / DINO-distance | After best config confirmed |
| 🟢 Low | Paper figures: comparison grids | Final step |

---

## 18. Architecture v2.0: Per-Layer Rank-Level MoE with Learned Encoding

### 18.0 — Bottleneck Analysis (from §15 results + post-hoc review)

Three systemic problems in v1.0 prevent the model from learning style *composition*:

#### A. Attention Collapse → Single-Expert Retrieval

**v1.0 behaviour**: At τ=0.1 the softmax gives ≥99% weight to one expert. The router is a 1-NN classifier, not a synthesiser.

**Root causes**:
1. **Stage-1 one-hot targets**: MSE against a target where GT=1.0 and all others=0.0 trains the MLP to discriminate, not to blend. There is zero gradient incentive to keep secondary experts alive.
2. **Temperature hack**: τ=0.1 was introduced (§13) to counteract dilution (max attention 0.44 across 109 experts). But this merely hides the real problem by forcing a hard-argmax — it doesn't teach the router *which* combination of lower-ranked experts would reconstruct a held-out style.

**v2.0 fix**:
- **Soft targets** (Stage 1): Replace hard one-hot with CLIP-similarity label smoothing.
  Pre-compute the pairwise cosine similarity matrix $S \in \mathbb{R}^{109 \times 109}$ between all expert style images' CLIP embeddings. For a sample with GT index $g$ in pool $\mathcal{P}$:
  $$\text{target}_{i} = \frac{\exp(S_{g,i} / \tau_{\text{label}})}{\sum_{j \in \mathcal{P}} \exp(S_{g,j} / \tau_{\text{label}})}, \quad i \in \mathcal{P}$$
  This teaches the router that "Baroque is somewhat similar to High Renaissance" — the attention naturally forms soft clusters rather than spikes.
- **Loss**: KL divergence $D_{\text{KL}}(\text{target} \| A)$ instead of MSE. Applied per-rank, averaged.
- **No temperature hack** at train time: τ=1.0 during training; τ tuning only at inference.

- **Entropy regularisation** (Stage 2): Add an explicit load-balancing penalty:
  $$\mathcal{L} = \mathcal{L}_{\text{LDM}} - \lambda \cdot \frac{1}{T} \sum_{t=1}^{T} \mathcal{H}(A_t)$$
  where $\mathcal{H}(A_t) = -\sum_i A_{t,i} \log A_{t,i}$ is the entropy of the attention distribution for tensor group $t$. $\lambda$ annealed from 0.1 → 0.01 over training. This explicitly prevents the LDM loss from collapsing all weight onto one expert.

#### B. Data Starvation

**v1.0 behaviour**: Stage-2 trained on 109 images; LDM loss flat at ~0.55.

**Root cause**: One image per style means the CLIP embedding of each style is a single point. The RoutingMLP memorises these points rather than learning a style *distribution*. Any unseen image from the same style carries a different CLIP embedding and gets mis-routed (top-1 accuracy = 8.3%).

**v2.0 fix**: WikiArt dataset (~80k images, ≥28 categories at `/home/eyavuz21/datasets/wikiart/`). Multiple images per style forces the router to learn **class centroids** in CLIP space, not individual image embeddings.

#### C. Information Destruction in Feature Extraction

**v1.0 behaviour**: Each expert LoRA's 80 adapter pairs (160 tensors) are collapsed to 160 × `[mean, std, max]` = 480 scalars. The RoutingMLP then produces a **single global key** $K_i \in \mathbb{R}^{r \times d}$ shared across all 80 adapter pairs.

**Two problems**:

1. **Statistical summarisation destroys rank-level structure**. The mean/std/max of a `(64, 1280)` tensor throws away *which* rank carries *which* feature. The model cannot distinguish "expert X puts colour info in rank 3" from "expert X puts texture info in rank 3."

2. **Global attention ignores layer heterogeneity**. In B-LoRA, different layers serve different roles:
   - Self-attention (attn1) in early blocks → spatial layout, texture repetition
   - Cross-attention (attn2) → text-to-visual alignment, semantic control
   - Later blocks → fine-grain details, colour palette
   
   A single $A \in \mathbb{R}^{N \times r}$ applied to *all* 80 adapter pairs means: if the router decides "give rank 1 to expert A," that decision applies to Q, K, V, out projections in self-attn AND cross-attn across ALL 10 transformer blocks. **But optimal style composition requires different expert-rank combinations per layer.** For example: rank 1 of Impressionism may carry brushstroke texture (useful in self-attn), while rank 5 of Color Field carries palette info (useful in cross-attn value projection). The current architecture cannot express this.

**v2.0 fix**: 
- **LoRA Rank Encoder** (replaces `[mean, std, max]` + RoutingMLP): A learned `Conv1d` over the raw weight matrix that preserves the rank dimension.
- **Per-tensor attention**: Each of the 80 adapter pairs gets its own attention map $A_t \in \mathbb{R}^{N \times r}$, producing a total attention tensor $\mathcal{A} \in \mathbb{R}^{T \times N \times r}$ where $T = 80$.

---

### 18.1 — v2.0 Architecture Specification

#### Tensor Layout (B-LoRA style block)

```
Per expert: 160 weight tensors = 80 adapter pairs (down + up)
├── 10 transformer blocks (block 0..9)
│   ├── attn1 (self-attention): 4 projections ── to_q, to_k, to_v, to_out.0
│   │   └── each: down (64, 1280) + up (1280, 64)
│   └── attn2 (cross-attention): 4 projections ── to_q, to_k, to_v, to_out.0
│       ├── to_q, to_out.0: down (64, 1280) + up (1280, 64)
│       └── to_k, to_v:     down (64, 2048) + up (1280, 64)   ← wider
└── Total: 60 down tensors @ (64, 1280) + 20 down tensors @ (64, 2048)
```

**Per expert**: 80 adapter pairs, 2 distinct input dims.
**Pool of N experts**: 80 × N attention maps, each (N, r).

#### Component 1: LoRARankEncoder (replaces `RoutingMLP` + `_compute_features`)

```
NEW: models/rank_encoder.py

class LoRARankEncoder(nn.Module):
    """
    Learns per-rank representations from raw LoRA down-projection weights.
    
    Treat W_down ∈ ℝ^{r × d_in} as a sequence of r tokens, each d_in-dimensional.
    Apply shared point-wise MLP (≡ Conv1d kernel=1) to project each rank vector
    independently into CLIP-compatible key space.
    
    Separate encoder heads for different d_in sizes (1280 vs 2048).
    """
    
    # Encoder for d_in=1280 (60 of 80 adapter pairs):
    enc_1280 = nn.Sequential(
        nn.Conv1d(1280, hidden, kernel_size=1),   # (N, 1280, r) → (N, hidden, r)
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Conv1d(hidden, clip_dim, kernel_size=1),
    )
    
    # Encoder for d_in=2048 (20 of 80 adapter pairs):
    enc_2048 = nn.Sequential(
        nn.Conv1d(2048, hidden, kernel_size=1),
        nn.LayerNorm(hidden),
        nn.GELU(),
        nn.Conv1d(hidden, clip_dim, kernel_size=1),
    )
    
    def encode(self, W_down, d_in):
        """
        W_down: (batch, r, d_in)  — batch may be N experts or N×T
        Returns: K ∈ (batch, r, clip_dim) — the Key matrix per expert per tensor
        """
        x = W_down.transpose(1, 2)                       # (batch, d_in, r)
        enc = self.enc_1280 if d_in == 1280 else self.enc_2048
        out = enc(x)                                      # (batch, clip_dim, r)
        return out.transpose(1, 2)                        # (batch, r, clip_dim)
```

**Key insight**: `Conv1d(d_in, clip_dim, kernel_size=1)` ≡ a shared linear layer applied independently to each rank position. It preserves the $r$ (rank) dimension while learning which of the $d_{in}$ features are stylistically important. Fully end-to-end differentiable.

**Parameter count** (with hidden=512, clip_dim=512):
- `enc_1280`: (1280×512 + 512×512) × 2 (with bias) ≈ **1.3M params**
- `enc_2048`: (2048×512 + 512×512) × 2 ≈ **2.1M params**  
- **Total: ~3.5M params** (vs 17.3M for old RoutingMLP) — 5× smaller, more expressive.

#### Component 2: Per-Tensor Attention

```
NEW behaviour in models/moe_lora_v2.py :: MoELoRAv2.forward()

For each adapter pair t ∈ {0, ..., 79}:
    d_in_t = 1280 or 2048 (known from tensor key)
    
    # 1. Build keys for all N experts at tensor t:
    W_down_stack = pool.get_stacked_tensors(pool_indices, down_key_t)  # (N, r, d_in)
    K_t = encoder.encode(W_down_stack, d_in_t)                         # (N, r, clip_dim)
    
    # 2. Rank-level attention:
    H_t = Q @ K_t.transpose(1,2)  per expert → bmm → (N, 1, r)
    H_t = H_t.squeeze(1) / sqrt(clip_dim)                             # (N, r)
    A_t = softmax(H_t / temperature, dim=0)                            # (N, r)
    
    # 3. Synthesise this tensor pair using A_t (NOT the global A):
    synth_down_t = (A_t[:, :, None] * W_down_stack).sum(dim=0)        # (r, d_in)
    synth_up_t   = (A_t[:, None, :] * W_up_stack).sum(dim=0)          # (d_out, r)

Returns:
    attention_per_tensor: (T, N, r)  — T=80 independent routing decisions
    synth_lora: Dict[str, Tensor]    — fully synthesised style-block state dict
```

**Critical difference from v1.0**: Each tensor pair gets its own $A_t$. The model can now express: "for `block5.attn2.to_v` (colour palette), lean on Impressionism rank 3; for `block2.attn1.to_q` (spatial layout), lean on Minimalism rank 1." This is what enables cross-expert style composition at the layer level.

#### Component 3: Soft Targets (Stage 1)

```
Pre-compute once:
    S[i,j] = cosine_sim(CLIP(style_image_i), CLIP(style_image_j))    # (109, 109)

Per training sample with GT index g, pool P of size N:
    soft_target = softmax(S[g, P] / τ_label)    # (N,) — soft distribution
    # Broadcast to (N, r) since target is same across all ranks within a tensor group
    
    # Loss per tensor group t:
    L_t = KL_div(log(A_t), soft_target)    # KL between predicted and soft target
    
    # Total Stage-1 loss:
    L = (1/T) Σ_t L_t
```

**τ_label** hyperparameter:
- τ_label → 0: sharp targets → approaches one-hot (v1.0 behaviour)
- τ_label → ∞: uniform → no discriminative signal
- **Recommended**: τ_label ∈ [0.2, 0.5] — enough smoothing to keep secondary experts alive

#### Component 4: WikiArt Training Pipeline

(As planned in §16, now integral to v2.0 rather than a follow-on study.)

Stage 1: Use WikiArt images as queries (multiple per style) → router sees full intra-class CLIP distribution.
Stage 2: Use WikiArt + entropy regularisation → LDM with meaningful gradients.

---

### 18.2 — Deprecated Components (v1.0 → v2.0)

| Component | v1.0 | v2.0 | Why |
|-----------|------|------|-----|
| Feature extraction | `[mean, std, max]` → 480-d | LoRARankEncoder (Conv1d) | Preserves per-rank structure |
| RoutingMLP | 480-d → K ∈ ℝ^{r×d} (17.3M params) | Removed; encoder outputs keys directly (3.5M) | Simpler, smaller, more expressive |
| Attention | Single global $A \in \mathbb{R}^{N \times r}$ | Per-tensor $\mathcal{A} \in \mathbb{R}^{T \times N \times r}$, T=80 | Layer-level composition |
| Stage-1 target | One-hot (MSE) | Soft CLIP-similarity (KL divergence) | Prevents single-expert collapse |
| Stage-1 loss | MSE on (N, r) | KL div, per-tensor, averaged over T | Distributional learning |
| Stage-2 loss | Pure LDM | LDM + entropy regularisation | Prevents attention collapse |
| Training data | 109 zoo images | WikiArt (~80k images) | Breaks single-image overfitting |
| Temperature | τ=0.1 at train+inference | τ=1.0 at train; τ tuning at inference only | Train to compose, sharpen at test time |
| `lora_features_cache.pt` | Stores 480-d features | Stores raw tensors only (no pre-computed features) | Encoder computes keys on-the-fly |

---

### 18.3 — v2.0 Implementation Plan

#### Step 1 — `models/rank_encoder.py` (NEW)

`LoRARankEncoder(hidden_dim, clip_dim)`:
- Two `Conv1d` encoder heads (`enc_1280`, `enc_2048`)
- `encode(W_down_batch, d_in)` → keys (batch, r, clip_dim)
- Shared across all experts and tensor positions

#### Step 2 — `models/moe_lora_v2.py` (NEW or major refactor of `moe_lora.py`)

`MoELoRAv2(pool, encoder, clip_model_id, rank, clip_dim)`:
- `forward(query_embedding, pool_indices, temperature)` → `(attention_per_tensor, synth_lora)`
- Internal loop (or batched) over 80 adapter pairs
- Returns $\mathcal{A} \in \mathbb{R}^{T \times N \times r}$
- Efficient batching: group 60 tensors with d_in=1280, 20 with d_in=2048 → 2 batched encoder calls

**Batching strategy** (critical for GPU efficiency):
```python
# Group all d_in=1280 tensors across all experts:
# W_1280: (N, 60, r, 1280) → reshape (N*60, r, 1280)
# → encoder → (N*60, r, clip_dim) → reshape (N, 60, r, clip_dim)
# Same for d_in=2048: (N, 20, r, 2048) → ...
# Total: just 2 batched Conv1d calls, not 80 sequential ones
```

#### Step 3 — `models/lora_pool.py` update

- Remove `_compute_features()` and `features` property (deprecated)
- Keep raw tensor storage (`_style_tensors`) — the encoder reads these directly
- Add helper: `get_down_tensors_by_dim(indices, d_in)` → batched tensor retrieval
- Rebuild cache without the 480-d feature vectors

#### Step 4 — `data/dataset.py` updates

- **`WikiArtDataset`** (Stage 1 + 2): loads images from WikiArt directories, maps WikiArt category → pool expert indices
- **Soft target computation**: pre-compute and store `S` (109×109 CLIP similarity matrix) once; dataset returns soft targets
- **`wikiart_label_map.json`**: auto-generated mapping of WikiArt category → list of pool expert indices

#### Step 5 — `train_stage1_v2.py` (NEW)

- Loss: per-tensor KL divergence with soft targets
- Soft target: `softmax(S[gt, pool] / τ_label)` broadcast to all rank positions
- Average loss across T=80 tensor groups
- τ\_label as CLI arg (default 0.3)
- Data: WikiArt images (multiple per style) via `WikiArtDataset`
- Updates: only `LoRARankEncoder` parameters

#### Step 6 — `train_stage2_v2.py` (NEW)

- Loss: $\mathcal{L}_{\text{LDM}} - \lambda \cdot \bar{\mathcal{H}}(\mathcal{A})$
- λ annealing schedule: linear from 0.1 → 0.01 over training
- Data: WikiArt images via `WikiArtDataset` (GT excluded from pool)
- Updates: only `LoRARankEncoder` parameters
- Start from v2.0 Stage-1 checkpoint

#### Step 7 — `inference_v2.py` (NEW or refactored)

- Load encoder checkpoint instead of RoutingMLP
- Per-tensor attention → 80 heatmaps (or grouped/averaged for visualisation)
- Save full $\mathcal{A} \in \mathbb{R}^{80 \times N \times r}$ in attention `.pt` for analysis
- Temperature scaling at inference (τ=0.1 as before, applied per-tensor)

#### Step 8 — Analysis & ablation

- Compare v1.0 (global attention) vs v2.0 (per-tensor) on same generalization suite (§15)
- Ablation: per-tensor (T=80) vs per-layer (T=20) vs global (T=1, reduced to v1.0)
- Ablation: soft targets (τ_label=0.3) vs one-hot vs uniform
- Ablation: with/without entropy regularisation in Stage 2

---

### 18.4 — v2.0 Memory & Compute Estimates

| Component | v1.0 | v2.0 |
|-----------|------|------|
| Trainable params | 17.3M (RoutingMLP) | ~3.5M (LoRARankEncoder) |
| Feature cache | 480-d per expert | Raw tensors only (already stored) |
| Attention tensor | (N, r) = (15, 64) = 960 floats | (80, N, r) = (80, 15, 64) = 76,800 floats |
| Encoder forward | 1 MLP pass per expert | 2 batched Conv1d calls (grouped by d_in) |
| Key computation | N × (480 → 32k) MLP | ~(N×80, d_in, r) → Conv1d → 2 calls total |
| GPU memory (est.) | ~3 GB (CLIP + RoutingMLP + pool) | ~4 GB (+raw W_down in VRAM) |
| Training speed (est.) | ~1.9 sec/step (S1) | ~2.5–3.0 sec/step (more attention, less params) |

Fits comfortably on a single V100 (16 GB).

---

### 18.5 — Updated File Structure (v2.0)

```
repos/MoLoRAs/lora_attention/
├── roadmap.md
├── models/
│   ├── __init__.py
│   ├── lora_pool.py          # Updated: remove _compute_features, add batched tensor getters
│   ├── rank_encoder.py       # NEW: LoRARankEncoder (Conv1d, replaces RoutingMLP)
│   ├── routing_mlp.py        # DEPRECATED (kept for v1.0 checkpoint loading)
│   ├── moe_lora.py           # DEPRECATED (kept for v1.0 inference)
│   └── moe_lora_v2.py        # NEW: per-tensor attention MoE
├── data/
│   ├── __init__.py
│   ├── dataset.py            # Updated: WikiArtDataset, soft targets, label map
│   └── wikiart_label_map.json # NEW: auto-generated category → pool idx mapping
├── utils/
│   ├── __init__.py
│   ├── lora_inject.py         # Unchanged
│   └── clip_similarity.py     # NEW: pre-compute 109×109 CLIP cosine sim matrix
├── train_stage1.py            # DEPRECATED (v1.0, kept for reference)
├── train_stage1_v2.py         # NEW: soft targets + per-tensor KL loss
├── train_stage2.py            # DEPRECATED
├── train_stage2_v2.py         # NEW: LDM + entropy reg + WikiArt
├── inference.py               # DEPRECATED (v1.0)
├── inference_v2.py            # NEW: per-tensor routing inference
├── analyse_generalization.py  # Updated for v2.0 attention format
└── slurm/
    ├── train_stage1.sh        # DEPRECATED
    ├── train_stage2.sh        # DEPRECATED
    ├── train_stage1_v2.sh     # NEW
    ├── train_stage2_v2.sh     # NEW
    ├── inference_sweep.sh     # DEPRECATED
    ├── generalization.sh      # To be updated for v2.0
    └── inference_v2.sh        # NEW
```

---

### 18.6 — v2.0 Training Plan & Schedule

| Phase | Job | Duration (est.) | Prerequisite |
|-------|-----|-----------------|--------------|
| 0 | Generate `wikiart_label_map.json` + CLIP similarity matrix (109×109) | ~10 min (CPU) | — |
| 1 | v2.0 Stage-1: WikiArt + soft targets + per-tensor KL | ~8–10h (V100) | Phase 0 |
| 2 | v2.0 Stage-2: WikiArt + LDM + entropy reg | ~12–14h (V100) | Phase 1 |
| 3 | Generalization suite (§15 rerun) with v2.0 checkpoint | ~3h (V100) | Phase 2 |
| 4 | Analysis + ablation + comparison grid | ~30 min | Phase 3 |

**Total estimated wall-clock**: ~24–28h across 4 SLURM jobs.

---

## 19. Next Steps (v2.0 Implementation Order)

| Priority | Task | Status |
|----------|------|--------|
| 🔴 1 | Implement `models/rank_encoder.py` (LoRARankEncoder) | Not started |
| 🔴 2 | Implement `models/moe_lora_v2.py` (per-tensor attention + batched encoder) | Not started |
| 🔴 3 | Update `models/lora_pool.py` (remove old features, add batched tensor getters) | Not started |
| 🔴 4 | Implement `utils/clip_similarity.py` (109×109 matrix) + generate `wikiart_label_map.json` | Not started |
| 🔴 5 | Implement `data/dataset.py` updates (WikiArtDataset + soft targets) | Not started |
| 🔴 6 | Implement `train_stage1_v2.py` (per-tensor KL + soft targets) | Not started |
| 🔴 7 | Implement `train_stage2_v2.py` (LDM + entropy regularisation) | Not started |
| 🔴 8 | Create v2.0 SLURM scripts | Not started |
| 🔴 9 | Submit v2.0 Stage-1 training | Blocked by 1–8 |
| 🔴 10 | Submit v2.0 Stage-2 training | Blocked by 9 |
| 🔴 11 | Re-run generalization suite + analysis | Blocked by 10 |
| 🟡 12 | Ablation: T=80 vs T=20 vs T=1 | After 11 |
| 🟡 13 | Ablation: τ_label sweep | After 11 |
| 🟢 14 | Paper figures + quantitative eval | Final |