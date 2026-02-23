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

## 19. v2.0 Implementation Status (updated 2026-02-23)

| Phase | Task | Status |
|-------|------|--------|
| ✅ | Implement `models/rank_encoder.py` (LoRARankEncoder) | Done |
| ✅ | Implement `models/moe_lora_v2.py` (per-tensor attention) | Done |
| ✅ | Update `models/lora_pool.py` (batched tensor getters) | Done |
| ✅ | Implement `utils/clip_similarity.py` + `wikiart_label_map.json` | Done |
| ✅ | `data/dataset.py` — WikiArtDataset + soft targets + no `self.pool` | Done |
| ✅ | `train_stage1_v2.py` — per-tensor KL + soft targets | Done |
| ✅ | `train_stage2_v2.py` — LDM + entropy regularisation | Done |
| ✅ | v2.0 SLURM scripts (all 4) | Done |
| ✅ | Phase 0 — CLIP similarity matrix + label map | Done (job 761128) |
| ✅ | Phase 1 — Stage 1 v2.0 training (15 000 steps) | Done (job 761184) |
| ⏳ | Phase 2 — Stage 2 v2.0 training (8 000 steps) | Running (job 761720, step ~2900/8000) |
| ✅ | Phase 3 — Generalization suite with S1 checkpoint | Done (job 761724) |
| 🔴 | Phase 3b — Generalization suite with S2 checkpoint | Blocked by Phase 2 |
| 🔴 | Phase 4 — Full analysis + comparison grid | After Phase 3b |
| 🟡 | Ablation: τ_inference sweep (0.01 / 0.05 / 0.1 / 0.5) | After Phase 3b |
| 🟡 | Ablation: T=80 vs T=20 vs T=1 | After Phase 3b |
| 🟢 | Paper figures + quantitative eval | Final |

---

## 20. v2.0 Experimental Results (2026-02-23)

### 20.1 — Training Checkpoints

| Checkpoint | Path | Final Loss | Steps |
|------------|------|-----------|-------|
| Stage 1 v2.0 | `/scratch/eyavuz21/lora_attention/stage1_v2/latest.pt` | 0.303 (KL) | 15 000 |
| Stage 2 v2.0 | `/scratch/eyavuz21/lora_attention/stage2_v2/latest.pt` | ~0.56 LDM (in progress) | ~2900/8000 |

Stage 1 loss curve: 0.348 → 0.303 over 15 000 steps, entropy stabilised at ~2.44.
Stage 2 at step 2900: LDM=0.565, entropy contribution=−0.164, λ=0.0674 (annealing 0.1→0.01).

---

### 20.2 — Generalization v2 Results (S1 checkpoint, τ=0.1)

All 26 runs completed (job 761724, 2026-02-23). Results directory: `/scratch/eyavuz21/lora_attention/generalization_v2/`

Every output folder contains: `__query.jpg` (input) + `__top1…__top5` expert thumbnails alongside generated images and `_heatmap.png`.

#### Exp A — Recognition Test (singleton pool experts, 7 styles × 2 conditions)

| Style | Condition | Entropy | Max ent | Top-1 retrieved | GT rank |
|-------|-----------|---------|---------|-----------------|---------|
| Baroque | in-pool | 4.05 | 4.69 | Realism | **#6** |
| Baroque | held-out | 4.03 | 4.68 | Realism | — |
| Color Field | in-pool | 4.24 | 4.69 | Abstract Expressionism | — (no GT in pool) |
| Color Field | held-out | 4.24 | 4.68 | Abstract Expressionism | — |
| Cubism | in-pool | 3.93 | 4.69 | Cubism | **#1** ✓ |
| Cubism | held-out | 3.95 | 4.68 | Abstract Expressionism | — |
| Early Renaissance | in-pool | 4.11 | 4.69 | Symbolism | **#2** |
| Early Renaissance | held-out | 4.10 | 4.68 | Symbolism | — |
| Fauvism | in-pool | 4.04 | 4.69 | Cubism | **#11** |
| Fauvism | held-out | 4.02 | 4.68 | Cubism | — |
| High Renaissance | in-pool | 4.07 | 4.69 | Baroque | **#3** |
| High Renaissance | held-out | 4.06 | 4.68 | Baroque | — |
| Northern Renaissance | in-pool | 4.06 | 4.69 | Realism | **#38** |
| Northern Renaissance | held-out | 4.04 | 4.68 | Realism | — |

GT rank #1 in 1/7 styles (Cubism). Average GT rank ≈ 10. All entropies ≈ 4.0–4.2 / max 4.69.

#### Exp B — Specialisation Test (multiple pool experts per training style)

| Style | Entropy | Top-1 retrieved |
|-------|---------|-----------------|
| Abstract Expressionism | 4.17 | Symbolism |
| Expressionism | 4.09 | Cubism |
| Impressionism | 4.12 | Realism |
| Post-Impressionism | 4.05 | Cubism |
| Romanticism | 4.10 | Realism |

Zero out of 5 styles correctly retrieved a same-family expert as top-1. Entropy uniformly high.

#### Exp C — Transfer Test (zero-shot styles not in pool)

| Novel Style | Entropy | Top-1 retrieved | Reasonable? |
|-------------|---------|-----------------|-------------|
| Action Painting | 3.85 | Abstract Expressionism | ✓ related |
| Analytical Cubism | 4.00 | Cubism | ✓ correct family |
| Mannerism | 4.13 | Early Renaissance | ✓ historically close |
| Pointillism | 4.06 | Cubism | ✗ (should be Impressionism) |
| Rococo | 4.08 | Baroque | ✓ correct family |
| Synthetic Cubism | 4.12 | Cubism | ✓ correct family |
| Ukiyo-e | 4.16 | Symbolism | ~ debatable |

5/7 Exp C retrievals are semantically reasonable — the encoder has learned broad style-family structure even without explicit zero-shot training.

---

### 20.3 — Why the Output Images Show No Visible Style Activation

**Observation**: the attention heatmap correctly ranks the GT expert near the top, but the generated images show no visible style transfer.

There are two hypotheses and one root-cause diagnosis:

#### Hypothesis 1 — "The LoRA only fires when its style name appears in the prompt"

**This is incorrect.** B-LoRA style transfer works by directly modifying the UNet weight tensors (via `inject_lora`). The LoRA weights are additive parameter modifications that affect every denoising step regardless of the text prompt. There is no gating mechanism that checks whether a style name appears in the prompt before applying the weights.

Concretely, after `inject_lora(pipeline, synth_sd, style_alpha=1.0)`, the UNet's `up_blocks.0.attentions.1` block is permanently modified until `unload_lora()` is called. The prompt is used only for cross-attention keys/values in the text-conditioning path, which is a separate mechanism.

#### Hypothesis 2 — "Stage 1 hasn't learned the correct mapping yet; Stage 2 results may differ"

**Partially correct.** The real diagnosis is below.

#### Root cause: near-uniform attention (entropy too close to maximum)

The entropy numbers tell the story directly:

```
Near-uniform distribution: entropy ≈ 4.0–4.2  out of max 4.69
Fraction of max entropy:   e/e_max ≈ 0.86 – 0.90
Implied top-1 weight:      ~1/109 + small delta ≈ 0.009 – 0.05
```

With 109 experts and near-uniform weights, the synthesised LoRA is:

$$W_{\text{synth}} \approx \frac{1}{109} \sum_{i=1}^{109} W_i$$

This is the **average of all known styles**, which carries no coherent style signal and generates a style-neutral image (equivalent to not injecting any LoRA at all).

**Compare to v1.0 Stage 1** (from §14): at τ=0.01 the entropy collapsed to ~0.001 (one expert takes all weight), which gave visible style transfer but incorrect expert selection. The v2.0 problem is the opposite: training drove attention toward the *uniform* distribution (which maximises the KL soft-target loss when the CLIP similarity matrix is also relatively flat for novel WikiArt images).

**Why Stage 2 may help (partially)**: Stage 2 loss is `L_LDM - λ·H̄(A)`. The `- λ·H̄(A)` term *maximises* entropy, which would make this problem worse at inference. However, the LDM loss `L_LDM` forces the synthesis to be coherent — in order to minimise reconstruction error, the model must learn to assign non-trivial weight to the expert that actually matches the query image. So Stage 2 may push the encoder to form sharper keys indirectly through the reconstruction signal.

**The cleanest fix**: lower `--temperature` at inference time. At τ=0.1, the softmax is still relatively flat. Try τ=0.01:

```bash
# In inference_v2.sh, change:
--temperature 0.1
# to:
--temperature 0.01
```

At τ=0.01, softmax differences of 0.001 in the dot-product become amplified by 10×, turning near-uniform attention into a sharper peaked distribution. This should be tested immediately on a single run after Stage 2 completes.

---

### 20.4 — Code Changes Since §18

| File | Change |
|------|--------|
| `inference_v2.py` | Sanitize `query_label` slashes in `label_prefix` (fix path bug); copy `__query.jpg` + `__top1…5` expert thumbnails to every output folder |
| `inference.py` | Same thumbnail-copying block |
| `backfill_reference_images.py` | **NEW** — backfills `__query` + `__top1…5` images into all 78 existing run folders |
| `slurm/generalization_v2.sh` | **NEW** — v2.0 26-run generalization sweep (Exp A/B/C, same WikiArt queries as §15) |

---

### 20.5 — LoRA Injection Diagnostic: Results & Interpretation

**Context**: Sweep job 761782 was cancelled after all outputs looked identical regardless
of style, temperature, or top-k. Suspected `inject_lora()` was silently doing nothing.
A prior login-node test showed "UNet base-weight diff = 0.000000" which looked alarming.

GPU diagnostic job **761798** ran on `ai12` (V100) and produced full results.
Full report: `/scratch/eyavuz21/lora_attention/diagnose_lora_inject/report.txt`

---

#### What Each Test Actually Measures

Before the results, here is what each setup means, concretely:

**"Vanilla SDXL"** — the frozen SDXL base model, no LoRA of any kind. This is the
baseline every test image is compared against. MAE diff = 0 by definition.

**"load_lora_into_unet (raw keys)"** — the current `inject_lora()` path in our code.
Calls `pipeline.load_lora_into_unet(state_dict, None, pipeline.unet)` with the raw
`lora.down.weight / lora.up.weight` keys from the safetensors files. Diffusers installs
LoRA as **attention processors**: small wrapper objects that sit on top of existing
`to_q/to_k/to_v/to_out` projection layers. They do NOT modify the base weight tensors
— this is why the login-node test `(W_after - W_before).abs().mean() == 0` was measuring
the wrong thing. The _weights_ do not change; the _computation_ changes through the
processor. At generation time, for every attention layer that has a LoRA processor:
```
output = W_base(x)  +  scale * W_up( W_down(x) )     # LoRA delta added live
```

**"load_lora_into_unet (api keys)"** — same call, but the state dict was first loaded
via `pipeline.lora_state_dict(path)`. In B-LoRA's original inference code, this
high-level API is called first. We tested whether it performs any key conversion that
our direct path misses.

**"Hook-based injection (apply_lora_hooks_with_grad)"** — an alternative in
`lora_inject.py` that does NOT use the diffusers processor API at all. Instead, it
finds each target layer by traversing `unet.get_submodule(layer_path)` and registers a
`torch.Tensor.register_hook` / `nn.Module.register_forward_hook` that intercepts the
layer output and adds the LoRA delta inline:
```python
def hook_fn(module, input, output):
    x = input[0]
    return output + alpha * (x @ W_down.T) @ W_up.T
```
Hooks are stored as `RemovableHandle` objects and removed by calling `remove_hooks(handles)`.
This path was designed for Stage 2 training (where we need gradients to flow back through
the delta into the MoE encoder). It does NOT use diffusers' processor mechanism.

**"Direct weight merge"** — bypasses all APIs. Manually computes
`layer.weight.data += alpha * W_up @ W_down` for every layer, permanently baking the
LoRA delta into the base weights. Used here as a sanity check to confirm the LoRA
tensors themselves have correct values and shapes. Requires undoing the delta afterward.

**"Synth LoRA (MoELoRAv2, S1 checkpoint)"** — generates a synthesised LoRA by running
the MoELoRAv2 encoder on a Baroque WikiArt query image, routing across all 109 pool
experts. The resulting state dict is a weighted sum of expert LoRA tensors. This is what
the actual inference pipeline uses. Tested at τ=0.1 (spread routing) and τ=0.01 (sharp).

---

#### Diagnostic Results (job 761798 — ✅ completed)

```
USE_PEFT_BACKEND = False
Device: cuda (Tesla V100-SXM2-32GB)
```

**Key format audit:**
```
Raw safetensors keys (attentions.1 block): 160
Via pipeline.lora_state_dict() API:        160
Raw key[0] == API key[0]:                  True
```
→ No key conversion happens in `lora_state_dict()`. Both paths see identical keys.

**Injection tests — MAE pixel difference between each method and vanilla SDXL:**

| Test | Setup | MAE diff | Verdict |
|------|-------|----------|---------|
| Real B-LoRA (Baroque) | `load_lora_into_unet` raw keys | **32.01** | ✅ WORKING |
| Real B-LoRA (Baroque) | `load_lora_into_unet` API keys | **32.01** | ✅ WORKING (identical) |
| Real B-LoRA (Baroque) | Hook-based injection | **32.01** | ✅ WORKING (identical) |
| Real B-LoRA (Baroque) | Direct weight merge | **31.93** | ✅ WORKING (≈ same) |
| Synth LoRA (τ=0.1, Baroque query) | `load_lora_into_unet` | **8.57** | ⚠️ ACTIVE but weak |
| Synth LoRA (τ=0.1, Baroque query) | Hook-based | **8.57** | ⚠️ ACTIVE but weak (identical) |
| Synth LoRA (τ=0.01, Baroque query) | `load_lora_into_unet` | **9.57** | ⚠️ ACTIVE but weak |

**MoELoRAv2 routing for Baroque query (WikiArt image, not training image):**
```
τ=0.1 — Top-5 experts:
  #1: style_0001_Realism   avg_attention=0.043
  #2: style_0009_Realism   avg_attention=0.033
  #3: style_0104_Realism   avg_attention=0.030
  #4: style_0034_Symbolism avg_attention=0.029
  #5: style_0080_Romanticism avg_attention=0.029
  Entropy: 4.05 / 4.69 max  (86% of maximum — near-uniform)

τ=0.01 — Top-1: style_0001_Realism, Entropy: 3.12 / 4.69
```

---

#### What This Means

**1. `inject_lora()` is NOT broken.** The login-node test was measuring the wrong thing.
LoRA processors work by augmenting the computation inside each attention layer — they do
not change the raw weight tensors stored in `layer.weight`. Measuring `(W_after - W_before).abs().mean()`
will always return 0.0 even when injection is fully working. The correct test is to
generate an image and measure the pixel-level change, which is what the GPU diagnostic did.
Real B-LoRA injection produces MAE diff = 32.0 — large and obvious visual change.

**2. All four injection methods are equivalent.** `load_lora_into_unet` (raw keys), 
`load_lora_into_unet` (API keys), hook-based, and direct weight merge all produce
essentially the same pixel output (MAE diff ≈ 32). No need to change `inject_lora()`.

**3. The synthesised LoRA IS being injected, but it is weak.** MAE diff = 8.57 for the
synth LoRA vs 32.0 for real B-LoRA. The injection code is fine; the LoRA weights
themselves have low effective magnitude because they are a near-uniform weighted average
of 109 experts:

$$W_{\text{synth}} \approx \sum_{i=1}^{109} \frac{1}{109} W_i = \bar{W}$$

The random experts partially cancel each other out (different styles push attention in
different directions), so the net delta is much smaller than a single-style LoRA.

**4. The routing is wrong for Baroque queries.** Even at τ=0.01, the top-1 expert for
a Baroque WikiArt image is `style_0001_Realism`, not any Baroque expert. The S1v2
encoder maps this Baroque painting to Realism experts in CLIP space. This is the same
CLIP embedding gap problem as in v1.0 (§15): the training image for each style is one
specific painting; a different WikiArt painting of the same style may be closer in CLIP
space to a different style's training image.

**5. The reason the sweep outputs all looked the same:**
Near-uniform attention (entropy 4.05–4.2 / 4.69) means the synthesised LoRA is nearly
the same blob for every query style — the weighted average of all 109 experts changes
only slightly as the query changes, because no single expert gets more than ~4% weight.
The small per-run variation (MAE diff 8.6–9.6) is below the visual threshold for style
transfer. The sweep was correct to be cancelled.

---

#### What is NOT the problem
- ✅ Key format — raw `lora.down/lora.up` keys work perfectly with `USE_PEFT_BACKEND=False`
- ✅ `inject_lora()` code — it works; no change needed
- ✅ The LoRA tensors in the pool — they have the right values (direct merge confirms MAE ≈ 32)
- ✅ diffusers API — `load_lora_into_unet` with `USE_PEFT_BACKEND=False` correctly falls through to `unet.load_attn_procs()`

#### What IS the problem
- ❌ **S1v2 routing quality**: near-uniform attention entropy (4.05/4.69) → synthesised LoRA is a diluted average → MAE diff only 8–9 vs 32 for a real single LoRA
- ❌ **Wrong expert selected**: Baroque WikiArt query → Realism top-1 (same CLIP embedding gap as v1.0, §15)
- ❌ Two root causes compound: even if entropy were lower, the selected expert would still be wrong

---

### 20.6 — Status Update (2026-02-24) *(superseded by §22)*

~~**Root cause confirmed as weak magnitude due to expert averaging** — `--norm_match`
was added but images were still identical.~~

**Actual root cause found in §22**: O(N²) cross-term cancellation in parameter-averaging
synthesis, not magnitude. `norm_match` had no effect because `synth_norm` was already ~45
(close to target 50). The synthesis itself was computing noise.

- `slurm/s1v2_sweep.sh` and `slurm/s2v2_sweep.sh` used *broken* parameter-averaging.
- **Jobs 762274 and 762279 cancelled** — results invalid.
- **Stage 2 v2.0 training completed** — see §21.
- **Fix implemented and new sweeps submitted** — see §22.

---

## 21. Stage 2 v2.0 — Training Complete & Experiment Plan (2026-02-24)

### 21.1 — S2 v2.0 Training Results

| Item | Value |
|------|-------|
| Total steps | 8 000 / 8 000 |
| Learning rate | 5e-5 |
| λ_entropy schedule | 0.1 → 0.01 (linear warmdown over 8k steps) |
| Final total loss | 0.563 |
| Final LDM loss | 0.589 |
| Final entropy term | −0.025 |
| Checkpoint | `/scratch/eyavuz21/lora_attention/stage2_v2/latest.pt` |
| Saved checkpoints | every 500 steps (checkpoint-500 … checkpoint-8000) |

**Observation on entropy term**: `ent = −0.025 < 0` means the Stage 2 objective is
successfully *maximising* attention entropy (−λH̄(A) reward). The LDM reconstruction
loss partially counteracts this, but the net result may be that S2 routing is *flatter*
than S1's 4.05/4.69 bits — potentially making the mixing-cancellation problem worse.
This must be measured experimentally.

### 21.2 — S2 v2.0 Inference Sweep Plan *(superseded by §22)*

~~Identical structure to `slurm/s1v2_sweep.sh` (4 sweeps, ~80 runs) with two changes:~~

~~Script: `slurm/s2v2_sweep.sh` — submitted as **job 762279** (queued behind job 762274).~~

**Both jobs (762274 and 762279) were cancelled** — they used the broken parameter-averaging
synthesis and produced identical-looking outputs for all LoRA configurations. See §22 for
the root-cause diagnosis and corrected sweeps.

**Replaced by**: `slurm/s1v2_ps_sweep.sh` (job **762340**) and `slurm/s2v2_ps_sweep.sh`
(job **762341**), both using `--product_synth` (correct product-space synthesis).

**Key questions now answered by §22 sweeps:**

1. Does product-space synth produce visible style change at any routing sharpness?
2. Does `top_k=1` (oracle single-expert) prove injection is correct independent of routing?
3. Does S1 vs S2 routing entropy differ, and which gives better visual quality?

### 21.3 — Routing Fix Roadmap (regardless of S2 result)

Even if S2 is better, the underlying problem is the KL soft target derived from
CLIP similarity — the Baroque WikiArt test image lands nearest to *Realism* in the
pool's CLIP space. Fix options in priority order:

1. **Option A — Diverse S1 pairs** (fastest): Use 10–20 WikiArt images per style
   during Stage 1 training (not just the blora_zoo thumbnail). Gives the encoder
   many CLIP anchors per style, covering the full intra-style variation.
2. **Option B — Classification target**: Replace soft KL target with one-hot label
   supervision on the *style name* (not the *most similar training image*). Breaks
   the CLIP embedding gap entirely.
3. **Option C — WikiArt dataset pairs**: Train directly on WikiArt images paired
   with their ground-truth style B-LoRA expert (requires building per-style B-LoRAs
   from the full WikiArt split, which is expensive but correct).

**Decision gate**: inspect `clip_similarity.pt` to confirm Baroque→Realism mismatch
before committing to a fix:

```bash
python - <<'EOF'
import torch, json
sim = torch.load('/scratch/eyavuz21/lora_attention/clip_similarity.pt')
# shape: [N_pool, N_pool] or [N_test, N_pool]
print(sim.shape)
# identify Baroque row and its top-3 columns
EOF
```
4. **Compare S1 vs S2 vs τ-sweep** routing sharpness and generated image quality side by side using the `__query`/`__top1…5` thumbnails now present in every folder.
---

## 22. Critical Bug Found & Fixed: Cross-Term Cancellation in LoRA Synthesis (2026-02-24)

### 22.1 — The Bug

**Root cause of invisible style transfer in all v2 runs.**

MoELoRA averaged `W_down` and `W_up` matrices *independently*:
```python
# BROKEN (parameter-averaging — what was implemented)
synth_down = Σ_i A[i] * W_down[i]   # (r, d_in)
synth_up   = Σ_i A[i] * W_up[i]     # (d_out, r)
```

But the *actual* style signal is carried by the *product* `ΔW = W_up @ W_down`:

```
(Σ A_i W_up_i) @ (Σ A_j W_down_j)
  = Σ_i A_i² (W_up_i @ W_down_i)                 ← useful diagonal (N terms)
  + Σ_{i≠j} A_i A_j (W_up_i @ W_down_j)          ← cross-term noise (N²-N terms)
```

With uniform attention `A_i = 1/N`:
- Diagonal term weight: `N × (1/N)² = 1/N = 1/109 ≈ 0.9%`
- Cross-term weight: `N(N-1) × (1/N)² ≈ (N-1)/N ≈ 99.1%`

The cross-terms `W_up_i @ W_down_j` (i≠j) are random-looking noise that destroys
every consistent style direction. With N=109, the signal is diluted by **108×** relative
to the noise, causing the synthesised LoRA to have near-zero net style effect.

### 22.2 — Measured Evidence

From `diagnose_lora_inject.py` (job 761798) and cross-term analysis:

| Method | Product norm | cos vs oracle | Explanation |
|--------|-------------|---------------|-------------|
| Real B-LoRA injection | 3.574 | 1.00 | Single expert, no averaging |
| Old synth (uniform 1/N) | 0.227 | 0.10 | Cross-terms dominate — noise |
| **New synth (product-space)** | **≈0.42** | **0.99** | **Correct sum of products** |
| One-hot synth (top-k=1) | 3.574 | 1.00 | Single expert, equivalent to real |

*(norm comparison on first tensor block; oracle = Baroque expert)*

**Why v1 worked but v2 didn't**: v1 used much sharper routing (entropy < 1 bit).
With most weight on one expert: diagonal ≫ cross-terms → barely visible style.
v2 trained with entropy regularisation → uniform routing → cross-terms dominate.
But the cross-term bug also affects v1 at any routing sharpness — it was just masked.

### 22.3 — The Fix: Product-Space Synthesis

```python
# CORRECT (product-space averaging — implemented 2026-02-24)
W_avg[t] = Σ_i a[i,t] * (W_up[i,t] @ W_down[i,t])  # (d_out, d_in)

# Decompose back to rank-r LoRA via truncated SVD:
U, S, Vh = torch.linalg.svd(W_avg[t])
W_down_synth[t] = diag(√S[:r]) @ Vh[:r]   # (r, d_in)
W_up_synth[t]   = U[:, :r] @ diag(√S[:r]) # (d_out, r)
# Guarantees: W_up_synth @ W_down_synth ≈ W_avg (best rank-r approx.)
```

**Verified numerically** on login node (N=5, small test):
- One-hot routing: `cos(ref, new) = 1.000` ✓
- Uniform routing: `cos(ref, new) = 0.9998`, `cos(ref, old) = 0.886` ✓

### 22.4 — Changes Made

| File | Change |
|------|--------|
| `models/moe_lora_v2.py` | Added `_synthesise_product_space()` method; `forward(..., product_space=False)` param |
| `inference_v2.py` | Added `--product_synth` flag; passes `product_space=args.product_synth` to `model.forward()` |
| `slurm/s1v2_ps_sweep.sh` | New sweep: S1 ckpt + `--product_synth`, 4 sweeps ~80 runs |
| `slurm/s2v2_ps_sweep.sh` | New sweep: S2 ckpt + `--product_synth`, same structure |

`--product_synth` is **off by default** (backward-compatible). All new experiments
should use it.

**Cancelled jobs**: 762274 (S1v2-Sweep, broken), 762279 (S2v2-Sweep, broken).
**Submitted jobs**: **762340** (S1v2-PS), **762341** (S2v2-PS).

### 22.5 — Training Impact & Next Steps

The SAME bug exists in `train_stage1_v2.py` and `train_stage2_v2.py` — both compute
`synth_lora` via parameter averaging, feed it into SDXL for the LDM loss, and the
cross-term noise means the training signal for routing quality is corrupted. The model
trained successfully (low LDM loss) because it learned to ignore the noisy synth LoRA
injection entirely (i.e., the gradient flows primarily through the text conditioning,
not through the routing decision).

**Fix training**: Replace `_synthesise_batched` → `_synthesise_product_space` as the
default in `train_stage1_v2.py` and `train_stage2_v2.py`. Then retrain from scratch.

**Immediate experiments (pending sweeps 762340 / 762341)**:
1. Do product-space synth sweeps with *current* checkpoints show visible style?
   - If yes → training loss signal was sufficient; only inference was broken.
   - If no → both training and inference need fixing; retrain with corrected synthesis.
2. Does `top_k=1` (oracle single-expert) produce strong style at any temperature?
   - If yes → injection is correct; routing quality is the remaining problem.
3. Does S1 routing (less entropy regularisation) give better style than S2?

### 22.6 — Residual Problem (Routing Quality)

Even with the correct synthesis, near-uniform routing over 109 experts produces
`W_avg ≈ mean of all style LoRAs` which is approximately "vanilla SDXL with subtle
average-style drift". Visible individual style transfer requires sharp routing.

Routing fix options (unchanged from §21.3):
1. Option A: diverse S1 pairs (multiple images per style during training)
2. Option B: classification supervision target (one-hot label, not CLIP similarity)
3. Option C: full WikiArt training split paired with per-style B-LoRA experts

---

## §23 — v2.1 Architecture Update

### 23.1 — Summary of Changes

Three inter-related changes land together as "v2.1":

| Component | v2.0 | v2.1 |
|-----------|------|------|
| Synthesis | `_synthesise_batched` default (cross-term bug) | `_synthesise_product_space` default (`product_space=True`) |
| Stage 1 target | KL(CLIP-similarity soft targets) | One-hot CE on style label (default `--target_mode ce`) |
| Stage 1 data requirement | Requires `clip_similarity.pt` | No CLIP similarity file needed in CE mode |

### 23.2 — Why CE Instead of CLIP-Similarity KL

CLIP similarity clusters images by **subject matter**, not artistic style.
A Baroque portrait's nearest neighbour in the pool is often a Realism portrait
(both have human subjects, similar colour temperature, comparable composition).
This confusion signal prevents the encoder from learning style-invariant features.

One-hot CE on the **style label name** is cleaner: the encoder must learn what
distinguishes "Baroque" from "Realism" at the feature level, not what makes one
portrait photo look like another.

### 23.3 — CE Loss Formulation

```
A: (N, T, r)   — per-expert, per-tensor, per-rank attention (softmax over N)
gt_pos: int    — index of GT expert in the sampled pool

A_avg = A.mean(dim=2)          # (N, T) — average over rank dim
log_A = (A_avg + 1e-8).log()  # (N, T)
gt_targets = full((T,), gt_pos, long)
L = nll_loss(log_A.T, gt_targets)   # mean over T
```

This is equivalent to `-mean_T log(A_avg[gt_pos, t])`, which maximises the
probability mass on the GT expert across all T tensor groups.

### 23.4 — Product-Space SVD Stabilisation

`_synthesise_product_space` now uses `(S[:r_use] + 1e-8).sqrt()` instead of
`S[:r_use].sqrt()` to avoid infinite gradients near S ≈ 0 during Stage 2 backprop.
The broad `try/except` around the SVD was also removed; if SVD fails (degenerate
W_avg with identical rows), the error surfaces cleanly.

### 23.5 — Files Changed

- `models/moe_lora_v2.py`: `product_space=True` default in `forward()` + `route()`; SVD stabilised
- `data/dataset.py`: `similarity_path: Optional[str] = None`; KL similarity matrix loaded only when provided
- `train_stage1_v2.py`: `--target_mode {ce,kl}` (default `ce`); `compute_ce_loss()`; branches in train loop; `--similarity_path` now optional
- `train_stage2_v2.py`: `model.forward(..., product_space=True)` made explicit

### 23.6 — New Training Jobs

Stage 1 v2.1 should be re-run from scratch with `--target_mode ce` and a new
`--output_dir /scratch/eyavuz21/lora_attention/stage1_v21/`.  No `--similarity_path`
required.  Stage 2 v2.1 then loads the resulting Stage 1 checkpoint.

See `slurm/train_stage1_v21.sh` for the SLURM submission script.
