# MoELoRA v3 Synthesis Roadmap

## Goal

The goal is not top-1 routing. The goal is to learn a function:

```text
query style image + frozen LoRA pool -> synthesized LoRA weights
```

The attention distribution is only the mechanism. Success is judged by whether the synthesized LoRA behaves like a useful style adapter for the query image.

## What We Already Proved

The v3 query/key architecture can learn exact-instance routing on the 4-style mini pool.

Validation:

```text
CLIP v3 strong: top1_acc=1.0, mean_gt_rank=1.0
VAE v3:         top1_acc=1.0, mean_gt_rank=1.0
```

This only proves that the learned image-query and LoRA-key spaces can align. It does not prove synthesis or image-space style transfer.

## Next Objective

Train the v3 model with a tensor-space synthesis loss:

```text
A = attention(query_image, LoRA_weight_keys)
synth_down, synth_up = weighted LoRA synthesis using A
DeltaW_synth = synth_up @ synth_down
DeltaW_target = W_up_target @ W_down_target
loss = product-space reconstruction loss(DeltaW_synth, DeltaW_target)
```

This directly trains the model to construct LoRA deltas, not just classify experts.

## Phase 1: In-Pool Reconstruction Canary

Ground-truth expert is included in the pool.

Purpose:

- verify the v3 attention -> synthesis -> tensor-loss path is correct
- verify gradients flow through synthesis into query/key transformers
- verify generated images are produced without injection errors

Expected result:

- tensor cosine similarity should increase
- relative reconstruction error should decrease
- generated v3 images may resemble direct B-LoRA if the selected tensor subset is sufficient
- if images are weak, that is not fatal yet because this canary may synthesize only a subset of tensor groups

## Phase 2: Hold-Out Reconstruction Canary

Ground-truth expert is excluded from the pool.

Purpose:

- test whether v3 can construct a missing style from other LoRAs
- this is closer to the real research goal

Expected result:

- harder than Phase 1
- may fail on only 4 styles because prior linear-composition experiments showed LoRA deltas are close to orthogonal
- if it fails, increase pool diversity before changing architecture

## Phase 3: Image-Space Check

For each trained checkpoint, generate comparison grids:

```text
query image | vanilla SDXL | direct B-LoRA | v3 synthesized LoRA
```

Prompt starts neutral:

```text
A dog
```

This checks whether tensor reconstruction produces visible style transfer.

## Current Mini Jobs

Initial submitted jobs should run only the Phase 1 in-pool synthesis canary first, with dependent inference jobs.

Stop criterion for this round:

- jobs start successfully
- training log appears
- inference job produces image files or a clear error log

We do not need to wait for a perfect scientific conclusion before inspecting the first image outputs.
