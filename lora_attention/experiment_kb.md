# MoELoRA Experiment Knowledge Base

This file is the lightweight, living knowledge base for the `lora_attention`
pipeline. It records what we have learned from training and inference runs so
we do not repeat invalidated experiments.

## Current Rules

- Treat product-space synthesis as the default.
- Treat any run that uses legacy parameter averaging as suspect unless it is a
  deliberate ablation.
- For exact-instance retrieval, prefer supervision tied to the source exemplar
  image rather than a coarse WikiArt category tag.
- When comparing visible style transfer, use neutral prompts as the primary
  visibility check; style prompts can compete with or mask the LoRA signal.
- Do not trust image quality alone; record routing entropy, top-1 expert, and
  whether the run used `top_k=1` or `norm_match`.
- Prefer chained train -> validate runs so each checkpoint gets a direct
  follow-up inference pass.
- When older checkpoints look visually flat under neutral prompts, treat the
  result as a likely weak checkpoint or weak style magnitude problem before
  blaming prompt wording alone; confirm with an `alpha` or `norm_match`
  sweep.

## What Has Worked

| Area | Evidence | Why it matters |
|------|----------|----------------|
| Product-space inference with explicit `--product_synth` | `EXPERIMENTS.md` §22 and the product-space sweeps | This avoids the O(N^2) cross-term cancellation bug from parameter averaging. |
| Oracle-style `top_k=1` comparisons | Existing sweep scripts and alpha diagnostics | Gives us a quick upper bound on what the routing can do when it becomes sharp. |
| Norm matching for visibility checks | `slurm/alpha_diag.sh` and inference logs | Useful when the router is diffuse and the synth LoRA is too weak to see. |
| Reference B-LoRA baseline runs | Existing sweeps and diagnostics | Confirms whether the synth pipeline is actually worse than direct expert injection. |
| Exact-instance Stage 1 supervision | `MoELoRA-Mini-Exact-S1-1007514.log` | Better matches the real retrieval task when each LoRA corresponds to one exemplar image. |
| Neutral prompt as the visibility check | Exact-mini Stage 2 outputs (`neutral_soft` / `neutral_top1`) | Neutral prompts expose the style signal more cleanly than style prompts in this setup. |
| Style-vs-neutral retrospective sweeps on the old style-only v2 checkpoints | `slurm/retro_prompt_sweep_v1.sh` | Isolates whether the earlier "no visible style" reports were partly a prompt artifact rather than a routing failure. |
| Exact-mini Stage 2 neutral outputs | `mini_exact_v1/outputs/stage2_exact_followup_validation/*/neutral_*` | Neutral prompts can produce strong visible style transfer when retrieval is sharp and the exact exemplar is in the pool. |
| Mini generalization replay on old checkpoints | `slurm/mini_generalization/neutral_generalization_mini_v1.sh` | Diagnostic-only replay; it does not train a fresh checkpoint for generalization. |

## What Did Not Work

| Area | Evidence | Failure mode |
|------|----------|--------------|
| Legacy parameter averaging in v2 inference sweeps | `MoELoRA-Inf-v2-832063.log` | Attention stayed near-uniform and the synthesized style signal was washed out. |
| Stage 2 v2 training with `product_space=False` | `train_stage2_v2.py`, `MoELoRA-S2v23-808394.log` | The model learned under the wrong synthesis rule, so the checkpoint and evaluation did not match the intended pipeline. |
| Inference sweep that omitted `--product_synth` | `slurm/inference_sweep_v2.sh` before this cleanup | The sweep was exercising the legacy path even though the codebase had already identified it as broken. |
| Prompt-only style cue dependence | Stage 2 dataset logic in `data/dataset.py` | The text prompt can carry too much of the style signal, which hides routing weaknesses. |
| Category-level Stage 1 supervision for single-image LoRAs | Coarse mini run + current label mapping | WikiArt categories are too coarse when the target LoRA comes from one specific exemplar image. |
| Old neutral replay on `stage1_v21` / `stage2_v22` / `stage2_v23` | `neutral_generalization_mini_v1` outputs | Routing remains near-uniform and images stay visually close across in-pool / holdout / zero-shot branches, so the old checkpoints still look weak under neutral prompts. |
| Mini generalization v2 `expA_inpool` visual outputs | `/scratch/eyavuz21/lora_attention/mini_generalization_v2/stage1_train/neutral_mini_v2/expA_inpool` | `neutral_soft` failed across reviewed cases; `neutral_top1` was mixed, with Baroque working but Cubism weak and Fauvism irrelevant/black-and-white. Routing top-1 success does not reliably imply visible style transfer. |
| Follow-up `expA_inpool_next` diagnostic | `/scratch/eyavuz21/lora_attention/diagnostics/expA_inpool_next_20260426_162640` | Direct/reference and synthesized outputs can alter images, but generated styles do not reliably match the query image. Baroque transfers a generic Baroque-like style but not the query style; Cubism misses the top-1/query style; Fauvism norm-match is somewhat closer but still not exact, while other variants are black-and-white/unrelated. |
| Direct-vs-MoE identity benchmark (`A dog`, `A dog in <style> style`) | `/scratch/eyavuz21/lora_attention/diagnostics/identity_benchmark_20260426_181500` | User visual review: outputs are not related to the intended reference style and should be treated as broken. Forced top-1 routing does not recover the direct B-LoRA look. |

## Active Hypotheses

| Hypothesis | Status | Next check |
|------------|--------|------------|
| Product-space synthesis plus `top_k=1` should expose whether routing is the real bottleneck | Unverified for the latest v2.3 checkpoint | Run a small post-train validation sweep on the fresh checkpoint. |
| Product-space training should make the train/infer objective consistent | Unverified | Re-run Stage 2 after flipping the synthesis default. |
| A smaller chained validation sweep will be more informative than a huge grid | Likely true | Use a short style + neutral validation pass after every training job. |
| Exact-instance supervision should sharpen routing more than category labels | Promising, but still being tested | Compare the exact mini run against the earlier coarse 4-style mini run. |
| Style-only v2 sweep results may have hidden the LoRA effect by over-specifying the prompt | Being tested now | Retrospectively re-run `stage2_v23` and `stage1_v21` with neutral prompts and compare against the old style-prompt folders. |
| Neutral-only generalization replay should tell us whether the style effect survives held-out and zero-shot queries | Planned | Run the mini neutral generalization benchmark on `stage1_v21`, `stage2_v22`, and `stage2_v23`, then inspect the out-of-pool outputs. |
| Increasing `style_alpha` or using `norm_match` may reveal whether the old flat outputs are caused by weak magnitude rather than total absence of LoRA effect | Newly motivated | Run a small neutral-prompt alpha sweep on the old checkpoints and compare against the current neutral replay. |
| A compact training run on a small WikiArt subset is needed to make the mini generalization benchmark real | Newly added | Train a fresh mini Stage 1 generalization checkpoint, then replay the neutral benchmark against that new model. |
| The next mini canary should use only singleton styles and judge only in-pool retrieval first | Newly added | Train a singleton-only Stage 1 mini benchmark and evaluate just the in-pool neutral cases before trying holdout or zero-shot again. |
| Mini generalization v2 may be solving routing without producing usable image-space style transfer | Supported by visual review | Compare direct reference B-LoRA outputs against synthesized top-1 outputs for the same prompts, and test whether alpha/norm matching fixes Cubism/Fauvism before expanding beyond in-pool. |
| Query-style identity is not preserved even when style effect is visible | Supported by `expA_inpool_next` visual review | Treat current router/synthesis as style-category transfer at best, not query-specific style matching; next benchmark should include query/reference alignment checks, not just visible stylization. |
| Forced top-1 synthesis still does not resemble the direct reference B-LoRA baseline | Supported by the `identity_benchmark_20260426_181500` visual review | De-prioritize more routing sweeps. The next debugging step should compare the actual synthesized tensors against the direct expert tensors and inspect the injection/code path, because the failure now looks structural rather than just retrieval-related. |

## Next-Step Decision Tree

Use the exact mini validation summary to choose the next run:

| Validation outcome | What it means | Next action | Submit script |
|--------------------|---------------|-------------|---------------|
| `good` | Exact-instance routing is strong enough to test through diffusion | Run a tiny exact Stage 2 follow-up on the same 4-expert setup | [`submit_mini_exact_good_stage2_v1.sh`](/home/eyavuz21/repos/MoLoRAs/lora_attention/slurm/mini_exact/submit_mini_exact_good_stage2_v1.sh) |
| `partial` | The router is learning, but the distribution is still too soft | Rerun Stage 1 with more views and sharper training temperature | [`submit_mini_exact_partial_v1.sh`](/home/eyavuz21/repos/MoLoRAs/lora_attention/slurm/mini_exact/submit_mini_exact_partial_v1.sh) |
| `bad` | The learnability question is still unresolved | Shrink to a 2-style toy overfit setup and check whether the router can memorize at all | [`submit_mini_exact_toy_overfit_v1.sh`](/home/eyavuz21/repos/MoLoRAs/lora_attention/slurm/mini_exact/submit_mini_exact_toy_overfit_v1.sh) |

Operational notes:
- The partial-outcome rerun writes to `mini_exact_v1/outputs/stage1_sharp`.
- The good-outcome Stage 2 follow-up writes to `mini_exact_v1/outputs/stage2_exact_followup`.
- The bad-outcome toy setup prepares a separate root at `/scratch/eyavuz21/lora_attention/mini_exact_toy_v1`.

## Run Ledger Template

Append a row for every meaningful run.

| Date | Job | Stage | Checkpoint | Synthesis | Prompt type | Result | Verdict | Notes |
|------|-----|-------|------------|-----------|-------------|--------|---------|-------|
| 2026-04-19 | 1007787 | mini_stage2_exact_validation | `/scratch/eyavuz21/lora_attention/mini_exact_v1/outputs/stage2_exact_followup/latest.pt` | product_space | mixed | runs=16 top1_acc=1.000 mean_gt_rank=1.000 entropy=0.3036 | good | Chained post-train check for the tiny exact Stage 2 follow-up. |
| 2026-04-19 | 1007757 | mini_stage2_exact_followup | `/scratch/eyavuz21/lora_attention/mini_exact_v1/outputs/stage2_exact_followup/latest.pt` | diffusion_followup | exact_exemplar | total=0.610998 ldm=0.610998 top1=1(0.256) | good | Tiny exact Stage 2 follow-up on the same 4-expert exact-instance setup. |
| 2026-04-19 | 1007514 / 1007515 | mini_stage1_exact | `/scratch/eyavuz21/lora_attention/mini_exact_v1/outputs/stage1/latest.pt` | routing_only | exact_exemplar | samples=32 top1_acc=1.000 loss=1.168327 entropy=1.3753 | good | Exact-instance Stage 1 canary with style-preserving augmentations and mixed 4-expert negatives; all 4 styles hit top-1 on validation. |
| 2026-04-19 | 1007291 / 1007292 | mini_stage1_category | `/scratch/eyavuz21/lora_attention/mini_v1/outputs/stage1/latest.pt` | routing_only | seen_styles | samples=32 top1_acc=0.750 loss=1.302836 entropy=1.3834 | partial | Coarse WikiArt-category proxy was learnable, but routing stayed near-uniform. |
| 2026-04-20 | planned | mini_neutral_generalization | `/scratch/eyavuz21/lora_attention/neutral_generalization_mini_v1/` | product_space | neutral | held-out singleton + zero-shot replay | pending | Neutral-first mini generalization benchmark queued for `stage1_v21`, `stage2_v22`, and `stage2_v23`. |
| 2026-04-20 | planned | neutral_alpha_visibility_sweep | `/scratch/eyavuz21/lora_attention/neutral_alpha_sweep_v1/` | product_space | neutral | alpha and norm-match visibility test | pending | Test whether the older flat neutral outputs become visible when LoRA magnitude is increased. |
| 2026-04-20 | 1008980 / 1008981 / 1008982 | mini_generalization_train_eval | `/scratch/eyavuz21/lora_attention/mini_generalization_v1/` | CE training + neutral replay | neutral | compact WikiArt subset training followed by generalization replay | pending | First actual training run for the mini generalization benchmark; the earlier neutral mini jobs were replay-only diagnostics. |
| 2026-04-20 | planned | mini_generalization_v2_train_eval | `/scratch/eyavuz21/lora_attention/mini_generalization_v2/` | CE training + in-pool neutral replay | neutral | singleton-only stricter Stage 1 canary | pending | Tightened rerun after v1 learned fuzzy neighbors instead of reliable in-pool expert identity. |
| 2026-04-20 | 1011330 / 1011331 | mini_generalization_v2_train_eval | `/scratch/eyavuz21/lora_attention/mini_generalization_v2/stage1_train/latest.pt` | product_space | neutral | expA_inpool visual review: `neutral_soft` failed; `neutral_top1` mixed; Baroque good, Cubism weak, Fauvism irrelevant | partial | Stage 1 trained sharply and often ranked GT #1, but image-space style transfer reliability is low. |
| 2026-04-26 | 1031909 | expA_inpool_next_diagnostic | `/scratch/eyavuz21/lora_attention/diagnostics/expA_inpool_next_20260426_162640` | reference + synth_top1 + norm_match | neutral | visible style changes but poor query-style match; Baroque generic, Cubism miss, Fauvism only partially closer under norm-match | partial | Confirms the issue is not only weak magnitude; current outputs do not faithfully match the query style identity. |
| 2026-04-26 | 1032369 | direct_vs_moe_identity_benchmark | `/scratch/eyavuz21/lora_attention/diagnostics/identity_benchmark_20260426_181500` | reference + synth_top1 + norm_match | dog neutral + dog style-word | user review: outputs unrelated to intended styles; benchmark treated as broken | bad | Forced top-1 still fails to reproduce the direct expert baseline, so the remaining problem is likely in synthesis representation or injection compatibility rather than only routing. |
| 2026-04-19 | local | mini_stage1_validation | `/scratch/eyavuz21/lora_attention/mini_v1/outputs/stage1/latest.pt` | routing_only | seen_styles | samples=32 top1_acc=0.750 loss=1.302836 entropy=1.3834 | partial | Mini routing-only validation on held-out seen-style images. |
| 2026-03-24 | 832063 | Inference | `stage1_v21/latest.pt` | Legacy | Style prompt | Near-uniform routing, weak style effect | Bad | Missing product synth in the sweep that produced the run. |
| 2026-03-19 | 808394 | Train | `stage2_v23/latest.pt` | Legacy | WikiArt stage 2 | Loss moved, but router stayed diffuse | Partial | Training and evaluation were not aligned with the intended synthesis rule. |

## Suggested Post-Run Verdicts

- `good`: visible style transfer, routing is interpretable, and the run beats vanilla SDXL.
- `partial`: some signal exists, but routing is diffuse or quality is inconsistent.
- `bad`: output is visually flat, routing is near-uniform, or the run used a known-broken path.
- `invalidated`: the run is not comparable because the code path or settings were wrong.
