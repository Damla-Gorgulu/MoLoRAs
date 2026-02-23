"""
lora_attention: Rank-Level Attention MoE for LoRA Style Transfer.

Architecture:
  - LoRAPool:   Loads and caches all expert LoRAs from the B-LoRA zoo.
  - RoutingMLP: Projects LoRA features → Key matrix K_i ∈ ℝ^{rank × clip_dim}.
  - MoELoRA:    Full pipeline: CLIP query → rank attention → Hadamard synthesis.

Training:
  - Stage 1 (train_stage1.py): MSE loss on attention matrix vs one-hot GT.
  - Stage 2 (train_stage2.py): LDM diffusion loss with hold-out pool.

Inference:
  - inference.py: Synthesise LoRA for a query image and generate styled outputs.
"""
