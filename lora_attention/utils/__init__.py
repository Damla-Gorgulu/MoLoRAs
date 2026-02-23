try:
    from .lora_inject import (
        inject_lora,
        unload_lora,
        LoRAInjectionContext,
        apply_lora_hooks_with_grad,
        remove_hooks,
        compute_ldm_loss,
    )
    __all__ = [
        "inject_lora",
        "unload_lora",
        "LoRAInjectionContext",
        "apply_lora_hooks_with_grad",
        "remove_hooks",
        "compute_ldm_loss",
    ]
except ImportError:
    # Allow lightweight submodules (e.g. clip_similarity) to import without
    # pulling in the full torch/diffusers stack.
    __all__ = []
