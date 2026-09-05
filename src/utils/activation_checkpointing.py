"""A finer gradient-checkpoint boundary than "one whole DiT block".

Stage B checkpoints each block (diffusers' enable_gradient_checkpointing), so only one
block is ever live — but that one block is expensive: at H3 sequence length the hybrid
block's backward transient is tens of GiB, split between attention, the linear branch
(frame_statistics' contiguous copies are ~1.5 GiB each) and the MLP (fc1's output
alone is several GiB in bf16).

All of that is live simultaneously because the checkpoint boundary sits outside the
whole block. Putting a second boundary around `attn` and `ff` means that when the MLP
is being backwarded the attention's intermediates have already been discarded, and vice
versa: the peak becomes max(attn, ff) instead of attn + ff. The outer per-block
checkpoint STAYS — without it the block's residual/modulation intermediates would be
retained at every depth, which is far worse.

    split_block_checkpoints(model)      # after build_model, before fully_shard

Patches `forward` rather than wrapping the module in a new parent, so module identity,
parameter names and every isinstance check keep working — `hybrid_new_parameters` and
`iter_hybrids` both walk `block.attn` expecting a HybridAttention, and checkpoint
keys must not gain a level.
"""
import torch
import torch.utils.checkpoint


def checkpoint_forward(module):
    """Make `module`'s forward its own checkpoint region. Idempotent."""
    if getattr(module, "_checkpointed", False):
        return module
    inner = module.forward

    def forward(*args, **kwargs):
        # Under no_grad there is nothing to recompute later and checkpoint would only
        # warn; inference and Stage A's teacher trunk both run that way.
        if not torch.is_grad_enabled():
            out = inner(*args, **kwargs)
        else:
            out = torch.utils.checkpoint.checkpoint(inner, *args, use_reentrant=False,
                                                    **kwargs)

        return out

    module.forward = forward
    module._checkpointed = True
    return module


def split_block_checkpoints(model, parts=("attn", "ff")):
    """Give each named part of every DiT block its own checkpoint region.

    Returns the number of regions installed. Nesting inside the block-level checkpoint
    is supported by non-reentrant checkpointing: on the block's recompute the inner
    regions again save nothing, so each one is materialised only while its own backward
    runs.
    """
    n = 0
    for block in model.transformer_blocks:
        for name in parts:
            mod = getattr(block, name, None)
            if mod is not None:
                checkpoint_forward(mod)
                n += 1
    return n
