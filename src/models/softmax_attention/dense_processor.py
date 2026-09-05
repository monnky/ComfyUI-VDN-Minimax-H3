"""Dense (full-sequence) softmax attention for the STOCK H3 attention module, on
FlexAttention's FlashAttention-4 backend. This is the teacher trunk's kernel in Stage A1
-- a model kernel, not trainer machinery.
"""
from functools import partial
import torch

try:
    from torch.nn.attention.flex_attention import flex_attention
except (ImportError, ModuleNotFoundError):
    flex_attention = None

try:
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
except (ImportError, ModuleNotFoundError):
    def _apply_rotary_emb(x, cos, sin):
        rotary_dim = cos.shape[-1]
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        cos = cos.to(x.dtype)[None, :, None, :]
        sin = sin.to(x.dtype)[None, :, None, :]
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = torch.cat((-x2, x1), dim=-1)
        return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)



class FlexFA4Processor:
    """MiniMaxH3AttnProcessor with the attention call swapped for FlexAttention's
    FlashAttention-4 (CuTeDSL) backend, which is faster than cuDNN SDPA at the H3 shape
    for both forward and forward+backward. Everything around the call (qkv projections,
    per-head norms, rotary) mirrors the stock processor, so LoRA-wrapped projections
    apply as-is.
    """

    _flex_flash = None  # compiled once per process, shared by every block

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("H3 packs one request per document; no attention mask expected.")
        if FlexFA4Processor._flex_flash is None:
            FlexFA4Processor._flex_flash = torch.compile(
                partial(flex_attention, kernel_options={"BACKEND": "FLASH"})
            )

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # flex takes (B, H, S, D); the processor layout is (B, S, H, D).
        out = FlexFA4Processor._flex_flash(
            query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
        )
        out = out.permute(0, 2, 1, 3).flatten(2, 3).type_as(query)
        out = attn.to_out[0](out)
        return attn.to_out[1](out)
