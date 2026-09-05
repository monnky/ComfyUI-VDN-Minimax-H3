"""Fused softmax-side kernels: rope, QK preparation, the softmax gate."""
import torch
import torch.nn.functional as F

try:
    from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
except (ImportError, ModuleNotFoundError):
    _apply_rotary_emb = None



def _rope_body(x, cos, sin):
    """`_apply_rotary_emb` verbatim, as one expression for inductor to fuse.

    Kept a copy rather than importing and compiling diffusers' function, because what is
    being compiled has to be readable next to what it replaces: any drift between this
    and _apply_rotary_emb would be a silent numerics change on the inference path only.
    """
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    cos = cos.to(x.dtype)[None, :, None, :]
    sin = sin.to(x.dtype)[None, :, None, :]
    x1, x2 = x_rot.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x_rot * cos + rotated * sin, x_pass), dim=-1)


_QK_PREP_CACHE = {}


def _qk_prep_body(t, weight, eps, cos, sin):
    """QK-norm and rope as one expression for inductor. [T, H, d] in and out."""
    t = F.rms_norm(t, (t.shape[-1],), weight, eps)
    return _rope_body(t.unsqueeze(0), cos, sin).squeeze(0)


def _qk_prep(t, weight, eps, cos, sin):
    """Everything between the q (or k) projection and the attention call. INFERENCE ONLY.

    Eager this is two full passes over a ~1.4 GiB tensor -- RMSNorm writes one copy and
    the rope writes another; fused it is one pass. flex reads the [T, H, d] layout as a
    view (see window_softmax_flex), so the store is contiguous.

    NOT bitwise vs the eager pair -- one rounding at the store instead of one per op,
    about one bf16 ulp relative, and closer to fp32 than the eager chain, not further.
    """
    if "fn" not in _QK_PREP_CACHE:
        _QK_PREP_CACHE["fn"] = torch.compile(_qk_prep_body, dynamic=False)
    return _QK_PREP_CACHE["fn"](t, weight, eps, cos, sin)


_SOFTMAX_GATE_CACHE = {}


def _softmax_gate_body(softmax_out, gate):
    return (softmax_out * gate.to(softmax_out.dtype)).reshape(softmax_out.shape[0], -1)


def apply_softmax_gate(softmax_out, gate, inference=False):
    """The per-head mass gate and the flatten to_out wants, as one kernel: eager the
    multiply is one pass over ~1.4 GiB and the reshape a second full copy; compiled it
    is one read and a store already in to_out's layout.
    """
    if not inference:
        out = _softmax_gate_body(softmax_out, gate)
    else:
        if "fn" not in _SOFTMAX_GATE_CACHE:
            _SOFTMAX_GATE_CACHE["fn"] = torch.compile(_softmax_gate_body, dynamic=False)
        out = _SOFTMAX_GATE_CACHE["fn"](softmax_out, gate)

    return out

