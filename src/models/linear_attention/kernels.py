"""Inference-only fused epilogue: RMSNorm + output gate in one pass."""
import torch
import torch.nn.functional as F

_LINEAR_EPILOGUE_CACHE = {}


def _linear_epilogue_body(readout, weight, gate, eps):
    """RMSNorm + the output gate, as one expression. See `linear_epilogue`."""
    ms = torch.linalg.vector_norm(
        readout, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / readout.shape[-1]
    normed = readout * torch.rsqrt(ms + eps).to(readout.dtype) * weight.to(readout.dtype)
    return (normed * gate).reshape(readout.shape[0], -1)


def _linear_epilogue_fhsd_body(readout, weight, gate, eps):
    """`_linear_epilogue_body` for a readout still in [F, H, S, d], with the transpose back
    to token order folded into the store."""
    ms = torch.linalg.vector_norm(
        readout, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / readout.shape[-1]
    normed = readout * torch.rsqrt(ms + eps).to(readout.dtype) * weight.to(readout.dtype)
    frames, heads, per_frame, dim = normed.shape
    rows = frames * per_frame
    return (normed.permute(0, 2, 1, 3).reshape(rows, heads * dim)
            * gate.reshape(rows, heads * dim))


def linear_epilogue(readout, weight, gate, eps, inference=False, fhsd=False):
    """The branch's last two steps, fused when the caller says it is inferring.

    Eager this is a noticeable share of the linear branch for arithmetic that is pure
    bandwidth: the readout is [F*S, H, d] = ~1.4 GiB at H3 scale, and the norm, the
    gate product and the multiply each walk all of it and write all of it back. One
    kernel reads it once.

    `fhsd=True` takes the readout as [F, H, S, d] and folds the transpose back to token
    order into the store: with q kept in [F,H,S,d] the readout is a plain batched
    matmul instead of an einsum that permutes q in and the result out, and the epilogue
    was writing a full copy anyway. The two layouts are BITWISE equal -- both spellings
    do the same multiplies on the same values, only the traversal order differs.

    NOT BITWISE against the eager spelling -- inductor keeps the reciprocal and the two
    multiplies in fp32 and rounds once at the store. Same class of difference as
    `_qk_prep`, and in the same direction (fewer roundings, not more), which is why
    the training body keeps the eager one: a teacher whose job is to be numerically
    comparable to a student must not quietly get the better arithmetic.
    """
    body = _linear_epilogue_fhsd_body if fhsd else _linear_epilogue_body

    if not inference:
        out = body(readout, weight, gate, eps)
    else:
        key = "fhsd" if fhsd else "fn"
        if key not in _LINEAR_EPILOGUE_CACHE:
            _LINEAR_EPILOGUE_CACHE[key] = torch.compile(body, dynamic=False)
        out = _LINEAR_EPILOGUE_CACHE[key](readout, weight, gate, eps)

    return out


