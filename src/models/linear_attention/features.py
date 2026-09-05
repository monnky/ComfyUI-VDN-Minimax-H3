"""Q/K/V feature preparation for the linear branch: separable short conv,
SiLU, L2 norm -- eager, compiled and Triton-tconv paths.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.ops.temporal_conv import temporal_conv_activate

def _temporal_shift(x, w, k, pad):
    """Depthwise k-tap conv over frames as shift-multiply-add. x [T, S, C]; w [C, k]
    (already in x's dtype); zero-padded, symmetric.

    The TRAINING spelling (inference has the Triton kernel in ops/temporal_conv.py,
    forward only). It is not written as F.conv1d/conv3d because inductor fuses this
    pointwise chain, taps and the SiLU/L2Norm tail together, forward and backward,
    while a cudnn depthwise temporal conv is a black box that runs well below
    bandwidth here; at the real shape the compiled shift is several times faster than
    either conv spelling, and all of them agree to within one bf16 ulp.
    """
    xp = F.pad(x, (0, 0, 0, 0, pad, pad))
    out = None
    for dt in range(k):
        part = xp[dt:dt + x.shape[0]] * w[:, dt].view(1, 1, -1)
        out = part if out is None else out + part
    return out


_TCONV = {"fn": None}


def _temporal_conv(x, w, k, pad):
    # Compiled on CUDA (several times faster than the eager shift), eager on CPU;
    # latched on the first call of the process. Not a knob.
    if _TCONV["fn"] is None:
        _TCONV["fn"] = torch.compile(_temporal_shift) if x.is_cuda else _temporal_shift
    return _TCONV["fn"](x, w, k, pad)


# --------------------------------------------------------------------------------------

_FEATURES_CACHE = {}


def _activate_body(tokens, l2norm):
    """SiLU [+ L2Norm], PRESERVING the input dtype.

    The .to() is load-bearing for TRAINING: under the align hook's bf16 autocast,
    F.normalize's norm is an autocast-fp32 op, so the division promotes the whole
    feature tensor to fp32 -- and frame_statistics runs inside a DELIBERATE
    autocast-off island (its A must stay fp32), where the fp32 k then meets the bf16 v
    in B's matmul and training dies with "expected BFloat16 but found Float". Casting
    back to the input dtype gives the norm accumulated in fp32 with the output in the
    compute dtype -- fla's L2Norm semantics. Inference is unaffected: its inputs are
    bf16 with no autocast, so normalize never promotes there.
    """
    x = F.silu(tokens)
    return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype) if l2norm else x


def _activate_fhsd_body(tokens, l2norm, num_frames, per_frame):
    """`_activate_body` storing [F, H, S, d] instead of [F*S, H, d]."""
    x = _activate_body(tokens, l2norm)
    heads, dim = x.shape[-2], x.shape[-1]
    return x.view(num_frames, per_frame, heads, dim).permute(0, 2, 1, 3).contiguous()


def _tconv_activate_body(x, w, k, pad, heads, head_dim, l2norm):
    """Reference for the Triton kernel: the eager chain, one expression."""
    out = _temporal_shift(x, w, k, pad).reshape(-1, heads, head_dim)
    return _activate_body(out, l2norm)


def prepare_linear_features(tokens, l2norm, conv=None, proj=None, num_frames=None,
                            frame_size=None, fhsd=None):
    """One linear-branch projection's features, EAGER: [conv ->] SiLU [-> L2Norm].
    The op sequence the checkpoints were trained under; the training body's path.

    `fhsd=(F, S)` stores q frame-major ([F, H, S, d]) instead of token-major.
    """
    if conv is not None and proj in conv.projs:
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        x, w_tm = conv.spatial(proj, tokens, num_frames, frame_size)
        out = _temporal_conv(x, w_tm, conv.KERNEL, conv.KERNEL // 2)
        out = _activate_body(out.reshape(-1, heads, head_dim), l2norm)
    elif fhsd is not None:
        out = _activate_fhsd_body(tokens, l2norm, *fhsd)
    else:
        out = _activate_body(tokens, l2norm)

    return out


def prepare_linear_features_inference(tokens, l2norm, conv=None, proj=None, num_frames=None,
                                      frame_size=None, fhsd=None):
    """`prepare_linear_features` as fused kernels. INFERENCE ONLY.

    Eager, the L2Norm is the single worst-utilised op in the linear branch --
    F.normalize computes the norm in one kernel and divides in another, so a ~1.35 GiB
    tensor is walked three times to apply one scale per row of 128. Where the short
    conv runs, the temporal shifts fuse in as well and the conv output never reaches
    HBM at all: one kernel for shift-add-SiLU-normalise instead of four (the Triton
    kernel in ops/temporal_conv.py; Triton is required).

    `fhsd=(F, S)` writes q frame-major: the store is strided either way, so it is free
    here and saves the readout a permute in and a permute out.

    NOT bitwise vs the eager function (one rounding at the store instead of one per
    op); same story and the same direction as `linear_epilogue`.
    """
    if conv is not None and proj in conv.projs:
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        x, w_tm = conv.spatial(proj, tokens, num_frames, frame_size)
        out = temporal_conv_activate(x, w_tm, conv.KERNEL, conv.KERNEL // 2,
                                     heads, head_dim, l2norm)
    elif fhsd is not None:
        out = _compiled(f"act_fhsd_{l2norm}", _activate_fhsd_body)(tokens, l2norm, *fhsd)
    else:
        out = _compiled(f"act_{l2norm}", _activate_body)(tokens, l2norm)

    return out


def _compiled(key, body):
    """One static-shape compile per (body, flags), built on first use."""
    if key not in _FEATURES_CACHE:
        _FEATURES_CACHE[key] = torch.compile(body, dynamic=False)
    return _FEATURES_CACHE[key]


class LinearAttentionSepConv(nn.Module):
    """Depthwise short conv on the linear branch's projections named in `projs` (the
    released checkpoints convolve K and V and leave Q as the raw NoPE features). Per
    convolved projection: depthwise 5x5 SPATIAL conv per frame, then a depthwise 5-tap
    TEMPORAL conv across frames (both bias-free). Effective 3D kernel = outer product
    w_tm x w_sp — 30 free parameters per channel instead of 125, restricted to rank-1
    space-time.

    Why separable rather than a dense 5^3 depthwise Conv3d: the full 5^3 depthwise
    conv3d has no fast kernel anywhere (ATen, cudnn, torch.compile and channels-last
    all route to or lose against a naive kernel), while the halves ride tuned paths:
    cudnn's NHWC depthwise-2D and Inductor-fused temporal shifts. That is more than an
    order of magnitude cheaper per projection. SANA-WM ships the same factorization.

    Layout: ZERO copies end to end. The token layout [F*S, C] read as [T, H, W, C]
    IS the channels_last memory format of [T, C, H, W], so the cudnn NHWC conv
    consumes a permute VIEW; its channels_last output permutes back to [T, S, C]
    contiguous for free — exactly what the temporal shifts want.

    Temporal boundary: zero padding, bidirectional (non-causal), crosses VAE chunks.
    Init keeps unit variance per stage: spatial std = 1/sqrt(25 taps), temporal
    std = 1/sqrt(5 taps). Random, NOT identity (SANA-WM inits identity): near-teacher
    behaviour at step 0 is owned by the output gate, and Stage A1 trains the branch
    from scratch anyway.
    """

    KERNEL = 5

    def __init__(self, channels, projs=("k", "v")):
        super().__init__()
        self.projs = tuple(projs)
        k = self.KERNEL
        for name in self.projs:
            sp = nn.Conv2d(channels, channels, k, padding=k // 2,
                           groups=channels, bias=False)
            nn.init.normal_(sp.weight, std=(k * k) ** -0.5)
            tm = nn.Conv1d(channels, channels, k, padding=k // 2,
                           groups=channels, bias=False)
            nn.init.normal_(tm.weight, std=k ** -0.5)
            setattr(self, f"{name}_sp", sp)
            setattr(self, f"{name}_tm", tm)

    def spatial(self, proj, tokens, num_frames, frame_size):
        """The 5x5 depthwise half, and the temporal weight the other half needs.

        Split out of `apply` so the inference path can put its own epilogue on the
        temporal shifts (see `prepare_linear_features_inference`) instead of writing
        the conv output to HBM and reading it straight back for the SiLU. `apply` is
        the same two halves in a row."""
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        channels = heads * head_dim
        # [F*S, H, d] -> channels_last VIEW of [T, C, gh, gw]; cudnn NHWC kernel
        volume = (tokens.reshape(num_frames, grid_h, grid_w, channels)
                  .permute(0, 3, 1, 2))
        w_sp = getattr(self, f"{proj}_sp").weight
        volume = F.conv2d(volume, w_sp, padding=self.KERNEL // 2, groups=channels)

        # channels_last output -> [T, S, C] contiguous view -> fused temporal shifts.
        # The fp32 master weight is cast explicitly: elementwise mul is not on the
        # autocast list, and fp32 x bf16 would silently promote the whole pass.
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        w_tm = getattr(self, f"{proj}_tm").weight.squeeze(1).to(x.dtype)   # [C, K]
        return x, w_tm

    def apply(self, proj, tokens, num_frames, frame_size):
        if proj not in self.projs:                       # unlisted projections pass through
            return tokens
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        x, w_tm = self.spatial(proj, tokens, num_frames, frame_size)
        out = _temporal_conv(x, w_tm, self.KERNEL, self.KERNEL // 2)
        return out.reshape(-1, heads, head_dim)

    def forward(self, q_raw, k_raw, v_raw, num_frames, frame_size):
        return (self.apply("q", q_raw, num_frames, frame_size),
                self.apply("k", k_raw, num_frames, frame_size),
                self.apply("v", v_raw, num_frames, frame_size))


