"""Small learnable layers of the linear branch. The branch norm is
ops.rms_norm.RMSNorm (nn.RMSNorm-compatible) -- no branch-specific norm class.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from src.models.ops.rms_norm import RMSNorm  # noqa: F401  (the branch norm)

class FrameKDAAlpha(nn.Module):
    """alpha_t = exp(-exp(A_log) * softplus(delta + bias)) per frame / head / state
    channel — KDA's double-exponential gate, in the official fla layout (layers/kda.py):
    A_log is [H] PER HEAD, the per-channel freedom is dt_bias [H*d_k], and the down/up
    data path (rank = head_dim, no biases) starts LIVE at torch default init. Init
    copies fla verbatim: A ~ U(1,16) per head, dt ~ log-uniform[1e-3, 1e-1] per channel,
    giving an initial alpha spectrum ~0.20..0.999 instead of a uniform constant, and a
    gradient |dalpha/dA_log| ~ A*dt that is orders of magnitude larger than under a
    uniform-constant init, where A_log barely moves."""

    def __init__(self, hidden_size, num_heads, head_dim, bottleneck=None):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        bottleneck = bottleneck or head_dim                      # fla: rank = head_dim
        self.down = nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, num_heads * head_dim, bias=False)
        # fla kda.py init, verbatim (incl. the 1e-4 clamp and inverse-softplus).
        self.A_log = nn.Parameter(
            torch.log(torch.empty(num_heads, dtype=torch.float32).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(num_heads * head_dim, dtype=torch.float32)
            * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, frame_mean_x, heads=None):                # [F, hidden]
        # `heads` (inference under Ulysses): evaluate this head range only -- `up`'s rows,
        # dt_bias and A_log take the slice; `down` is shared by every head.
        if heads is None:
            up_w, dt_bias, a_log, n_heads = (self.up.weight, self.dt_bias, self.A_log,
                                             self.num_heads)
        else:
            rows = slice(heads.start * self.head_dim, heads.stop * self.head_dim)
            up_w, dt_bias, a_log = self.up.weight[rows], self.dt_bias[rows], self.A_log[heads]
            n_heads = heads.stop - heads.start

        # autocast OFF for the same reason frame_statistics and _run_scans turn it off:
        # `down`/`up` are Linears, autocast intercepts them at the op level, and a
        # `.float()` afterwards would promote an ALREADY-bf16 delta — the promotion
        # looks like a guarantee and provides none.
        #
        # It matters because the scan multiplies alpha across all frames, so a
        # per-element error compounds: a bf16 alpha's tail errors, after ~100 frames,
        # move the retention (how much of the far state survives the clip) by tens of
        # percent for the worst channels.
        #
        # The WEIGHTS are promoted too, not just the input. With autocast off nothing
        # reconciles dtypes any more, so an fp32 input against a bf16-stored `down`
        # (inference keeps the branch in bf16 to match the checkpoint) would raise
        # "expected mat1 and mat2 to have the same dtype". Casting the input alone was
        # never fp32 math anyway: it would have been an fp32 activation multiplied by
        # an 8-mantissa-bit weight.
        with torch.autocast(device_type=frame_mean_x.device.type, enabled=False):
            delta = F.linear(frame_mean_x.float(), self.down.weight.float())
            delta = F.linear(delta, up_w.float())
            delta = delta + dt_bias.float()
            scale = torch.exp(a_log.float())[:, None]           # [H, 1], broadcast d_k
            delta = delta.view(-1, n_heads, self.head_dim)
            return torch.exp(-scale * F.softplus(delta.float()))   # [F,H,d_k] fp32

