"""RMSNorm with fp32 second-moment accumulation, nn.RMSNorm-compatible.

Parameterisation matches nn.RMSNorm(dim, eps): weight [dim], ones-init, so
`linear_attention.norm.weight` loads into either implementation.

Why not F.rms_norm / nn.RMSNorm directly. At the branch's shape ([F*S, H, d] with
F*S ~ 100k, H*d = 7168, bf16):

    x.pow(2).mean(dtype=fp32)   ~1e-3 relative error   small extra memory
    x.float().pow(2).mean()     exact                  a full fp32 copy of x
    vector_norm(dtype=fp32)^2   ~1e-7 relative error   almost no extra memory

`pow(2)` runs in the INPUT dtype and rounds every square to bf16's 8 mantissa bits
before the fp32 accumulation ever starts; promoting first is exact but materialises
the fp32 copy. vector_norm squares and accumulates in fp32 without materialising
anything. The weight is likewise cast DOWN, never the input up -- a bf16 x fp32
elementwise mul would silently promote the whole tensor.
"""
import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = torch.linalg.vector_norm(
            x, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / x.shape[-1]
        return x * torch.rsqrt(ms + self.eps).to(x.dtype) * self.weight.to(x.dtype)
