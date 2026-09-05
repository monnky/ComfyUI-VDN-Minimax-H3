"""OutputGate: the sigmoid gate BOTH branches put on their output.

Shared on purpose -- the softmax side instantiates it per-head (init 0.99: the softmax
branch IS the teacher at step 0) and the linear side per-channel with init="random" (a live
low-rank path from step 0, matching fla's g_proj).
"""
import math

import torch
from torch import nn


class OutputGate(nn.Module):
    """Sigmoid gate on a branch's output, zero-initialised so it starts at `init_value`.

    Both branches gate their output, but at different granularity:

      SOFTMAX (head_dim=None) one value per (token, head), broadcast over channels. The
            windowed softmax renormalises to 1 no matter how little mass it saw, so this
            gate scales the softmax branch back toward the share it captured — a scalar
            property of a DISTRIBUTION, hence per-head. Starts at 0.99: the softmax
            branch IS the teacher at step 0.
      LINEAR  (head_dim set)  one value per (token, head, channel): a routing decision on
            a new pathway. Low rank (bottleneck = head_dim by default).

    The init matters, especially on the linear branch: to_out_linear and every writer
    feeding the scan reach the loss ONLY through this gate, so the optimiser's cheapest
    descent direction is to close it and starve them all at once. A constant init near
    sigmoid(0.5) is the point of maximum gate mobility and invites exactly that collapse;
    `init="random"` (linear-branch gate) instead keeps torch's default init on both
    linears, matching fla's g_proj — a LIVE low-rank path, input-dependent from step 0.
    The softmax gate always uses `init="constant"`: it is the mixing gate between the
    two branches (no fla analogue) and its 0.99 is what keeps the softmax branch at the
    teacher on step 0.
    """

    def __init__(self, hidden_size, num_heads, head_dim=None, bottleneck=None,
                 init_value=0.9, init="constant"):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.init_value = init_value
        out_features = num_heads * (head_dim or 1)
        self.down = None if bottleneck is None else nn.Linear(hidden_size, bottleneck,
                                                              bias=False)
        self.up = nn.Linear(bottleneck or hidden_size, out_features, bias=True)
        if init == "constant":        # gate == init_value for every token at step 0
            nn.init.zeros_(self.up.weight)
            nn.init.constant_(self.up.bias, math.log(init_value / (1.0 - init_value)))
        elif init != "random":        # torch default on both linears: live from step 0
            raise ValueError(f"OutputGate init must be 'constant' or 'random', got {init!r}")

    def forward(self, x):
        """x: [tokens, hidden] -> gate in (0, 1), [tokens, H, d] or [tokens, H, 1]."""
        if self.up.weight.dtype != x.dtype or self.up.weight.device != x.device:
            self.to(dtype=x.dtype, device=x.device)
        gate = torch.sigmoid(self.up(x if self.down is None else self.down(x)))
        return gate.view(-1, self.num_heads, self.head_dim or 1)


