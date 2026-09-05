"""One Triton kernel for the linear branch's temporal conv, its SiLU, and its L2 norm.

WHY. `_tconv_activate_body` hands the chain to inductor, which is already a large win
over eager (the five shifted multiply-adds fuse into one pass, and the SiLU and the norm
ride along). What it cannot fix is that the shift spelling READS EACH ELEMENT FIVE
TIMES: `xp[dt:dt+T] * w[:, dt]` is a full-tensor expression per tap, so a ~1.35 GiB
tensor is walked five times for a 5-tap stencil whose taps overlap almost completely.

The kernel below tiles over FRAMES. One program owns BLOCK_T consecutive frames, one
spatial row and one head, and issues its five taps as five loads of the same
[BLOCK_T, head_dim] shape offset by dt -- which overlap by BLOCK_T - 1 rows, so the
hardware cache serves all but the first and HBM sees each frame about
(BLOCK_T + 4) / BLOCK_T times. That is roughly a 2x speedup over the compiled shift
chain at the real shape; BLOCK_T = 16 was the best of 8/16/32 there.

WHAT IT DOES NOT DO. The 5x5 spatial half stays on cudnn: its NHWC depthwise kernel
beats every alternative tried, including a compiled 25-tap stencil.

INFERENCE ONLY, like everything else on this path: the kernel accumulates in fp32 and
rounds once, where the eager chain rounds after every tap, so it is not bitwise (about
one bf16 ulp relative, and closer to an fp32 reference than the chain it replaces).

Triton and a CUDA tensor are REQUIRED: the inference path has no other implementation
of this stage. The kernel's contract -- the 5-tap symmetric stencil, a power-of-two
head_dim >= 16 (the L2 norm is a reduction inside one program's channel block),
contiguous operands -- is checked and violated with a ValueError, not a silent
alternative path.
"""

import torch
import triton
import triton.language as tl

BLOCK_T = 16


@triton.jit
def _tconv_act_kernel(X, W, OUT, T, S_, C_, BLOCK_T: tl.constexpr,
                      D_: tl.constexpr, L2: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_h = tl.program_id(2)
    chan = pid_h * D_ + tl.arange(0, D_)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    valid = rows < T

    acc = tl.zeros((BLOCK_T, D_), dtype=tl.float32)
    for dt in tl.static_range(5):
        r = rows + dt - 2
        ok = valid & (r >= 0) & (r < T)                    # zero padding, both ends
        v = tl.load(X + (r[:, None] * S_ + pid_s) * C_ + chan[None, :],
                    mask=ok[:, None], other=0.0).to(tl.float32)
        wd = tl.load(W + chan * 5 + dt).to(tl.float32)
        acc += v * wd[None, :]

    y = acc * tl.sigmoid(acc)                              # SiLU
    if L2:
        inv = 1.0 / tl.sqrt(tl.maximum(tl.sum(y * y, axis=1), 1e-12))
        y = y * inv[:, None]
    tl.store(OUT + (rows[:, None] * S_ + pid_s) * C_ + chan[None, :],
             y.to(OUT.dtype.element_ty), mask=valid[:, None])


def temporal_conv_activate(x, w, kernel, pad, heads, head_dim, l2norm):
    """x [T, S, C] (CUDA, contiguous), w [C, 5] -> [T*S, heads, head_dim]."""
    if not x.is_cuda:
        raise ValueError("temporal_conv_activate is a Triton kernel; x must be on CUDA")
    if kernel != 5 or pad != 2:
        raise ValueError(f"the kernel unrolls exactly 5 symmetric taps, got kernel={kernel} "
                         f"pad={pad}")
    if head_dim & (head_dim - 1) or head_dim < 16:
        raise ValueError(f"head_dim must be a power of two >= 16 (tl.arange), got {head_dim}")
    T, S_, C_ = x.shape
    if C_ != heads * head_dim:
        raise ValueError(f"C={C_} != heads*head_dim={heads * head_dim}")
    if not (x.is_contiguous() and w.is_contiguous()):
        raise ValueError("x and w must be contiguous")
    out = torch.empty_like(x)
    _tconv_act_kernel[(triton.cdiv(T, BLOCK_T), S_, heads)](
        x, w, out, T, S_, C_, BLOCK_T=BLOCK_T, D_=head_dim, L2=l2norm,
        num_warps=4, num_stages=2)
    return out.reshape(-1, heads, head_dim)
