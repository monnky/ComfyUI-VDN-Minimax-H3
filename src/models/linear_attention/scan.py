"""Frame statistics, the bidirectional scans, and the state gather.

For a query frame t with softmax window [lo, hi]:

    left  = prefix_states[lo - 1]        # frames 0..lo-1, decayed to t via bridge
    right = suffix_states[hi + 1]        # frames hi+1..F-1, decayed to t via bridge
    linear_state[t] = bridge(left) + bridge(right)

i.e. exactly the state of everything OUTSIDE the window, in the query frame's frame
of reference. Sequence ends are masked (or read the text state when one was given);
bridge="alpha" decays through the window's span. gather_linear_state's docstring
below carries the full contract.
"""
import collections
import contextlib

import torch

def frame_statistics(kf, vf, beta, a_fp32=True, inference=False):
    """Per-frame delta-rule statistics, shared by the branch and its test.

        A[f,h,k,l] = sum_s k[f,h,s,k] beta[f,h,s] k[f,h,s,l]
        B[f,h,v,k] = sum_s v[f,h,s,v] beta[f,h,s] k[f,h,s,k]

    Written as batched matmuls rather than an fp32 einsum: an fp32 einsum materialises
    three [F,H,S,d] copies (~3 GB each at H3 scale) and misses the tensor cores. Only
    the small [F,H,d,d] results are promoted to fp32, which is what the scan needs.

    The .contiguous() calls are load-bearing, not defensive: kf/vf arrive from
    .permute(0,2,1,3), so the contraction axis S carries stride H*d — cuBLAS cannot
    vectorise those loads and the batched GEMM collapses to a fraction of its
    throughput. The explicit repack is about 2x faster and bit-exact.

    MUST run with autocast OFF (the decorator): an ambient bf16 autocast intercepts
    matmul at the OP level, so the explicit `.float()` promotions below are not enough —
    autocast would re-downcast the operands, silently turning the fp32 A this function
    exists to guarantee back into a bf16 one (and materialising the useless fp32 copies
    anyway). Stage A's align hook runs the student under autocast; this guard is what
    keeps the A statistics exact there.
    """
    with torch.autocast(device_type=kf.device.type, enabled=False):
        return _frame_statistics(kf, vf, beta, a_fp32, inference=inference)


_STATS_PREP_CACHE = {}


@contextlib.contextmanager
def _tf32_matmul():
    """TF32 for the fp32 GEMM inside, restored on the way out. INFERENCE ONLY.

    A = (k*beta)^T k is computed in fp32 rather than bf16 because bf16's 8 mantissa bits
    break the conditioning I+A depends on (see `_frame_statistics`). TF32 has 10, and on
    real activations that is enough: the smallest eigenvalue of I+A is unchanged to
    many decimals and A itself differs from fp32 by ~1e-5 relative where bf16 differs
    by ~1e-3. The GEMM runs about 2x faster.

    Scoped to the one matmul on purpose. The flag is global, and turning it on for the
    whole branch would also put factor_apply and the scan's fp32 bmms on tensor cores;
    that is a bigger numerical change for a much smaller saving, and it is not this
    function's call to make.

    Not a graph-capture hazard: setting a backend flag launches nothing, and a captured
    graph replays whichever kernel cuBLAS selected while it was set.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _frame_stats_prep_body(kf, vf, beta):
    """The four operands the two GEMMs need, off one read of k and one of v.

    Every `.contiguous()` here is load-bearing, for a reason that survives the fusion:
    kf and vf arrive from `.permute(0, 2, 1, 3)`, so the contraction axis S carries
    stride H*d, and a cuBLAS batched GEMM handed those strides drops to a fraction of
    its throughput. Dropping the calls because "the fused kernel reads it once anyway"
    makes the prologue faster and the fp32 A GEMM downstream so much slower that the
    whole of frame_statistics loses. The stores have to be contiguous, not just few.
    """
    kf16 = kf.contiguous()
    kf32 = kf16.float()
    scaled32 = (kf32 * beta.unsqueeze(-1).float()).contiguous()
    vb = (vf * beta.unsqueeze(-1).to(vf.dtype)).contiguous()
    return kf16, kf32, scaled32, vb


def _frame_stats_prep(kf, vf, beta, inference=False):
    """Everything before the two GEMMs in `_frame_statistics`, as one kernel.

    Eager it is five passes over tensors that are ~1.35 GiB in bf16 and twice that in
    fp32: kf.contiguous(), vf.contiguous(), kf.float(), kf32 * beta32,
    (vf * b).contiguous() -- most of what `frame_statistics` costs, all of it well
    below bandwidth. One kernel reads k and v once each and writes exactly the four
    operands cuBLAS is about to consume.

    BITWISE, unlike the other fused kernels here, and it is worth saying why: nothing is
    reassociated. A widening cast is exact, and the two multiplies keep their operand
    dtypes (fp32 x fp32 for A's, bf16 x bf16 for B's). Only the number of round trips to
    HBM changes. The dropped `vf.contiguous()` is likewise not a numerical change --
    the multiply produced a contiguous tensor either way.
    """
    if not inference:
        out = _frame_stats_prep_body(kf, vf, beta)
    else:
        if "fn" not in _STATS_PREP_CACHE:
            _STATS_PREP_CACHE["fn"] = torch.compile(_frame_stats_prep_body, dynamic=False)
        out = _STATS_PREP_CACHE["fn"](kf, vf, beta)

    return out


def _frame_statistics(kf, vf, beta, a_fp32, inference=False):
    kf, kf32, scaled32, vb = _frame_stats_prep(kf, vf, beta, inference=inference)
    if a_fp32:
        # A IN FP32, B LEFT IN BF16. A is the one the scan inverts, and bf16 breaks a
        # property it needs. A = sum_s beta_s k_s k_s^T is symmetric by construction, but
        # computed as (k*b)^T @ k the (k,l) and (l,k) entries multiply differently-ROUNDED
        # operands, so the result is only symmetric to bf16 precision. On real
        # activations that asymmetry is large enough to push the smallest eigenvalue
        # of I+A well below the 1 the maths guarantees; torch.linalg.cholesky reads
        # the lower triangle only, so it then factorises an indefinite matrix and
        # vdn_solve dies. Random keys do NOT reproduce this — real patches within a
        # frame are strongly correlated, so A's off-diagonals (and their absolute
        # rounding error) are large.
        if inference:
            with _tf32_matmul():
                A = torch.matmul(scaled32.transpose(-1, -2), kf32)
        else:
            A = torch.matmul(scaled32.transpose(-1, -2), kf32)
    else:
        A = torch.matmul((kf * beta.unsqueeze(-1).to(kf.dtype)).contiguous()
                         .transpose(-1, -2), kf).float()
    # Free, and independent of dtype: guarantees cholesky factorises the matrix we mean.
    A = 0.5 * (A + A.transpose(-1, -2))

    # B is a plain readout, never inverted, and its error enters the state linearly —
    # bf16 is fine and it keeps the tensor cores on the one that costs.
    B = torch.matmul(vb.transpose(-1, -2), kf).float()                     # [F,H,d_v,d_k]
    return A, B


def _run_scans(backend, alpha, A_raw, B_raw, text_state=None):
    """Phase A (batched) then phase B (serial): forward and reverse state banks.

    `text_state` [H,d_v,d_k] replaces the zero start of BOTH scans (see
    BidirectionalLinearBranch._text_state). Both directions get the same state, so every
    frame's two directional states carry the prompt; frames themselves are still
    injected exactly once per direction.

    Phase B is a plain linear recurrence state_t = state_{t-1} @ transition_t +
    injection_t, so both the loop and its backward are one bmm per frame — no
    factorisation inside the loop.

    The WHOLE scan runs with autocast OFF. An ambient bf16 autocast intercepts matmul
    at the OP level, so `.float()` on the inputs is not enough: it re-downcasts the
    result, the VDN path additionally hands bf16 to cuSOLVER (which has no bf16
    Cholesky kernel), and the recurrence itself is fp32 state math on [F,H,d,d]
    tensors — small enough that the fp32 bmms cost nothing.

    """
    with torch.autocast(device_type=A_raw.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, A_raw, B_raw)
        num_frames = transitions.shape[0]                          # [F,H,d,d], [F,H,d_v,d]

        start = (torch.zeros_like(injections[0]) if text_state is None
                 else text_state.to(injections.dtype))
        state = start
        forward_states = []
        for frame in range(num_frames):
            state = state @ transitions[frame] + injections[frame]
            forward_states.append(state)

        state = start
        reverse_states = [None] * num_frames
        for frame in range(num_frames - 1, -1, -1):
            state = state @ transitions[frame] + injections[frame]
            reverse_states[frame] = state

        return torch.stack(forward_states), torch.stack(reverse_states)


def _run_scans_inference(backend, alpha, A_raw, B_raw, text_state=None):
    """Same recurrence as `_run_scans`, same numbers, less live memory.

    `_run_scans` accumulates F states into a python list and then torch.stack()s it --
    the list and the stack are two full copies of an [F,H,d,d] fp32 bank, and the
    forward list is still alive while the reverse scan builds its own. At H3 scale
    (F=102, H=56, d=128) each bank row is 3.5 MiB, so the peak inside that function is
    about 1.4 GiB per layer. Writing straight into a preallocated tensor holds one bank
    per direction instead of two: ~0.7 GiB.

    That spelling is NOT safe to use in training. Index-assigning into a preallocated
    tensor inside the loop makes each write a node in the graph the recurrence already
    depends on; `torch.stack` on a list built by a chain of out-of-place ops is the
    shape autograd handles cleanly here. So the two spellings stay separate rather than
    one being "fixed" to look like the other.

    Values are bit-for-bit identical to `_run_scans`: same ops in the same order, only
    the destination differs. `torch.baddbmm(inj, state, trans, out=bank[frame])` is the
    same three operands as `state @ transitions[f] + injections[f]` followed by a store,
    with cuBLAS doing the add in its fp32 epilogue instead of a separate kernel and a
    separate copy -- three launches per frame become one, which is close to 2x on a
    recurrence that is launch-bound rather than compute-bound.
    """
    with torch.autocast(device_type=A_raw.device.type, enabled=False):
        transitions, injections = backend.factor_apply(alpha, A_raw, B_raw)
        num_frames = transitions.shape[0]

        start = (torch.zeros_like(injections[0]) if text_state is None
                 else text_state.to(injections.dtype))
        prefix = torch.empty((num_frames, *start.shape), dtype=injections.dtype,
                             device=injections.device)
        suffix = torch.empty_like(prefix)

        state = start
        for frame in range(num_frames):
            torch.baddbmm(injections[frame], state, transitions[frame],
                          out=prefix[frame])
            state = prefix[frame]

        state = start
        for frame in range(num_frames - 1, -1, -1):
            torch.baddbmm(injections[frame], state, transitions[frame],
                          out=suffix[frame])
            state = suffix[frame]

        return prefix, suffix


BRIDGE_MODES = ("alpha", "none")


def gather_linear_state(prefix_states, suffix_states, alpha, bounds, bridge="alpha",
                     text_state=None, inference=False, out_dtype=None):
    """Everything OUTSIDE the softmax window, summarised in the query frame's frame of
    reference. [F,H,d_v,d_k].

    prefix_states[j] holds frames 0..j and suffix_states[j] holds frames j..F-1, so for a
    query frame t with window [lo, hi] the complement is exactly prefix_states[lo-1] plus
    suffix_states[hi+1] — one frame outside on each side, nothing counted twice. Frames
    at the sequence ends have no neighbour on one side; the index is clamped so the
    gather stays in range and the contribution is then masked to zero (a vectorised
    gather cannot skip elements).

    `bridge` chooses how those two states reach frame t:

      "alpha"  multiply in prod_u alpha_u over the frames in between: advance the
               recurrence through the window while pretending those frames wrote
               nothing — which, for this branch, they did not (softmax covers them, so
               the full transition would double count). The design default.
      "none"   use the states exactly as gathered. Decouples alpha's two jobs at the
               cost of making the output gate relearn the overall gain — and the gate is
               the most collapse-prone part of the module. The bridge is NOT a constant
               factor across frames, so to_out_linear cannot absorb it.

    `text_state` [H,d_v,d_k] is the state the two scans STARTED from. With
    it, a query whose window already touches a clip end does not get zero from that
    side: it gets the text state, decayed in over exactly the frames between the
    boundary and t — the same arithmetic the interior rows get, since the scans' virtual
    index -1 (forward) and F (reverse) both hold it. Without it (text_state=None) the ends
    contribute nothing, which is the original complement-of-the-window semantics.

    The whole block is vectorised over frames on purpose: a per-frame python loop
    builds one autograd node per frame per direction, and their backward — not the
    forward — dominates the linear branch's training cost.

    `inference=True` runs the arithmetic as one compiled kernel and `out_dtype` folds the
    downcast the caller would otherwise do into its store. `_gather_body` holds the
    arithmetic; everything above the split is index construction, now cached.
    """
    assert bridge in BRIDGE_MODES, bridge
    num_frames = prefix_states.shape[0]
    device = prefix_states.device
    idx = _gather_indices(bounds, num_frames, device)

    if not inference:
        out = _gather_body(prefix_states, suffix_states, alpha, text_state,
                           bridge == "alpha", out_dtype, **idx)
    else:
        key = (bridge, text_state is not None, out_dtype)
        if key not in _GATHER_CACHE:
            _GATHER_CACHE[key] = torch.compile(_gather_body, dynamic=False)
        out = _GATHER_CACHE[key](prefix_states, suffix_states, alpha, text_state,
                                 bridge == "alpha", out_dtype, **idx)

    return out


# Same eviction bound the BlockMask cache uses (flex_attention.MAX_CACHED_MASKS).
MAX_CACHED_MASKS = 64
_GATHER_INDEX_CACHE = collections.OrderedDict()
_GATHER_CACHE = {}


def _gather_indices(bounds, num_frames, device):
    """The index tensors `_gather_body` needs, built once per (bounds, F, device):
    rebuilding them per call would be two host-to-device copies per layer per
    denoising step for values that depend only on the geometry.
    """
    key = (tuple(bounds), num_frames, str(device))
    cached = _GATHER_INDEX_CACHE.get(key)
    if cached is not None:
        _GATHER_INDEX_CACHE.move_to_end(key)
        return cached
    last_before = torch.tensor([lo for lo, _ in bounds], device=device) - 1
    first_after = torch.tensor([hi for _, hi in bounds], device=device) + 1
    cached = dict(
        before_idx=last_before.clamp(min=0),
        after_idx=first_after.clamp(max=num_frames - 1),
        has_before=(last_before >= 0),     # frame 0 has nothing to its left
        has_after=(first_after < num_frames),   # ...and frame F-1 nothing to its right

        # The bridge index is NOT the gather index at the ends -- see the note in
        # _gather_body.
        bridge_before=(last_before + 1).clamp(min=0),
        bridge_after=first_after.clamp(max=num_frames),
        frames=torch.arange(num_frames, device=device),
    )
    _GATHER_INDEX_CACHE[key] = cached
    while len(_GATHER_INDEX_CACHE) > MAX_CACHED_MASKS:
        _GATHER_INDEX_CACHE.popitem(last=False)
    return cached


def _gather_body(prefix_states, suffix_states, alpha, text_state, bridge_alpha, out_dtype,
                 before_idx, after_idx, has_before, has_after, bridge_before,
                 bridge_after, frames):
    """The arithmetic of `gather_linear_state`, with the index tensors already built.

    Split out so inference can hand the whole thing to one compiled kernel: eager it is
    two gathers, two wheres, two multiplies and a combine over a 350 MiB fp32 bank --
    seven passes for what is one read of each side and one store.
    """
    state_before = prefix_states[before_idx]
    state_after = suffix_states[after_idx]
    if text_state is not None:
        # The out-of-range side reads the START of the scan, not a frame state.
        text_state = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, text_state)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, text_state)
    if bridge_alpha:
        # prod_{u=a..b} alpha_u as a difference of log-prefix sums, so any (a, b) pair is
        # one subtraction rather than a product loop. The leading zero row makes the
        # prefix EXCLUSIVE, i.e. the empty product is 1. fp32 throughout (alpha is fp32).
        log_alpha = torch.log(alpha.clamp_min(1e-12))
        log_alpha_prefix = torch.cat([torch.zeros_like(log_alpha[:1]),
                                      log_alpha.cumsum(0)])              # [F+1, H, d]

        # `bridge_before` / `bridge_after` are NOT the gather indices at the ends (see
        # _gather_indices). A boundary row gathers a clamped, then discarded, state but
        # must decay the text state over the frames it really skipped: from virtual -1
        # that is [0..t] (prefix row 0), from virtual F it is [t..F-1] (prefix row F,
        # which exists -- the bank has F+1 rows). Clamping both the same way would have
        # decayed the text state over one frame too few.
        alpha_from_before = torch.exp(log_alpha_prefix[frames + 1]
                                      - log_alpha_prefix[bridge_before])    # [F,H,d]
        alpha_from_after = torch.exp(log_alpha_prefix[bridge_after]
                                     - log_alpha_prefix[frames])            # [F,H,d]
        # alpha is per KEY channel: unsqueeze(2) broadcasts it over d_v, not d_k
        state_before = state_before * alpha_from_before.unsqueeze(2)
        state_after = state_after * alpha_from_after.unsqueeze(2)

    if text_state is not None:                       # both sides always contribute now
        out = state_before + state_after
    else:
        out = (state_before * has_before.view(-1, 1, 1, 1)
               + state_after * has_after.view(-1, 1, 1, 1))
    return out if out_dtype is None else out.to(out_dtype)


