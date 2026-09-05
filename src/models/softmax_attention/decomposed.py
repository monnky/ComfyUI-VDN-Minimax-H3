"""The window softmax as a union of DENSE attentions -- no BlockMask, no sparse kernel.

The c1 mask is not arbitrary sparsity: every kept pair lies in one of a few dense
rectangles. Split the QUERY rows into groups whose kept KV set is identical, and each
group is a plain dense attention:

  dense-q   global (text/audio) rows, plus anchor-frame queries under rows/both ->
            one flash_attn_func call against the FULL kv (no copy).
  windows   remaining frames grouped by identical window bounds (== chunks) -> ONE
            flash_attn_varlen_func call over per-group gathered kv
            [globals + window frames + anchor frames].

Same math as the masked kernel -- each query's softmax spans exactly its kept set in
one pass -- so outputs differ from flex by bf16 reduction order only (a few 1e-3
relative, the same distance two flex backends show against each other). On sm100 the
dense calls are markedly faster than the block-sparse flex kernel at this mask density.

Inference-only and opt-in (config kernels.softmax_backend = decomposed, or auto on sm100;
hybrid_transform.set_softmax_backend latches it before the first forward): training keeps
the flex path.
"""


import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from src.models.sequence_layout import SequenceLayout

_PLAN_CACHE = {}
MAX_CACHED_PLANS = 4
_STATE = {}
SOFTMAX_BACKENDS = ("auto", "flex", "decomposed", "ref")


def resolve_softmax_backend(backend: str) -> str:
    """Config ``kernels.softmax_backend`` -> the implementation that will run. ``auto``
    is ``decomposed`` on sm100 only: the win is Blackwell-specific (on sm90 the flex
    block-sparse path already runs close to its dense kernel, and the gather cost eats
    the rest), so one YAML serves both architectures and sm90 inference stays on flex.
    ``flex`` / ``decomposed`` force either kernel; ``ref`` is the eager reference for
    parity/debug."""
    if backend not in SOFTMAX_BACKENDS:
        raise ValueError(f"kernels.softmax_backend={backend!r}; expected one of "
                         f"{SOFTMAX_BACKENDS}")
    if backend != "auto":
        return backend
    on_sm100 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10
    return "decomposed" if on_sm100 else "flex"


def set_decomposition(enabled: bool) -> None:
    """Process-level latch read by the flex path (hybrid_attention, ulysses): the
    inference entrypoints set it through hybrid_transform.set_softmax_backend; training
    never does, so the default state is OFF."""
    _STATE["enabled"] = bool(enabled)


def decomposition_enabled():
    return bool(_STATE.get("enabled")) and not _STATE.get("disabled")


def decomposition_state() -> dict:
    """What actually ran, for the render record: the resolved switch and the
    failure latch (mark_decomposition_broken)."""
    return {"enabled": bool(_STATE.get("enabled")),
            "latched_off": bool(_STATE.get("disabled"))}


class _Plan:
    __slots__ = ("dense_q", "win_q", "kv_gather", "cu_q", "cu_k", "max_q", "max_k",
                 "has_windows", "window_groups")

    def __init__(self, layout: SequenceLayout, bounds, anchor_frames, device):
        S = layout.seq_len
        F, TPF = layout.num_frames, layout.tokens_per_frame
        vs, ve = layout.video_start, layout.video_end
        anchor_set = {0, F - 1} if anchor_frames in ("columns", "rows", "both") else set()
        dense_row_frames = anchor_set if anchor_frames in ("rows", "both") else set()
        dense_col_frames = anchor_set if anchor_frames in ("columns", "both") else set()

        def frame_rows(f):
            return (vs + f * TPF, vs + (f + 1) * TPF)

        global_ranges = [r for r in ((0, vs), (ve, S)) if r[0] < r[1]]

        def merge(ranges):
            out = []
            for a, b in sorted(ranges):
                if out and out[-1][1] >= a:
                    out[-1] = (out[-1][0], max(out[-1][1], b))
                else:
                    out.append((a, b))
            return out

        def cat_ranges(ranges):
            return torch.cat([torch.arange(a, b, device=device) for a, b in ranges])

        # dense-q rows: globals + anchor-row frames
        dense_ranges = merge(global_ranges + [frame_rows(f) for f in sorted(dense_row_frames)])
        self.dense_q = cat_ranges(dense_ranges) if dense_ranges else torch.empty(
            0, dtype=torch.long, device=device)

        # window groups: consecutive frames sharing identical bounds (== chunks)
        groups = []
        for f in range(F):
            if f in dense_row_frames:
                continue
            if groups and bounds[groups[-1][-1]] == bounds[f] and groups[-1][-1] == f - 1:
                groups[-1].append(f)
            else:
                groups.append([f])

        q_idx, kv_idx, q_lens, k_lens = [], [], [], []
        for frames in groups:
            lo, hi = bounds[frames[0]]
            kv_frames = sorted(set(range(max(lo, 0), min(hi + 1, F))) | dense_col_frames)
            q_r = merge([frame_rows(f) for f in frames])
            kv_r = merge(global_ranges + [frame_rows(f) for f in kv_frames])
            qi, ki = cat_ranges(q_r), cat_ranges(kv_r)
            q_idx.append(qi); kv_idx.append(ki)
            q_lens.append(len(qi)); k_lens.append(len(ki))

        self.has_windows = bool(groups)
        self.window_groups = list(zip(q_idx, kv_idx)) if groups else []
        if self.has_windows:
            self.win_q = torch.cat(q_idx)
            self.kv_gather = torch.cat(kv_idx)
            zero = torch.zeros(1, dtype=torch.long)
            self.cu_q = torch.cat([zero, torch.tensor(q_lens).cumsum(0)]).to(
                device, torch.int32)
            self.cu_k = torch.cat([zero, torch.tensor(k_lens).cumsum(0)]).to(
                device, torch.int32)
            self.max_q, self.max_k = max(q_lens), max(k_lens)
        else:
            self.win_q = torch.empty(0, dtype=torch.long, device=device)

        order = torch.cat([self.dense_q, self.win_q])
        if len(order) != S:
            raise ValueError(f"decomposition covers {len(order)} of {S} rows")


def _plan(layout, bounds, anchor_frames, device):
    key = (layout.seq_len, layout.video_start, layout.num_frames,
           layout.tokens_per_frame, tuple(bounds), anchor_frames, str(device))
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = _Plan(layout, bounds, anchor_frames, device)
        while len(_PLAN_CACHE) > MAX_CACHED_PLANS:
            _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    return _PLAN_CACHE[key]


def window_softmax_decomposed(query, key, value, layout, bounds, scale,
                              anchor_frames="none"):
    """[T, H, d] q/k/v -> [T, H, d], exactly the window+anchors+globals mask, as
    dense calls scatter-written straight into a contiguous output."""
    plan = _plan(layout, bounds, anchor_frames, query.device)

    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)

    if len(plan.dense_q):
        qd = query[plan.dense_q]
        od = scaled_dot_product_attention(
            qd.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0), scale=scale)
        out[plan.dense_q] = od[0].transpose(0, 1)

    if plan.has_windows:
        used_varlen = False
        try:
            from flash_attn import flash_attn_varlen_func
            kw = key[plan.kv_gather]
            vw = value[plan.kv_gather]
            ow = flash_attn_varlen_func(
                query[plan.win_q], kw, vw,
                cu_seqlens_q=plan.cu_q, cu_seqlens_k=plan.cu_k,
                max_seqlen_q=plan.max_q, max_seqlen_k=plan.max_k, softmax_scale=scale)
            ow = ow[0] if isinstance(ow, tuple) else ow
            out[plan.win_q] = ow
            used_varlen = True
        except Exception:
            pass

        if not used_varlen:
            # High-speed native PyTorch SDPA path (zero compile delay on Windows / consumer GPUs)
            for qi, ki in plan.window_groups:
                qc = query[qi].transpose(0, 1).unsqueeze(0)
                kc = key[ki].transpose(0, 1).unsqueeze(0)
                vc = value[ki].transpose(0, 1).unsqueeze(0)
                oc = scaled_dot_product_attention(qc, kc, vc, scale=scale)
                out[qi] = oc[0].transpose(0, 1)

    return out


def mark_decomposition_broken(reason):
    """Latch the decomposition off for the process, loudly, mirroring the FLASH
    fallback: a slower window beats a dead render."""
    _STATE["disabled"] = True
    print("hybrid_attention: window decomposition failed; falling back to flex for "
          f"the rest of the process. Reason: {reason}", flush=True)
