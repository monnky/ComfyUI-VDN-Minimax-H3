"""HybridAttention: the orchestrator and nothing else.

Drop-in replacement for the diffusers MiniMaxH3Attention inside a DiT block. It owns
the shared QKV, calls the softmax window and the linear branch, and fuses:

    softmax_output = orig.to_out( softmax_gate(x) * window_softmax(q, k, v) )
    linear_readout = output_gate(x) * RMSNorm( linear_attention(q_raw, k_raw, v_raw) )
    output         = softmax_output + to_out_linear(linear_readout)   # video rows only

to_out_linear takes nn.Linear's default init. Init-only: any checkpoint overwrites this
weight on load.
"""
import torch
from torch import nn
import torch.nn.functional as F

try:
    from diffusers.models.attention_dispatch import dispatch_attention_fn
except (ImportError, ModuleNotFoundError):
    import torch.nn.functional as F
    def dispatch_attention_fn(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None):
        return F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

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


from src.models.attention_gates import OutputGate
from src.models.sequence_layout import SequenceLayout  # noqa: F401  (re-export: layout consumers)
from src.models.linear_attention import BidirectionalLinearBranch
from src.models.softmax_attention import (apply_softmax_gate, build_window_block_mask,
                                          window_bounds, window_softmax_flex,
                                          window_softmax_reference)
from src.models.softmax_attention.kernels import _qk_prep
from src.models.ops.fp8_linear import Fp8Linear, quantize_activation
from src.checkpoints.key_mapping import ANCHOR_FRAME_MODES


class HybridAttention(nn.Module):
    """Drop-in replacement for the diffusers `MiniMaxH3Attention` inside a DiT block:
    same forward signature `(hidden_states, rotary_emb, attention_mask)`, same output
    contract (pre-residual attention output, batch-first). Batch must be 1 — H3 packs
    one request per document. Set `.layout` (SequenceLayout) before each forward and
    `.radius` to choose the frame window; the operating range is r <= 5.
    `radius >= num_frames-1` (or layout=None) makes the softmax branch equal to full
    attention.

    `.teacher_mode = True` makes forward a pure pass-through to the original attention —
    Stage A runs the trunk that way (teacher trajectory, no_grad) and computes the
    student output in a forward hook, keeping alignment a training strategy rather than
    a property of the module."""

    _current_latent_shape = None

    def __init__(self, orig_attn, hidden_size, delta_rule="sana_scaled", radius=4,
                 chunk=0, enable_softmax_gate=True, linear_head_dim=None,
                 softmax_impl="flex", anchor_frames="none", enable_text_state=False,
                 bridge="alpha", a_fp32=True, short_conv=()):
        """Keyword names mirror the ModelSpec transform config one-to-one
        (`softmax_attention.{radius,chunk}`, `anchor_frames`,
        `linear_attention.{delta_rule,linear_head_dim,bridge,a_fp32,enable_text_state}`,
        `linear_attention.short_conv.targets` as `short_conv`, `enable_softmax_gate`);
        `softmax_impl` is the one runtime knob ("flex" | "decomposed" | "ref", set by
        hybrid_transform.set_softmax_backend) and never enters a spec.

        anchor_frames: how frames 0 and F-1 sit in the softmax mask -- "columns" (every
        video query sees all of both frames), "rows" (those two frames' queries see the
        whole sequence), "both", or "none". A cross-branch fact: under "both" the two
        frames are exact softmax in both directions, so the linear branch drops them from
        its input (skip_ends) and the softmax/linear partition stays exact. Under
        "columns" or "rows" alone the partition would not be exact, so the branch keeps
        covering them."""
        super().__init__()
        # Store orig_attn without registering it as a submodule, avoiding PyTorch adding the 'orig.' prefix
        object.__setattr__(self, "orig", orig_attn)

        self.num_heads = getattr(orig_attn, "heads", getattr(orig_attn, "num_heads", 24))
        self.head_dim = getattr(orig_attn, "head_dim", 128)
        self.radius = radius
        self.chunk = chunk           # 0 = frame window ("r<n>"); K = K-frame chunks ("c<n>")
        self.softmax_impl = softmax_impl                         # "flex" | "decomposed" | "ref"

        if anchor_frames not in ANCHOR_FRAME_MODES:
            raise ValueError(f"anchor_frames={anchor_frames!r}; expected one of "
                             f"{ANCHOR_FRAME_MODES}")
        self.anchor_frames = anchor_frames

        # Seed both linear-branch scans with the prompt (see BidirectionalLinearBranch.
        # forward). Needs a layout that carries the text rows — layout_from_indices
        # must have been given text_indices.
        self.enable_text_state = enable_text_state
        self.linear_attention_enabled = True    # False = window-only ablation (pure sparse attention)
        self.layout: SequenceLayout = None
        self.teacher_mode = False

        # Two inference-only levels, separated so a benchmark can attribute the speedup.
        # `hybrid_inference_mode` covers only work intrinsic to the hybrid algorithm:
        # the FLASH window kernel and the tuned linear far branch. `inference_mode`
        # additionally enables general fusions such as QK-norm + RoPE. The production
        # setter turns both on; defaults stay slow-but-correct for training and forgotten
        # switches.
        self.hybrid_inference_mode = False
        self.inference_mode = False
        d_linear = linear_head_dim or self.head_dim
        if isinstance(short_conv, dict):
            short_conv = short_conv.get("targets", ())
        self.linear_attention = BidirectionalLinearBranch(
            hidden_size, self.num_heads, d_linear, delta_rule=delta_rule, bridge=bridge,
            a_fp32=a_fp32, short_conv=short_conv)
        # The linear branch's own output projection, torch-default init: the readout it
        # consumes is SiLU'd and RMS-normalised, so there is no reason to seed it from
        # orig.to_out. Stage A1 trains it from scratch.
        self.to_out_linear = nn.Linear(self.num_heads * d_linear, hidden_size,
                                       bias=False)
        self.enable_softmax_gate = enable_softmax_gate
        if enable_softmax_gate:
            # per-head, direct (not low rank). 0.99 keeps the softmax branch at the
            # teacher on step 0.
            self.softmax_gate = OutputGate(hidden_size, self.num_heads, init_value=0.99)

    @property
    def heads(self):
        return self.num_heads

    def load_hybrid_weights(self, state_dict):
        """Loads linear branch and gating weights directly into this HybridAttention module."""
        clean = {k: v for k, v in state_dict.items() if not k.startswith("orig.")}
        return self.load_state_dict(clean, strict=False)

    def _qkv(self, x, rotary_emb=None, rope_freqs=None):
        """Replicates MiniMaxH3AttnProcessor up to (and excluding) the attention call,
        via the original module's submodules — projections (LoRA-wrapped ones apply),
        QK-norm and RoPE included. Also returns the raw (pre-QK-norm, pre-RoPE) q/k/v
        for the shared-QKV linear branch. x: [total, hidden]; everything [total, H, d].
        Under fp8 the three projections share one quantisation of x.

        Inference runs QK-norm + rope as ONE kernel (`_qk_prep`, not bitwise); training
        keeps the eager ops the checkpoints were trained under."""
        qkv_module = getattr(self.orig, "qkv_proj", None)
        q_norm_module = getattr(self.orig, "q_norm", None)
        k_norm_module = getattr(self.orig, "k_norm", None)

        # Handle ComfyUI's Attention class (single qkv_proj, q_norm, k_norm)
        if qkv_module is not None:
            s = x.shape[0]
            inner = self.num_heads * self.head_dim
            qkv = qkv_module(x)
            # High-speed low-rank Turbo LoRA (consumes only ~110 MB VRAM, 24x fewer FLOPs)
            turbo_lora_qkv = getattr(self, "turbo_lora_qkv", None)
            if turbo_lora_qkv is not None:
                la_cat, lqb, lkb, lvb, lscale = turbo_lora_qkv
                if la_cat.device != x.device or la_cat.dtype != x.dtype:
                    la_cat = la_cat.to(device=x.device, dtype=x.dtype)
                    lqb = lqb.to(device=x.device, dtype=x.dtype)
                    lkb = lkb.to(device=x.device, dtype=x.dtype)
                    lvb = lvb.to(device=x.device, dtype=x.dtype)
                    self.turbo_lora_qkv = (la_cat, lqb, lkb, lvb, lscale)
                mid = F.linear(x, la_cat)
                mid_q, mid_k, mid_v = mid.chunk(3, dim=-1)
                dq = F.linear(mid_q, lqb)
                dk = F.linear(mid_k, lkb)
                dv = F.linear(mid_v, lvb)
                delta_qkv = torch.cat([dq, dk, dv], dim=-1) * lscale
                qkv = qkv + delta_qkv
            elif getattr(self, "turbo_delta_qkv", None) is not None:
                td = self.turbo_delta_qkv
                if td.device != x.device or td.dtype != x.dtype:
                    self.turbo_delta_qkv = td.to(device=x.device, dtype=x.dtype)
                qkv = qkv + torch.matmul(x, self.turbo_delta_qkv.t())
            q, k, v = qkv.split(inner, dim=-1)
            query_raw = q.view(s, self.num_heads, self.head_dim)
            key_raw = k.view(s, self.num_heads, self.head_dim)
            value = v.view(s, self.num_heads, self.head_dim)

            if rope_freqs is not None:
                q_view = query_raw.view(1, s, self.num_heads, self.head_dim).clone()
                k_view = key_raw.view(1, s, self.num_heads, self.head_dim).clone()
                try:
                    import comfy.model_management
                    import comfy.quant_ops.ck
                    qw = comfy.model_management.cast_to(q_norm_module.weight, device=x.device) if hasattr(q_norm_module, "weight") else None
                    kw = comfy.model_management.cast_to(k_norm_module.weight, device=x.device) if hasattr(k_norm_module, "weight") else None
                    rot = rope_freqs.shape[-3] * 2 if (hasattr(rope_freqs, "shape") and len(rope_freqs.shape) >= 3) else 96
                    comfy.quant_ops.ck.rms_rope_split_half_(q_view, k_view, rope_freqs, qw, kw, epsilon=q_norm_module.eps, rot_dim=rot)
                    query = q_view[0]
                    key = k_view[0]
                except Exception:
                    query = q_norm_module(query_raw)
                    key = k_norm_module(key_raw)
            elif rotary_emb is not None:
                query = q_norm_module(query_raw)
                key = k_norm_module(key_raw)
                query = _apply_rotary_emb(query.unsqueeze(0), *rotary_emb).squeeze(0)
                key = _apply_rotary_emb(key.unsqueeze(0), *rotary_emb).squeeze(0)
            else:
                query = q_norm_module(query_raw)
                key = k_norm_module(key_raw)

            return query, key, value, (query_raw, key_raw, value)

        # Handle Diffusers model structure (to_q, to_k, to_v)
        projections = (orig.to_q, orig.to_k, orig.to_v)

        if all(isinstance(p, Fp8Linear) for p in projections):
            x_fp8, x_scale = quantize_activation(x)
            qkv = [p.forward_quantized(x_fp8, x_scale, out_dtype=x.dtype) for p in projections]
        else:
            qkv = [p(x) for p in projections]

        query_raw, key_raw, value = (t.unflatten(-1, (orig.heads, -1)) for t in qkv)  # [total, H, d]

        if self.inference_mode and rotary_emb is not None:
            query = _qk_prep(query_raw, orig.norm_q.weight, orig.norm_q.eps, *rotary_emb)
            key = _qk_prep(key_raw, orig.norm_k.weight, orig.norm_k.eps, *rotary_emb)
        else:
            query, key = orig.norm_q(query_raw), orig.norm_k(key_raw)
            if rotary_emb is not None:
                query = _apply_rotary_emb(query.unsqueeze(0), *rotary_emb).squeeze(0)
                key = _apply_rotary_emb(key.unsqueeze(0), *rotary_emb).squeeze(0)

        return query, key, value, (query_raw, key_raw, value)

    def _bounds(self, layout):
        return window_bounds(layout.num_frames, self.radius, self.chunk)

    def forward(self, hidden_states, rotary_emb=None, attention_mask=None, rope_freqs=None, transformer_options={}, **kwargs):
        is_2d = (hidden_states.dim() == 2)
        x = hidden_states if is_2d else hidden_states[0]

        if self.teacher_mode:
            if hasattr(self.orig, "qkv_proj"):
                return self.orig(hidden_states, rope_freqs=rope_freqs, transformer_options=transformer_options)
            else:
                return self.orig(hidden_states, rotary_emb, attention_mask)

        out = self._hybrid_forward(x, rotary_emb=rotary_emb, rope_freqs=rope_freqs, transformer_options=transformer_options)
        return out if is_2d else out.unsqueeze(0)

    def _hybrid_forward(self, x, rotary_emb=None, rope_freqs=None, transformer_options={}):
        total_tokens = x.shape[0]
        cur_layout = getattr(HybridAttention, "_current_layout", None)
        if cur_layout is not None and cur_layout.seq_len == total_tokens:
            layout = cur_layout
            self.layout = layout
        else:
            root_latent_shape = getattr(HybridAttention, "_current_latent_shape", None)

            if root_latent_shape is not None:
                num_frames, raw_h, raw_w = root_latent_shape
                frame_h = max(1, raw_h // 2)
                frame_w = max(1, raw_w // 2)
                tokens_per_frame = frame_h * frame_w
                video_len = num_frames * tokens_per_frame
                video_start = max(0, total_tokens - video_len)
                video_end = total_tokens
                text_start, text_end = 0, video_start
            elif orig_shape is not None and len(orig_shape) >= 5:
                num_frames = int(orig_shape[2])
                frame_h = max(1, int(orig_shape[3]) // 2)
                frame_w = max(1, int(orig_shape[4]) // 2)
                tokens_per_frame = frame_h * frame_w
                video_len = num_frames * tokens_per_frame
                video_start = max(0, total_tokens - video_len)
                video_end = total_tokens
                text_start, text_end = 0, video_start
            else:
                frame_h, frame_w = 19, 33
                tokens_per_frame = frame_h * frame_w
                num_frames = max(1, total_tokens // tokens_per_frame)
                video_len = num_frames * tokens_per_frame
                video_start = max(0, total_tokens - video_len)
                video_end = total_tokens
                text_start, text_end = 0, video_start

            layout = self.layout
            if (layout is None
                or layout.seq_len != total_tokens
                or layout.tokens_per_frame != tokens_per_frame
                or layout.num_frames != num_frames
                or getattr(layout, "frame_height", None) != frame_h
                or getattr(layout, "frame_width", None) != frame_w):
                layout = SequenceLayout(
                    seq_len=total_tokens,
                    video_start=video_start,
                    num_frames=num_frames,
                    tokens_per_frame=tokens_per_frame,
                    frame_height=frame_h,
                    frame_width=frame_w,
                    text_start=text_start,
                    text_len=max(0, text_end - text_start),
                )
                self.layout = layout
            if not getattr(HybridAttention, "_logged_layout", False):
                HybridAttention._logged_layout = True
                print(f"[ComfyUI-RT-VDN-MMH3] SequenceLayout: total={total_tokens} | Video: {layout.num_frames} frames @ {layout.frame_height}x{layout.frame_width} ({layout.tokens_per_frame} tpf), range [{layout.video_start}:{layout.video_end}] | Text: [{layout.text_start}:{layout.text_start+layout.text_len}] (Backend: {self.softmax_impl})")

        hybrid_inference = getattr(self, "hybrid_inference_mode", False) or getattr(self, "inference_mode", False)
        bounds = self._bounds(layout) if layout is not None else None
        full_cover = layout is None or all(
            lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
        scale = self.head_dim ** -0.5

        query, key, value, qkv_raw = self._qkv(x, rotary_emb=rotary_emb, rope_freqs=rope_freqs)
        use_flex = (not full_cover) and self.softmax_impl in ("flex", "decomposed") and x.is_cuda

        if full_cover:
            # A window wide enough to cover every frame IS the original attention, so go
            # through the stock processor's own dispatch rather than a bare SDPA call:
            backend_opt = getattr(type(getattr(self.orig, "processor", None)), "_attention_backend", None)
            softmax_out = dispatch_attention_fn(
                query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0),
                attn_mask=None, dropout_p=0.0, is_causal=False,
                backend=backend_opt,
            ).squeeze(0)
            linear_active = False
        elif use_flex:
            softmax_out = None
            if self.softmax_impl == "decomposed":
                from src.models.softmax_attention.decomposed import window_softmax_decomposed
                try:
                    softmax_out = window_softmax_decomposed(
                        query, key, value, layout, bounds, scale,
                        anchor_frames=self.anchor_frames)
                except Exception as exc:
                    print(f"[VDN-Minimax] Notice: decomposed SDPA fallback error: {exc}. Using flex.")
            if softmax_out is None:
                block_mask = build_window_block_mask(layout, bounds, value.device,
                                                     anchor_frames=self.anchor_frames)
                softmax_out = window_softmax_flex(query, key, value, block_mask, scale,
                                                   inference=hybrid_inference)
            linear_active = True
        else:
            softmax_out = window_softmax_reference(query, key, value, layout, bounds, scale,
                                              anchor_frames=self.anchor_frames)
            linear_active = True

        # Drop roped query/key once local attention completes
        del query, key, value
        full_coverage = (layout.num_frames <= (2 * self.radius + 1) * self.chunk)
        linear_active = not full_coverage

        if self.enable_softmax_gate and not full_coverage:
            if hasattr(self, "softmax_gate") and self.softmax_gate is not None:
                if self.softmax_gate.up.weight.dtype != x.dtype or self.softmax_gate.up.weight.device != x.device:
                    self.softmax_gate.to(dtype=x.dtype, device=x.device, non_blocking=True)
            flat = apply_softmax_gate(softmax_out, self.softmax_gate(x),
                                     inference=self.inference_mode)
            # Offload softmax_gate immediately after use
            if hasattr(self, "softmax_gate") and self.softmax_gate is not None:
                self.softmax_gate.to("cpu", non_blocking=True)
        else:
            flat = softmax_out.reshape(x.shape[0], -1)

        out_module = getattr(self.orig, "out_proj", None)
        flat_cast = flat.type_as(x)
        if out_module is not None:
            out = out_module(flat_cast)
        elif hasattr(self.orig, "to_out"):
            out = self.orig.to_out[0](flat_cast)
            if len(self.orig.to_out) > 1:
                out = self.orig.to_out[1](out)
        else:
            out = flat_cast

        # Low-rank Turbo LoRA for Out projection
        turbo_lora_out = getattr(self, "turbo_lora_out", None)
        if turbo_lora_out is not None:
            loa, lob, loscale = turbo_lora_out
            if loa.device != flat_cast.device or loa.dtype != flat_cast.dtype:
                loa = loa.to(device=flat_cast.device, dtype=flat_cast.dtype)
                lob = lob.to(device=flat_cast.device, dtype=flat_cast.dtype)
                self.turbo_lora_out = (loa, lob, loscale)
            delta_out = F.linear(F.linear(flat_cast, loa), lob) * loscale
            out = out + delta_out
        elif getattr(self, "turbo_delta_out", None) is not None:
            td_out = self.turbo_delta_out
            if td_out.device != flat_cast.device or td_out.dtype != flat_cast.dtype:
                self.turbo_delta_out = td_out.to(device=flat_cast.device, dtype=flat_cast.dtype)
            out = out + torch.matmul(flat_cast, self.turbo_delta_out.t())

        del softmax_out

        if linear_active and self.linear_attention_enabled:
            hybrid_inference = getattr(self, "hybrid_inference_mode", False) or getattr(self, "inference_mode", False)
            # Just-In-Time load linear branch to GPU right before use
            if hasattr(self, "linear_attention") and self.linear_attention is not None:
                p = next(self.linear_attention.parameters(), None)
                if p is not None and (p.dtype != x.dtype or p.device != x.device):
                    self.linear_attention.to(dtype=x.dtype, device=x.device, non_blocking=True)
            if hasattr(self, "to_out_linear") and self.to_out_linear is not None:
                if self.to_out_linear.weight.dtype != x.dtype or self.to_out_linear.weight.device != x.device:
                    self.to_out_linear.to(dtype=x.dtype, device=x.device, non_blocking=True)

            video_start, video_end = layout.video_start, layout.video_end
            video_x = x[video_start:video_end]
            video_qkv_raw = tuple(t[video_start:video_end] for t in qkv_raw)
            text_x = text_qkv_raw = None
            if self.enable_text_state:
                text_start, text_end = layout.text_range
                text_x = x[text_start:text_end]
                text_qkv_raw = tuple(t[text_start:text_end] for t in qkv_raw)

            # The grid is only DEMANDED when the conv exists: layout.frame_size raises
            # on grid-less layouts, and every conv-less config must keep working with
            # them (only frame_size consumers are gated on short_conv).
            linear_readout = self.linear_attention(video_x, layout.num_frames, layout.tokens_per_frame,
                                   bounds, qkv_raw=video_qkv_raw,
                                   frame_size=(layout.frame_size
                                               if self.linear_attention.short_conv is not None
                                               else None),
                                   skip_ends=self.anchor_frames == "both",
                                   text_x=text_x, text_qkv_raw=text_qkv_raw,
                                   inference=hybrid_inference)

            # The clone is autograd's: `out` is the output of self.orig.to_out[1], and
            # writing into it in place would corrupt what backward needs. With no graph
            # to protect there is nothing to copy -- and at H3 scale this tensor is
            # [~105k, 5376] bf16 = 1.05 GiB, copied once per layer per denoising step.
            if torch.is_grad_enabled():
                out = out.clone()
            out[video_start:video_end] += self.to_out_linear(linear_readout.type_as(x))

            # OFFLOAD IMMEDIATELY right after its use is over
            if hasattr(self, "linear_attention") and self.linear_attention is not None:
                self.linear_attention.to("cpu", non_blocking=True)
            if hasattr(self, "to_out_linear") and self.to_out_linear is not None:
                self.to_out_linear.to("cpu", non_blocking=True)
            del linear_readout

        return out
