"""The window geometry and the reference softmax.

window_bounds says WHICH frames each query frame may attend to; the reference
implementation is the correctness oracle every kernel test compares against.
"""
import torch
import torch.nn.functional as F

from src.models.sequence_layout import SequenceLayout


def window_bounds(num_frames, radius, chunk=0):
    """Per-frame inclusive softmax-window bounds [lo, hi], unclamped.

    chunk == 0  FRAME mode: the centered window |t_q - t_k| <= radius.

    chunk == K  CHUNK-ALIGNED mode ("c<radius>"): frame t belongs to chunk t // K and
        sees whole chunks [c - radius, c + radius]. The window is a property of the
        CHUNK, not of the frame — frames 5 and 9 (both chunk 1, K=5) get the identical
        window — which is the point: the VAE encodes every K latent frames
        independently, so a frame that sees only part of a neighbouring chunk sees a
        fragment of a unit that was never coded as separable. c1 therefore guarantees
        every frame a complete previous, current and next chunk.

        A frame window cannot express this. Centering radius r on t gives frame 5 the
        span [5-r, 5+r] and frame 9 the span [9-r, 9+r]; whatever r is, one of them
        straddles a chunk boundary. Alignment is what is being bought here, not width.

        The last chunk is short when K does not divide num_frames (102 = 20*5 + 2) and
        that is fine — it is still a whole chunk, just a smaller one.
    """
    if chunk <= 0:
        bounds = [(t - radius, t + radius) for t in range(num_frames)]
    else:
        bounds = [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1)
                  for t in range(num_frames)]

    return bounds



def window_softmax_reference(query, key, value, layout: SequenceLayout, bounds, scale,
                        anchor_frames="none"):
    """Reference (loop) window softmax. The shipped path is window_softmax_flex; this
    exists to check it, and to run on CPU where Triton is unavailable.

    query/key/value: [total, H, d], already QK-normed and RoPE'd. Exact softmax in which
    (video, video) pairs are restricted to the per-frame window, while every pair
    involving a global (text/audio) token stays dense in both directions.

    One SDPA call per query frame, so it is O(F) launches — correct and easy to read,
    but only meant for CPU tests and for checking the block-sparse kernel against."""
    heads, head_dim = query.shape[1], query.shape[2]
    video_start, video_end = layout.video_start, layout.video_end
    num_frames, tokens_per_frame = layout.num_frames, layout.tokens_per_frame
    global_idx = layout.global_index(query.device)
    out = torch.empty_like(query)

    def sdpa(q_rows, k_rows, v_rows):
        """[rows, H, d] -> [rows, H, d] via one dense attention over the given rows."""
        attended = F.scaled_dot_product_attention(
            q_rows.permute(1, 0, 2).unsqueeze(0), k_rows.permute(1, 0, 2).unsqueeze(0),
            v_rows.permute(1, 0, 2).unsqueeze(0), scale=scale)
        return attended.squeeze(0).permute(1, 0, 2)

    # global queries attend to the whole sequence (exact)
    if global_idx.numel():
        out[global_idx] = sdpa(query[global_idx], key, value)

    # video queries: their window's frames, plus all global keys
    global_key, global_value = key[global_idx], value[global_idx]
    frame_shape = (num_frames, tokens_per_frame, heads, head_dim)
    video_query = query[video_start:video_end].view(frame_shape)
    video_key = key[video_start:video_end].view(frame_shape)
    video_value = value[video_start:video_end].view(frame_shape)
    for frame in range(num_frames):
        lo = max(bounds[frame][0], 0)
        hi = min(bounds[frame][1], num_frames - 1)
        if anchor_frames in ("rows", "both") and frame in (0, num_frames - 1):
            lo, hi = 0, num_frames - 1           # anchor ROWS are dense (see the mask)

        # anchor COLUMNS: frames 0 and F-1 for every query, without duplicating keys the
        # window already covers (a duplicated key would double its softmax mass)
        extra = [f for f in ((0, num_frames - 1) if anchor_frames in ("columns", "both") else ())
                 if not lo <= f <= hi]
        extra_key = [video_key[f].reshape(-1, heads, head_dim) for f in extra]
        extra_value = [video_value[f].reshape(-1, heads, head_dim) for f in extra]
        window_key = torch.cat([global_key,
                                video_key[lo:hi + 1].reshape(-1, heads, head_dim)]
                               + extra_key)
        window_value = torch.cat([global_value,
                                  video_value[lo:hi + 1].reshape(-1, heads, head_dim)]
                                 + extra_value)
        row_start = video_start + frame * tokens_per_frame
        out[row_start: row_start + tokens_per_frame] = sdpa(
            video_query[frame], window_key, window_value)
    return out

