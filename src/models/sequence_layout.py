"""Packed-sequence geometry shared by both attention branches. window_bounds lives
with the softmax package; this module is the part every consumer of the PACKED LAYOUT
needs, model or not.
"""
import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceLayout:
    seq_len: int
    video_start: int          # first video row in the packed sequence
    num_frames: int           # F: latent frames
    tokens_per_frame: int     # S: patched spatial tokens per latent frame (t-major order)

    # Patched spatial grid, S = frame_height * frame_width. 0 means "not supplied":
    # S alone cannot be factored back into a grid (1008 is 24x42, but also 16x63), so
    # anything that needs the volume shape — ShortConv — must be given it explicitly
    # rather than guessing. Everything else only ever needed the token count.
    frame_height: int = 0
    frame_width: int = 0

    # Where the PROMPT rows live. Distinct from `global_index` on purpose: that one
    # returns text+audio (everything the softmax keeps dense), while the linear branch's
    # text state must see the text and NOT the soundtrack. 0 = not supplied.
    text_start: int = 0
    text_len: int = 0

    @property
    def video_end(self):
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def text_range(self):
        if not self.text_len:
            raise ValueError(
                "this layout carries no text rows; pass text_indices=... to "
                "layout_from_indices (the linear branch's text state needs the prompt "
                "rows, and text+audio from global_index is not the same thing)"
            )
        return self.text_start, self.text_start + self.text_len

    @property
    def frame_size(self):
        if not (self.frame_height and self.frame_width):
            raise ValueError(
                "this layout carries no spatial grid; pass frame_size=(H, W) to "
                "layout_from_indices — a 3D convolution cannot infer it from S alone"
            )
        if self.frame_height * self.frame_width != self.tokens_per_frame:
            raise ValueError(f"grid {self.frame_height}x{self.frame_width} != "
                             f"{self.tokens_per_frame} tokens/frame")
        return self.frame_height, self.frame_width

    def global_index(self, device):
        """Indices of the non-video rows (text, audio). The diffusers port packs no
        padding, so every row is content."""
        idx = torch.arange(self.seq_len, device=device)
        return torch.cat([idx[:self.video_start], idx[self.video_end:]])


def layout_from_indices(video_indices: torch.Tensor, num_latent_frames: int,
                        tokens_per_frame: int, seq_len: int,
                        frame_size=None, text_indices=None) -> SequenceLayout:
    """Derive the layout from `build_packed_sequence` outputs. The t2va layout puts the
    target video rows contiguous and t-major at the end of the sequence; verify rather
    than assume, because everything downstream silently depends on it."""
    video_start = int(video_indices[0].item())
    if int(video_indices[-1].item()) - video_start + 1 != video_indices.numel():
        raise ValueError("video rows are not contiguous in the packed sequence")
    if video_indices.numel() != num_latent_frames * tokens_per_frame:
        raise ValueError(
            f"{video_indices.numel()} video rows != {num_latent_frames} frames x "
            f"{tokens_per_frame} tokens/frame"
        )
    height, width = frame_size or (0, 0)
    if frame_size and height * width != tokens_per_frame:
        raise ValueError(f"frame_size {height}x{width} != {tokens_per_frame} tokens/frame")
    text_start, text_len = 0, 0
    if text_indices is not None and text_indices.numel():
        text_start = int(text_indices[0].item())
        if int(text_indices[-1].item()) - text_start + 1 != text_indices.numel():
            raise ValueError("text rows are not contiguous in the packed sequence")
        text_len = int(text_indices.numel())
    return SequenceLayout(seq_len=seq_len, video_start=video_start,
                        num_frames=num_latent_frames, tokens_per_frame=tokens_per_frame,
                        frame_height=height, frame_width=width,
                        text_start=text_start, text_len=text_len)

