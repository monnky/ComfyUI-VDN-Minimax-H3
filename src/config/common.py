"""Shared structured-config sections.

Dataclasses, consumed through OmegaConf.structured(): unknown YAML/CLI fields error,
type mismatches error, and MISSING fields error at to_object() time with the field's
full path. Fields that a stage must supply carry MISSING rather than a default, so
"forgot to set it" is a load-time failure, never a silently-defaulted run.

The split between sections follows the three-layer rule:
`model.*` is architecture (destined for the ModelSpec, checkpoint-bound), everything
else is how THIS run trains or infers, and `runtime.*` is kernel/backend choice that
must never enter a checkpoint.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import MISSING

# The registries the schema admits (mirrored in src/checkpoints/key_mapping.py).
DELTA_RULES = ("sana_scaled", "vdn_solve", "vdn_scaled")
BRIDGES = ("alpha", "none")
SHORT_CONV_TARGETS = ("q", "k", "v")
ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")


@dataclass
class BaseModelConfig:
    """Which Hugging Face model we start from. `resolved_config`/hash live in the
    ModelSpec, not here -- this is the pointer, the spec is the receipt."""
    library: str = "diffusers"
    class_name: str = "MiniMaxH3Transformer3DModel"
    source: str = MISSING                    # snapshot dir or hub id
    subfolder: str = "transformer"
    revision: Optional[str] = None
    local_files_only: bool = True
    dtype: str = "bfloat16"
    config_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SoftmaxAttentionConfig:
    """The window every video query's softmax is restricted to. chunk=0: a FRAME window,
    |t_q - t_k| <= radius (centred, "r<radius>"). chunk=K: a CHUNK-ALIGNED window, the
    query's K-frame chunk plus `radius` whole chunks either side ("c<radius>")."""
    radius: int = MISSING
    chunk: int = 0


@dataclass
class ShortConvConfig:
    """Depthwise 5x5-spatial x 5-tap-temporal conv on the linear branch's projections
    named here (subset of q/k/v; empty = no conv)."""
    targets: List[str] = field(default_factory=lambda: ["k", "v"])


@dataclass
class LinearAttentionConfig:
    delta_rule: str = MISSING                # one of DELTA_RULES
    linear_head_dim: Optional[int] = None    # None = inherit softmax head_dim; the
                                             # RESOLVED value is what enters ModelSpec
    bridge: str = "alpha"
    short_conv: ShortConvConfig = field(default_factory=ShortConvConfig)
    enable_text_state: bool = False
    a_fp32: bool = True


@dataclass
class HybridAttentionConfig:
    enable_softmax_gate: bool = True

    # Frames 0 and F-1 in the softmax mask: "columns" (every video query sees all of
    # both), "rows" (their queries see everything), "both", "none". Sits at this level
    # because it is a cross-branch fact: only "both" makes the two frames exact softmax,
    # and only then does the linear branch drop them (skip_ends).
    anchor_frames: str = "none"
    softmax_attention: SoftmaxAttentionConfig = field(default_factory=SoftmaxAttentionConfig)
    linear_attention: LinearAttentionConfig = field(default_factory=LinearAttentionConfig)


@dataclass
class ArchitectureConfig:
    hybrid_attention: HybridAttentionConfig = field(default_factory=HybridAttentionConfig)


@dataclass
class ModelConfig:
    base: BaseModelConfig = field(default_factory=BaseModelConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)


@dataclass
class DataConfig:
    # Index of your pre-encoded latents dataset; see "Training data" in the README for
    # the on-disk layout it points into.
    index_file: str = MISSING
    num_workers: int = 4


@dataclass
class InitializationConfig:
    checkpoint: Optional[str] = None         # stage schemas override the requiredness


@dataclass
class OptimizerConfig:
    """One cosine schedule for everyone: lr_at(step) needs (lr, min_lr, warmup_steps)
    here plus training.max_steps. Three parameter groups (lora x1, branch_big x
    branch_lr_scale, branch_small x small_lr_scale), assigned from module STRUCTURE
    (src/utils/lr_classes.py), never from name substrings."""
    name: str = "adamw"
    lr: float = 1.0e-4
    min_lr: float = 5.0e-6
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    branch_lr_scale: float = 0.4
    small_lr_scale: float = 2.0
    small_eps: float = 1.0e-10


@dataclass
class TrainingConfig:
    max_steps: int = MISSING
    gradient_accumulation_steps: int = 1
    seed: int = 0


@dataclass
class DistributedConfig:
    strategy: str = "fsdp"
    offload_activations: bool = False


@dataclass
class CheckpointConfig:
    output_dir: str = MISSING
    save_every: int = 50


@dataclass
class KernelsConfig:
    softmax_backend: str = "flex"            # flex | decomposed | ref | auto (inference adds
                                             # auto = per arch); runtime: NEVER enters ModelSpec


@dataclass
class RuntimeConfig:
    kernels: KernelsConfig = field(default_factory=KernelsConfig)


def validate_enums(cfg) -> None:
    """The registry checks OmegaConf's type system cannot express."""
    arch = getattr(getattr(cfg, "model", None), "architecture", None)
    if arch is None:
        return
    ha = arch.hybrid_attention
    if ha.linear_attention.delta_rule not in DELTA_RULES:
        raise ValueError(f"delta_rule {ha.linear_attention.delta_rule!r} "
                         f"not in {DELTA_RULES}")
    if ha.linear_attention.bridge not in BRIDGES:
        raise ValueError(f"bridge {ha.linear_attention.bridge!r} not in {BRIDGES}")
    targets = list(ha.linear_attention.short_conv.targets)
    if any(t not in SHORT_CONV_TARGETS for t in targets) or len(set(targets)) != len(targets):
        raise ValueError(f"short_conv.targets {targets!r} must be a distinct subset of "
                         f"{SHORT_CONV_TARGETS}")
    if ha.anchor_frames not in ANCHOR_FRAME_MODES:
        raise ValueError(f"anchor_frames {ha.anchor_frames!r} not in {ANCHOR_FRAME_MODES}")
