"""B: architecture from the A1/A2 checkpoint; LoRA is the ONLY structural addition
and its spec is written into the ModelSpec on first save.

B fresh: `lora` section required (rank/alpha/targets).
B resume: architecture AND adapters come from the B checkpoint; the trainer must
refuse a YAML `lora` section that disagrees with the checkpoint's adapter spec --
schema-level we cannot know which mode this is, so the field stays optional here
and the trainer owns that check.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from src.config.common import DataConfig, RuntimeConfig

# The DiT blocks' original attention projections AND the two token-refiner blocks.
DEFAULT_LORA_TARGETS = (
    "attn.orig.to_q", "attn.orig.to_k", "attn.orig.to_v", "attn.orig.to_out.0",
    "token_refiner.refiner_blocks.*.attn.to_q",
    "token_refiner.refiner_blocks.*.attn.to_k",
    "token_refiner.refiner_blocks.*.attn.to_v",
    "token_refiner.refiner_blocks.*.attn.to_out.0",
)


@dataclass
class LoraConfig:
    rank: int = 64
    alpha: int = 64
    targets: List[str] = field(default_factory=lambda: list(DEFAULT_LORA_TARGETS))


@dataclass
class BInitialization:
    checkpoint: str = MISSING
    # Override for the BASE weights path only; identity stays with the spec.
    base_source: Optional[str] = None


@dataclass
class BOptimizer:
    name: str = "adamw"
    lr: float = 1.0e-4
    min_lr: float = 5.0e-6
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    branch_lr_scale: float = 0.4
    small_lr_scale: float = 2.0
    small_eps: float = 2.0e-9


@dataclass
class BTraining:
    max_steps: int = 500
    gradient_accumulation_steps: int = 1
    seed: int = 0
    objective: str = "teacher"
    lambda_diffusion_video: float = 0.0
    lambda_diffusion_audio: float = 0.0
    audio_loss_weight: float = 1.0
    ignore_stopped: bool = False
    auto_resume: bool = True


@dataclass
class BDistributed:
    strategy: str = "fsdp2"
    activation_checkpointing: bool = True
    shard_size: int = 8
    offload_activations: bool = True
    memory_fraction: float = 0.92


@dataclass
class BCheckpoint:
    output_dir: str = MISSING
    save_every: int = 10
    keep_states: int = 3
    early_saves: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])


@dataclass
class StageBConfig:
    initialization: BInitialization = field(default_factory=BInitialization)
    lora: Optional[LoraConfig] = None
    data: DataConfig = field(default_factory=DataConfig)
    training: BTraining = field(default_factory=BTraining)
    optimizer: BOptimizer = field(default_factory=BOptimizer)
    distributed: BDistributed = field(default_factory=BDistributed)
    checkpoint: BCheckpoint = field(default_factory=BCheckpoint)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
