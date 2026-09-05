"""A2: architecture comes from the A1 checkpoint (v2 weights or train_state artifact);
this schema deliberately has NO `model` section.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from src.config.common import DataConfig, RuntimeConfig


@dataclass
class A2Initialization:
    checkpoint: str = MISSING            # v2 artifact from A1

    # Override for the BASE weights path only (e.g. a node-local copy of the
    # snapshot); the model's identity still comes from the checkpoint's ModelSpec.
    base_source: Optional[str] = None


@dataclass
class A2Optimizer:
    name: str = "adamw"
    lr: float = 1.0e-4
    min_lr: float = 5.0e-6
    warmup_steps: int = 25
    weight_decay: float = 0.0
    grad_clip: float = 1.0               # global; 0 disables
    big_lr_scale: float = 1.0
    small_lr_scale: float = 5.0
    small_eps: float = 2.0e-9


@dataclass
class A2Training:
    max_steps: int = 100
    gradient_accumulation_steps: int = 1
    seed: int = 0
    audio_loss_weight: float = 1.0
    freeze_softmax_gate: bool = False
    ignore_stopped: bool = False
    auto_resume: bool = True


@dataclass
class A2Distributed:
    strategy: str = "fsdp2"
    activation_checkpointing: bool = True
    shard_size: int = 8                  # HSDP: shard within, replicate across
    offload_activations: bool = True
    memory_fraction: float = 0.92


@dataclass
class A2Checkpoint:
    output_dir: str = MISSING
    save_every: int = 10
    keep_states: int = 3
    early_saves: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])


@dataclass
class StageA2Config:
    initialization: A2Initialization = field(default_factory=A2Initialization)
    data: DataConfig = field(default_factory=DataConfig)
    training: A2Training = field(default_factory=A2Training)
    optimizer: A2Optimizer = field(default_factory=A2Optimizer)
    distributed: A2Distributed = field(default_factory=A2Distributed)
    checkpoint: A2Checkpoint = field(default_factory=A2Checkpoint)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
