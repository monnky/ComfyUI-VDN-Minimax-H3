"""A1: the ONE stage that creates the hybrid architecture from YAML.
Every later stage inherits the architecture from a checkpoint, so every later schema
simply lacks the `model.architecture` section -- an override attempt is an unknown
field, refused by the loader before any model exists.

The architecture block shares its layout with the ModelSpec. A1's lr is one linear
ramp over the whole run (warmup_steps == max_steps by default): annealing is A2's job.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING

from src.config.common import (CheckpointConfig, DataConfig, DistributedConfig,
                               ModelConfig, RuntimeConfig)


@dataclass
class A1Optimizer:
    """muP split, matching Stage A2: width-fan-in matrices take the conservative lr,
    fixed-fan-in / vector params the larger one (assigned STRUCTURALLY -- see
    utils.lr_classes.lr_class_map)."""
    name: str = "adamw"
    lr: float = 1.0e-4
    min_lr: float = 5.0e-6
    warmup_steps: int = 50            # == max_steps: the ramp IS the schedule
    weight_decay: float = 0.0
    grad_clip: float = 0.1            # applied PER LAYER in A1
    big_lr_scale: float = 1.0
    small_lr_scale: float = 5.0
    small_eps: float = 2.0e-9


@dataclass
class A1Training:
    max_steps: int = 50
    gradient_accumulation_steps: int = 1
    seed: int = 0
    freeze_softmax_gate: bool = False   # hold the softmax gate at its 0.99 init
    per_layer_every: int = 25
    truncate_blocks: int = 0            # smoke tests only; checkpoints stamped unusable
    teacher_backend: str = "flex_fa4"   # the TEACHER trunk's kernel; runtime, not spec


@dataclass
class A1Checkpoint(CheckpointConfig):
    output_dir: str = MISSING
    save_every: int = 25
    early_saves: List[int] = field(default_factory=lambda: [0, 1, 2, 4, 8])


@dataclass
class StageA1Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: A1Training = field(default_factory=A1Training)
    optimizer: A1Optimizer = field(default_factory=A1Optimizer)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: A1Checkpoint = field(default_factory=A1Checkpoint)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
