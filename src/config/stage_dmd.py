"""Stage-DMD: DMD2 (no GAN) for the few-step Turbo LoRA on a frozen Stage-B VDN.

Two variants, ONE knob: `dmd.real_score` is `dense` (the released dense model as the
real score) or `vdn` (the frozen VDN itself). Everything else is shared.
"""
from dataclasses import dataclass, field

from omegaconf import MISSING

from src.config.common import DataConfig, RuntimeConfig
from src.config.stage_b import BCheckpoint, BDistributed, BInitialization

REAL_SCORES = ("dense", "vdn")
VDN_ADAPTER_NAME = "default"


@dataclass
class DInitialization(BInitialization):
    """The frozen VDN seed; its branch and Stage-B adapter are never trained here."""

    source_step: int = 2000


@dataclass
class TurboConfig:
    """The trainable few-step adapter: initialised from an external few-step LoRA,
    and the NFE grid it is sampled on (the shifts a distilled adapter is only valid
    at). `family` is a free-form provenance label stamped into the artifacts."""

    checkpoint: str = MISSING
    adapter_name: str = "turbo"
    family: str = "external_v4_step600_ema"
    num_steps: int = 8
    video_shift: float = 12.0
    audio_shift: float = 3.0


@dataclass
class DMDConfig:
    real_score: str = "dense"            # dense | vdn
    fake_adapter_name: str = "fake"
    fake_rank: int = 128
    # DMD2's loop: every sub-iteration draws a fresh prompt, rolls out, updates the fake
    # once; the generator updates on the first of every `fake_updates_per_step`
    # sub-iterations (sharing that sample). `fake_warmup_steps` fake-only sub-iterations
    # come first; a short warm-up is enough, since the fake departs from the real score
    # after its very first update.
    fake_updates_per_step: int = 3
    fake_warmup_steps: int = 10
    u_min: float = 0.02                  # DM / fake timesteps: u ~ U[u_min, u_max] -> (12u, 3u)
    u_max: float = 0.98


@dataclass
class GenerationConfig:
    """The generator's output geometry -- the eval render's, not the dataset's."""

    num_frames: int = 345
    latent_height: int = 48
    latent_width: int = 84


@dataclass
class DTraining:
    max_steps: int = 500                 # generator updates
    seed: int = 0
    audio_loss_weight: float = 0.2
    ignore_stopped: bool = False
    auto_resume: bool = True


@dataclass
class DOptimizer:
    """Two cosine schedules on one AdamW: the generator's (`turbo`, keyed on generator
    steps) and the fake score's (`fake`, keyed on fake updates)."""

    name: str = "adamw"
    lr: float = 1.0e-5
    min_lr: float = 1.0e-6
    warmup_steps: int = 100
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    fake_lr: float = 2.0e-5
    fake_min_lr: float = 2.0e-6
    fake_warmup_steps: int = 10
    fake_grad_clip: float = 1.0


@dataclass
class StageDMDConfig:
    initialization: DInitialization = field(default_factory=DInitialization)
    turbo: TurboConfig = field(default_factory=TurboConfig)
    dmd: DMDConfig = field(default_factory=DMDConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: DTraining = field(default_factory=DTraining)
    optimizer: DOptimizer = field(default_factory=DOptimizer)
    distributed: BDistributed = field(default_factory=BDistributed)
    checkpoint: BCheckpoint = field(default_factory=BCheckpoint)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def validate_stage_dmd(cfg) -> None:
    if cfg.dmd.real_score not in REAL_SCORES:
        raise ValueError(f"dmd.real_score must be one of {REAL_SCORES}, "
                         f"got {cfg.dmd.real_score!r}")
    names = (cfg.turbo.adapter_name, cfg.dmd.fake_adapter_name)
    if len(set(names)) != 2 or VDN_ADAPTER_NAME in names:
        raise ValueError(f"turbo/fake adapter names must be distinct and neither may be "
                         f"{VDN_ADAPTER_NAME!r} (the inherited Stage-B VDN adapter); got {names}")
    if cfg.turbo.num_steps < 1:
        raise ValueError("turbo.num_steps must be >= 1")
    if cfg.turbo.video_shift <= 0 or cfg.turbo.audio_shift <= 0:
        raise ValueError("turbo shifts must be positive")
    if cfg.dmd.fake_rank < 1:
        raise ValueError("dmd.fake_rank must be >= 1")
    if cfg.dmd.fake_updates_per_step < 1:
        raise ValueError("dmd.fake_updates_per_step must be >= 1")
    if cfg.dmd.fake_warmup_steps < 0:
        raise ValueError("dmd.fake_warmup_steps must be >= 0")
    if not (0.0 <= cfg.dmd.u_min < cfg.dmd.u_max <= 1.0):
        raise ValueError(f"need 0 <= dmd.u_min < dmd.u_max <= 1, got "
                         f"[{cfg.dmd.u_min}, {cfg.dmd.u_max}]")
    if cfg.training.audio_loss_weight < 0:
        raise ValueError("training.audio_loss_weight must be >= 0")
    if cfg.generation.num_frames < 1:
        raise ValueError("generation.num_frames must be >= 1")
    if cfg.generation.latent_height % 2 or cfg.generation.latent_width % 2:
        raise ValueError("generation latent height/width must be even (2x2 spatial patches)")
    if cfg.optimizer.warmup_steps < 1 or cfg.optimizer.fake_warmup_steps < 1:
        raise ValueError("both lr warmups must be >= 1 step")
