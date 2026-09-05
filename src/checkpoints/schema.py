"""Checkpoint v2: what an artifact file contains, and nothing else.

Two artifact kinds cover everything the trainers write:

  "weights"      the small per-save file (hybrid_step*.pt / hybrid_lora_step*.pt):
                 model_spec + weights. What eval and inference consume.
  "train_state"  the resumable file (train_state*.pt): everything above plus
                 optimizer / step / rng_state / resolved_training_config / metadata.

Optimizer state is a {param_name: state_dict} mapping, which is what makes offline key
renames possible at all. An index-keyed optimizer state is refused at write time.

`scheduler` stays absent by design: the lr is a pure function of step (lr_schedule.py),
so (step, resolved_training_config) IS the scheduler state.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

CHECKPOINT_FORMAT_VERSION = 2
ARTIFACT_KINDS = ("weights", "train_state", "optimizer_shard")

_REQUIRED = {
    "weights": ("checkpoint_format_version", "kind", "model_spec", "weights"),
    "train_state": ("checkpoint_format_version", "kind", "model_spec", "weights",
                    "step", "rng_state", "resolved_training_config"),

    # ZeRO-1 stages (A1) shard the moments by owner rank and write one file per rank
    # next to the train_state; the main artifact's metadata names the shards. Kept a
    # separate KIND so a shard can never be mistaken for a resumable state.
    "optimizer_shard": ("checkpoint_format_version", "kind", "optimizer"),
}


class UnsupportedCheckpointFormat(RuntimeError):
    pass


@dataclass
class CheckpointArtifact:
    kind: str
    model_spec: Dict[str, Any]                      # ModelSpec.to_dict()
    weights: Dict[str, Any]                         # name -> tensor
    optimizer: Optional[Dict[str, Any]] = None      # name -> AdamW state (train_state)
    step: Optional[int] = None
    rng_state: Optional[Dict[str, Any]] = None
    resolved_training_config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                   "kind": self.kind, "model_spec": self.model_spec,
                   "weights": self.weights, "metadata": self.metadata}
        if self.kind == "train_state":
            payload.update(optimizer=self.optimizer, step=self.step,
                           rng_state=self.rng_state,
                           resolved_training_config=self.resolved_training_config)
        elif self.kind == "optimizer_shard":
            payload = {"checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                       "kind": self.kind, "optimizer": self.optimizer,
                       "metadata": self.metadata}
        return payload


def validate_payload(payload: Dict[str, Any], where: str = "checkpoint"):
    version = payload.get("checkpoint_format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise UnsupportedCheckpointFormat(
            f"{where}: unsupported checkpoint format "
            f"{version if version is not None else 'pre-v2'}; this code reads "
            f"format {CHECKPOINT_FORMAT_VERSION} only"
        )
    kind = payload.get("kind")
    if kind not in ARTIFACT_KINDS:
        raise UnsupportedCheckpointFormat(f"{where}: unknown artifact kind {kind!r}")
    missing = [k for k in _REQUIRED[kind] if payload.get(k) is None]
    if missing:
        raise UnsupportedCheckpointFormat(f"{where}: {kind} artifact is missing {missing}")
    legacy_meta = [k for k in payload.get("weights") or {} if k.startswith("__")]
    if legacy_meta:
        raise UnsupportedCheckpointFormat(
            f"{where}: weights carry pre-v2 meta stamps {legacy_meta}; v2 stores that "
            f"information in the model_spec")
    if kind == "train_state" and payload.get("optimizer") is None \
            and not payload.get("metadata", {}).get("optimizer_shards"):
        raise UnsupportedCheckpointFormat(
            f"{where}: train_state carries neither an inline optimizer nor "
            f"metadata.optimizer_shards")
    opt = payload.get("optimizer")
    if opt is not None:
        bad = [k for k in opt if not isinstance(k, str)]
        if bad:
            raise UnsupportedCheckpointFormat(
                f"{where}: optimizer state must be keyed by parameter NAME "
                f"(got e.g. {bad[0]!r}); index-keyed state cannot survive renames")
    return payload
