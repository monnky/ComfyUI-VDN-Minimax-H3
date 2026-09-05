"""Lazy exports (PEP 562). Eager ones would cycle: model_spec imports
key_mapping (a leaf in this package), and loader/writer import model_spec back."""
import importlib

_EXPORTS = {
    "load_checkpoint": "src.checkpoints.loader",
    "checkpoint_head_sha256": "src.checkpoints.loader",
    "export_checkpoint_dir": "src.checkpoints.export_dir",
    "save_checkpoint": "src.checkpoints.writer",
    "CheckpointArtifact": "src.checkpoints.schema",
    "UnsupportedCheckpointFormat": "src.checkpoints.schema",
}


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
