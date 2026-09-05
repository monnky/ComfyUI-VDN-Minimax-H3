"""The one reader. Checkpoint v2 only; anything else is refused, not shimmed. A path
may be the single .pt or the exploded directory (src/checkpoints/export_dir.py);
callers get the same CheckpointArtifact."""
import torch

from src.checkpoints.export_dir import is_checkpoint_dir, load_checkpoint_dir

from src.checkpoints.schema import (CheckpointArtifact, UnsupportedCheckpointFormat,
                                    validate_payload)
from src.models.model_spec import ModelSpec


def checkpoint_head_sha256(path: str) -> str:
    """Identity stamp for a render record: sha256 of the first MiB of the .pt; for the
    exploded directory, of model_spec.json, metadata.json and the first MiB of DATA
    (past the header) of every safetensors file, so two exports with the same tensor
    names but different weights stamp differently."""
    import hashlib
    import os
    import struct

    digest = hashlib.sha256()
    if not is_checkpoint_dir(path):
        with open(path, "rb") as f:
            digest.update(f.read(1 << 20))
        return digest.hexdigest()

    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            full = os.path.join(root, name)
            with open(full, "rb") as f:
                if name.endswith(".safetensors"):
                    header_len = struct.unpack("<Q", f.read(8))[0]
                    f.seek(8 + header_len)
                digest.update(name.encode())
                digest.update(f.read(1 << 20))
    return digest.hexdigest()


def load_checkpoint(path: str, mmap: bool = True) -> CheckpointArtifact:
    if is_checkpoint_dir(path):
        return load_checkpoint_dir(path)

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    except Exception as exc:
        raise UnsupportedCheckpointFormat(f"{path}: not a readable checkpoint "
                                          f"({type(exc).__name__}: {exc})") from exc
    if not isinstance(payload, dict):
        raise UnsupportedCheckpointFormat(f"{path}: expected a dict payload, got "
                                          f"{type(payload).__name__}")
    validate_payload(payload, where=path)
    if payload.get("model_spec") is not None:
        ModelSpec.from_dict(payload["model_spec"])
    return CheckpointArtifact(
        kind=payload["kind"], model_spec=payload.get("model_spec"),
        weights=payload.get("weights"), optimizer=payload.get("optimizer"),
        step=payload.get("step"), rng_state=payload.get("rng_state"),
        resolved_training_config=payload.get("resolved_training_config"),
        metadata=payload.get("metadata", {}),
    )
