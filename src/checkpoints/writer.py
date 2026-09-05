"""Atomic v2 writes: validate, write .tmp, os.replace -- the same pattern the mp4
encoder uses, so a save interrupted mid-write never leaves a half-written file under
the final name."""
import os

import torch

from src.checkpoints.schema import CheckpointArtifact, validate_payload
from src.models.model_spec import ModelSpec


def save_checkpoint(artifact: CheckpointArtifact, path: str) -> str:
    payload = validate_payload(artifact.to_payload(), where=path)
    if "model_spec" in payload:                      # shards carry no spec of their own
        ModelSpec.from_dict(payload["model_spec"])   # spec must validate to be written
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path
