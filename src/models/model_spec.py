"""ModelSpec: the checkpoint-bound statement of WHAT the model is.

Three layers of configuration exist and this is the middle one:
  1. the Hugging Face base config -- what the original H3 is,
  2. THIS -- what we did to it structurally (transforms, adapters),
  3. the experiment/runtime config -- how one run trains or infers (never stored here).

Rules the validator enforces:
  - every value is RESOLVED: `linear_head_dim: null` is refused, 128 is stored;
  - runtime kernel choices (softmax_backend, inference_kernels, fp8, ...) are refused
    as transform-config keys -- they must never enter a checkpoint;
  - enum values come from the same registries the config schema admits.

`hybrid_transform_spec()` also accepts the flat, pre-v2 spelling of the transform
config and maps it through key_mapping.map_config.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from src.checkpoints.key_mapping import (ANCHOR_FRAME_MODES, DELTA_RULE_VALUES,
                                         SHORT_CONV_TARGETS, map_config)

SPEC_FORMAT_VERSION = 2
HYBRID_TRANSFORM_VERSION = 2      # 2: anchor_frames modes + short_conv targets
BRIDGE_VALUES = ("alpha", "none")
# Runtime knobs (current and retired names) that must never be mistaken for architecture.
_RUNTIME_KEYS = ("softmax_backend", "rmsnorm_backend", "fp8", "compile",
                 "inference_kernels", "optimized_paths", "w_o_far_scale", "window_decomp",
                 "warmup_steps")


def config_hash(resolved_config: Dict[str, Any]) -> str:
    """sha256 over a canonical JSON rendering -- key order cannot move the hash."""
    blob = json.dumps(resolved_config, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class BaseSpec:
    library: str
    class_name: str
    source: str
    subfolder: str
    revision: Optional[str]
    resolved_config: Dict[str, Any]
    config_hash: str = ""

    def __post_init__(self):
        want = config_hash(self.resolved_config)
        if not self.config_hash:
            self.config_hash = want
        elif self.config_hash != want:
            raise ValueError(f"base.config_hash {self.config_hash[:12]} does not match "
                             f"resolved_config ({want[:12]}) -- the spec was edited "
                             f"without rehashing, or the config drifted")


@dataclass
class TransformSpec:
    type: str
    version: int
    config: Dict[str, Any]


@dataclass
class AdapterSpec:
    type: str
    version: int
    config: Dict[str, Any]


@dataclass
class ModelSpec:
    format_version: int
    base: BaseSpec
    transforms: List[TransformSpec] = field(default_factory=list)
    adapters: List[AdapterSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ModelSpec":
        spec = ModelSpec(
            format_version=payload["format_version"],
            base=BaseSpec(**payload["base"]),
            transforms=[TransformSpec(**t) for t in payload.get("transforms", [])],
            adapters=[AdapterSpec(**a) for a in payload.get("adapters", [])],
        )
        validate_spec(spec)
        return spec


def _require_resolved(config: Dict[str, Any], where: str):
    for key, value in config.items():
        if value is None:
            raise ValueError(f"{where}.{key} is unresolved (null); a ModelSpec stores "
                             f"the value the model was actually built with")
        if isinstance(value, dict):
            _require_resolved(value, f"{where}.{key}")


def validate_spec(spec: "ModelSpec"):
    if spec.format_version != SPEC_FORMAT_VERSION:
        raise ValueError(f"spec format_version {spec.format_version} != "
                         f"{SPEC_FORMAT_VERSION}")
    for t in spec.transforms:
        _require_resolved(t.config, f"transforms[{t.type}]")
        leaked = sorted(set(_flat_keys(t.config)) & set(_RUNTIME_KEYS))
        if leaked:
            raise ValueError(f"transforms[{t.type}] carries runtime/deleted keys "
                             f"{leaked}; those never enter a ModelSpec")
        if t.type == "hybrid_attention":
            if t.version != HYBRID_TRANSFORM_VERSION:
                raise ValueError(
                    f"hybrid_attention transform version {t.version}; this code reads "
                    f"version {HYBRID_TRANSFORM_VERSION} (anchor_frames modes, short_conv "
                    f"targets).")
            _validate_hybrid(t.config)
    for a in spec.adapters:
        _require_resolved(a.config, f"adapters[{a.type}]")
    return spec


def _flat_keys(config: Dict[str, Any]):
    for key, value in config.items():
        yield key
        if isinstance(value, dict):
            yield from _flat_keys(value)


def _validate_hybrid(config: Dict[str, Any]):
    lin = config.get("linear_attention", {})
    soft = config.get("softmax_attention", {})
    if lin.get("delta_rule") not in DELTA_RULE_VALUES:
        raise ValueError(f"delta_rule {lin.get('delta_rule')!r} not in {DELTA_RULE_VALUES}")
    if lin.get("bridge") not in BRIDGE_VALUES:
        raise ValueError(f"bridge {lin.get('bridge')!r} not in {BRIDGE_VALUES}")
    short_conv = lin.get("short_conv")
    targets = short_conv.get("targets") if isinstance(short_conv, dict) else None
    if (not isinstance(targets, list) or len(set(targets)) != len(targets)
            or any(t not in SHORT_CONV_TARGETS for t in targets)):
        raise ValueError(f"short_conv must be {{'targets': <distinct subset of "
                         f"{SHORT_CONV_TARGETS}>}}, got {short_conv!r}")
    if config.get("anchor_frames") not in ANCHOR_FRAME_MODES:
        raise ValueError(f"anchor_frames {config.get('anchor_frames')!r} not in "
                         f"{ANCHOR_FRAME_MODES}")
    if not isinstance(lin.get("linear_head_dim"), int):
        raise ValueError(f"linear_head_dim must be a resolved int, got "
                         f"{lin.get('linear_head_dim')!r}")
    if not isinstance(soft.get("radius"), int):
        raise ValueError(f"softmax_attention.radius must be a resolved int, got "
                         f"{soft.get('radius')!r}")
    for key in ("enable_softmax_gate",):
        if not isinstance(config.get(key), bool):
            raise ValueError(f"{key} must be a resolved bool, got {config.get(key)!r}")


def hybrid_transform_spec(legacy_or_nested: Dict[str, Any]) -> TransformSpec:
    """A hybrid_attention TransformSpec from either spelling of the conversion kwargs.

    Accepts the FLAT pre-v2 dict (backend/far_head_dim/use_local_mass_gate/radius/
    chunk/...) or an already-nested v2 dict, and emits the nested v2 layout. Field and
    value renames go through key_mapping.map_config."""
    if "softmax_attention" in legacy_or_nested:            # already v2-nested
        nested = dict(legacy_or_nested)
    else:
        flat = map_config(dict(legacy_or_nested))
        nested = dict(
            enable_softmax_gate=flat.pop("enable_softmax_gate", True),
            anchor_frames=flat.pop("anchor_frames", "none"),
            softmax_attention=dict(radius=flat.pop("radius"), chunk=flat.pop("chunk", 0)),
            linear_attention=flat,        # delta_rule, linear_head_dim, bridge, short_conv,
        )                                 # enable_text_state, a_fp32

    return TransformSpec("hybrid_attention", HYBRID_TRANSFORM_VERSION, nested)
