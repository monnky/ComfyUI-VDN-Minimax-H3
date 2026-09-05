"""The lr-class assignment for the hybrid transform's parameters, derived from module
STRUCTURE instead of parameter-name substrings (a name-based table would silently
reclassify parameters after any rename).

small = fixed-fan-in / vector parameters (the KDA alpha's A_log, dt_bias and rank-d
up-projection, the output gate's up-projection, the branch norm gain, the depthwise
short-conv taps); big = width-fan-in matrices (alpha/gate down-projections, beta_proj,
to_out_linear, the softmax gate) -- the collapse-prone set stays conservative.
"""
from src.models.attention_gates import OutputGate
from src.models.linear_attention.features import LinearAttentionSepConv
from src.models.linear_attention.layers import FrameKDAAlpha
from src.models.ops.rms_norm import RMSNorm
from src.models.hybrid_transform import iter_hybrids


def lr_class_map(model) -> dict:
    """{parameter-object id: "small" | "big"} for every hybrid-transform parameter.
    Keyed on identity, not name -- names are presentation, structure is truth."""
    classes = {}
    for attn in iter_hybrids(model):
        for p in attn.parameters():
            classes[id(p)] = "big"                    # default: conservative
        lin = attn.linear_attention
        alpha: FrameKDAAlpha = lin.alpha
        for p in (alpha.A_log, alpha.dt_bias, *alpha.up.parameters()):
            classes[id(p)] = "small"
        gate: OutputGate = lin.output_gate
        for p in gate.up.parameters():
            classes[id(p)] = "small"
        norm: RMSNorm = lin.norm
        classes[id(norm.weight)] = "small"
        conv = lin.short_conv
        if isinstance(conv, LinearAttentionSepConv):
            for p in conv.parameters():
                classes[id(p)] = "small"
        for p in attn.orig.parameters():              # base weights: not trainables
            classes.pop(id(p), None)
    return classes


def branch_lr_class_of(model, name: str, param) -> str:
    """The lr class of one parameter, by identity (`name` is unused)."""
    return lr_class_map(model).get(id(param), "big")
