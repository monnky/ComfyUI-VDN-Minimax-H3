"""Pre-v2 -> v2 renames for checkpoint keys and transform-config fields: the single
mapping table, plus the value registries the config schema and the ModelSpec share.

Rule order is load-bearing: `.attn.far.gate.` must fire before the generic
`.attn.far.`, or the gate lands at linear_attention.gate instead of
linear_attention.output_gate. First match wins, exactly one substitution per key.

Keys that match NO rule pass through unchanged. That is not a fallback, it is the
contract: LoRA adapter keys (`.attn.orig.*.lora_A/B.*` on the DiT blocks AND
`token_refiner.refiner_blocks.*.attn.*.lora_A/B.*` -- the refiner is a LoRA target
too) and the meta stamps are already spelled the way v2 wants them.
"""

# (old_substring, new_substring), most specific first.
KEY_RULES = (
    (".attn.far.gate.", ".attn.linear_attention.output_gate."),
    (".attn.far.", ".attn.linear_attention."),
    (".attn.local_gate.", ".attn.softmax_gate."),
    (".attn.w_o_far.", ".attn.to_out_linear."),
)

# Pre-v2 state-dict meta stamps. They are consumed into the ModelSpec on conversion
# and never appear in a v2 weights dict.
META_KEYS = ("__window__", "__branch__", "__truncated__")

# Conversion-kwarg renames, pre-v2 flat config -> v2 transform config.
# None = deleted (w_o_far_scale was init-only; the weights already carry the fact).
CONFIG_FIELD_RENAMES = {
    "backend": "delta_rule",
    "far_head_dim": "linear_head_dim",
    "use_local_mass_gate": "enable_softmax_gate",
    "text_state": "enable_text_state",
    "w_o_far_scale": None,
}

DELTA_RULE_VALUES = ("sana_scaled", "vdn_solve", "vdn_scaled")

# Pre-v2 short_conv was (False, True, "sep") -- a bool/str union. v2 spells the short
# conv by the projections it convolves: False is no targets, "sep" is the separable
# K/V conv. True was a dense 5^3 depthwise conv that no longer exists -- refused.
SHORT_CONV_TARGETS = ("q", "k", "v")
SHORT_CONV_VALUES = {False: [], "sep": ["k", "v"]}

# The anchor frames (0 and F-1) can be dense as softmax COLUMNS (everyone sees them),
# as ROWS (they see everyone), or both; pre-v2 True is "both". Only "both" makes the
# softmax/linear partition exact, so only "both" lets the linear branch skip them.
ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")


def map_key(name: str) -> str:
    """v2 spelling of one pre-v2 state-dict / optimizer-state key."""
    if name in META_KEYS:
        return name
    for old, new in KEY_RULES:
        if old in name:
            return name.replace(old, new, 1)
    return name


def map_keys(names):
    """Map a whole key set; refuse collisions rather than silently merging tensors."""
    out = {name: map_key(name) for name in names}
    seen = {}
    for old, new in out.items():
        if new in seen:
            raise ValueError(f"key collision: {old!r} and {seen[new]!r} both map to {new!r}")
        seen[new] = old
    return out


def map_config(config: dict) -> dict:
    """v2 spelling of a pre-v2 flat transform-config dict."""
    out = {}
    for key, value in config.items():
        new_key = CONFIG_FIELD_RENAMES.get(key, key)
        if new_key is None:
            continue
        out[new_key] = value
    if "short_conv" in out:
        value = out["short_conv"]
        if isinstance(value, dict):
            targets = list(value["targets"])
        elif isinstance(value, (list, tuple)):
            targets = list(value)
        elif value is True:
            raise ValueError("short_conv=True is the dense 5^3 depthwise Conv3d, which "
                             "this code no longer implements")
        else:
            targets = SHORT_CONV_VALUES[value]
        out["short_conv"] = {"targets": targets}
    if "anchor_frames" in out:
        mode = out["anchor_frames"]
        if isinstance(mode, bool):
            mode = "both" if mode else "none"
        if mode not in ANCHOR_FRAME_MODES:
            raise ValueError(f"unknown anchor_frames {mode!r}; expected one of "
                             f"{ANCHOR_FRAME_MODES}")
        out["anchor_frames"] = mode
    rule = out.get("delta_rule")
    if rule is not None and rule not in DELTA_RULE_VALUES:
        raise ValueError(f"unknown delta_rule {rule!r}; expected one of {DELTA_RULE_VALUES}")
    return out
