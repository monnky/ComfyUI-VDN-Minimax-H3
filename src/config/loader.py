"""One loader for every entrypoint.

    cfg = load_config(StageBConfig, argv)

Control surface is exactly three flags -- --config, --print-config, --validate-only --
plus dotlist overrides. Everything else that used to be a CLI flag is a YAML field.

Precedence: structured defaults < YAML < CLI dotlist. OmegaConf's struct mode makes an
unknown key in either layer a hard error; to_object() makes a still-MISSING mandatory
field a hard error naming its full path. Both fire before any model or dataset is
touched.
"""
import argparse
import sys
from typing import Any, Callable, Optional, Sequence

from omegaconf import MISSING, OmegaConf
from omegaconf.errors import MissingMandatoryValue, OmegaConfBaseException


class ConfigError(SystemExit):
    """Load-time config failure. SystemExit so an entrypoint dies with the message
    and a nonzero code without a traceback wall -- the traceback never says anything
    the message does not."""

    def __init__(self, message: str):
        super().__init__(f"config error: {message}")


def load_config(schema_cls, argv: Optional[Sequence[str]] = None,
                extra_validators: Sequence[Callable[[Any], None]] = ()):
    """Parse argv against `schema_cls`; return the merged, fully-validated config
    (an OmegaConf struct). Honors --print-config / --validate-only by exiting."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--config", action="append", default=[],
                        help="YAML file(s), merged in order")
    parser.add_argument("--print-config", action="store_true",
                        help="print the resolved config and exit")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate and exit 0 (or die with the reason)")
    args, dotlist = parser.parse_known_args(argv)

    bad = [d for d in dotlist if d.startswith("--")]
    if bad:
        raise ConfigError(f"unknown flag(s) {bad}; only --config / --print-config / "
                          f"--validate-only exist -- everything else is a YAML field "
                          f"or a dotlist override like optimizer.lr=4e-5")

    cfg = OmegaConf.structured(schema_cls)
    try:
        for path in args.config:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
        if dotlist:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(dotlist)))
    except OmegaConfBaseException as exc:
        # first line carries the complaint; ValidationError appends full_key lines --
        # keep those, they name the exact field
        detail = [l.strip() for l in str(exc).splitlines() if l.strip()]
        keep = [l for l in detail if l.startswith("full_key")]
        raise ConfigError("; ".join(detail[:1] + keep)) from exc

    try:
        OmegaConf.to_object(cfg)          # MISSING check; keep the DictConfig though
    except MissingMandatoryValue as exc:
        detail = [l.strip() for l in str(exc).splitlines() if l.strip()]
        keep = [l for l in detail if l.startswith("full_key")]
        raise ConfigError("; ".join(detail[:1] + keep)) from exc

    for validate in extra_validators:
        try:
            validate(cfg)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc

    if args.print_config:
        print(OmegaConf.to_yaml(cfg), end="")
        sys.exit(0)
    if args.validate_only:
        print("config ok", file=sys.stderr)
        sys.exit(0)
    return cfg


def resolved_dict(cfg) -> dict:
    """Plain-dict snapshot for stamping into train state (resolved_training_config)."""
    return OmegaConf.to_container(cfg, resolve=True)
