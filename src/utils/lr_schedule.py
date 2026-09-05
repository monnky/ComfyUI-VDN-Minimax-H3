"""The lr schedule, in one dependency-free place.

Every trainer computes lr as a pure function of `step` -- there is no scheduler object --
and the resume check needs the same function. Dependency-free on purpose, so a report
tool does not need torch to know the shape of a cosine.
"""
import math
from types import SimpleNamespace

SCHEDULE_ARGS = ("lr", "min_lr", "warmup_steps", "max_steps")


def lr_at(step: int, args) -> float:
    if step < args.warmup_steps:
        lr = args.lr * (step + 1) / args.warmup_steps
    else:
        progress = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))

    return lr


def schedule_sum(ran_steps, lr, min_lr, warmup_steps, max_steps):
    """Total lr actually spent over `ran_steps`: sum of lr_at, not peak x steps.

    This is the correct denominator for a utilisation: AdamW's per-element step is ~the
    CURRENT lr, so the reachable displacement over a run is the sum of the schedule, and a
    run that spends half its steps below peak can never reach peak*steps.
    """
    a = SimpleNamespace(lr=lr, min_lr=min_lr, warmup_steps=warmup_steps,
                        max_steps=max_steps)
    return sum(lr_at(s, a) for s in range(ran_steps))


def schedule_stamp(args):
    return {k: getattr(args, k) for k in SCHEDULE_ARGS if hasattr(args, k)}


def check_schedule_stamp(saved, args, rank=0):
    """Warn, loudly, when a resume changes the shape of the lr curve.

    Warn rather than fail on purpose: runs may be auto-requeued, a hard stop costs more
    than a changed schedule -- and deliberately extending max_steps is a legitimate
    thing to do. It just has to be visible.
    """
    if not saved or rank != 0:
        return
    now = schedule_stamp(args)
    diff = {k: (saved.get(k), now.get(k)) for k in now if saved.get(k) != now.get(k)}
    if not diff:
        return
    print("=" * 78, flush=True)
    print("WARNING: lr schedule args CHANGED across this resume:", flush=True)
    for k, (was, is_) in diff.items():
        print(f"    {k}: {was} -> {is_}", flush=True)
    try:
        old = SimpleNamespace(**{**now, **saved})
        step = getattr(args, "_resume_step", 0)
        a, b = lr_at(step, old), lr_at(step, args)
        print(f"    lr at the resume step ({step}): {a:.3e} -> {b:.3e}  "
              f"({b / max(a, 1e-30):.2f}x)", flush=True)
    except Exception:
        pass
    print("=" * 78, flush=True)
