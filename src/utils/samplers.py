"""Sampler wrappers."""
import torch


class SkipFirst(torch.utils.data.Sampler):
    """Wrap a sampler and skip its first `skip` indices -- a data-order-exact resume
    without loading the skipped samples. `skip` is zeroed after the resumed epoch."""

    def __init__(self, sampler):
        self.sampler, self.skip = sampler, 0

    def __iter__(self):
        it = iter(self.sampler)
        for _ in range(self.skip):
            next(it)
        yield from it

    def __len__(self):
        return len(self.sampler) - self.skip
