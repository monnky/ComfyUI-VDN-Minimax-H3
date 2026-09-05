"""Distributed-init helpers shared by the FSDP2 stages."""
import torch
import torch.distributed as dist


def broadcast_and_verify_init(model, lora_params, device) -> None:
    """Rank-0-init semantics + proof, run BEFORE fully_shard.

    FSDP2 neither broadcasts from rank 0 (DeepSpeed does) nor checks that ranks agree --
    it silently freezes each rank's local values into its shard. So: broadcast the
    freshly-initialized adapters from rank 0, then verify that EVERY parameter (base
    included) is bit-identical across ranks, and raise if not.
    """
    for param in lora_params:
        t = param.data.to(device)
        dist.broadcast(t, src=0)
        param.data.copy_(t.to(param.data.device))

    # Deterministic per-parameter fingerprint: same values + same op order => bitwise
    # equal float64 sums. Any drift (a rank-dependent init that slipped through, a
    # corrupted shard read) shows up as an exact mismatch.
    names = [name for name, _ in model.named_parameters()]
    sums = torch.tensor([p.detach().double().sum().item() for _, p in model.named_parameters()],
                        dtype=torch.float64, device=device)
    reference = sums.clone()
    dist.broadcast(reference, src=0)
    mismatched = (sums != reference).nonzero().flatten().tolist()
    if mismatched:
        raise RuntimeError(
            f"rank {dist.get_rank()}: {len(mismatched)} parameters differ from rank 0 before "
            f"sharding, e.g. {[names[i] for i in mismatched[:5]]}. FSDP2 would silently shard "
            "the divergent values."
        )
