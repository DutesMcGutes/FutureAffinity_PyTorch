from __future__ import annotations

import torch

from futureaffinity.data.datatypes import Batch

# which Batch presence-mask field gates each task's loss
_TASK_PRESENCE_FIELD = {
    "diffusion": "has_structure",
    "confidence": "has_structure",  # needs ground-truth coords, same as the diffusion loss
    "contacts": "has_contacts",
    "affinity": "has_affinity",
    "ddg": "has_ddg",
}


def aggregate_losses(
    per_task_losses: dict[str, torch.Tensor], batch: Batch, task_weights: dict[str, float]
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine per-example, per-task losses into one scalar, masking out unlabeled rows.

    This is what lets a single batch mix structure-only, affinity-only, and
    docking-only examples: each task's contribution is only averaged over
    the examples that actually carry that label, and a task with zero
    labeled examples in the batch contributes nothing (instead of NaN).
    """
    total = torch.zeros((), device=next(iter(per_task_losses.values())).device)
    logs: dict[str, float] = {}

    for task, per_example_loss in per_task_losses.items():
        presence = getattr(batch, _TASK_PRESENCE_FIELD[task]).to(per_example_loss.dtype)
        count = presence.sum()
        if count.item() == 0:
            logs[task] = float("nan")
            continue
        masked_mean = (per_example_loss * presence).sum() / count
        weight = task_weights.get(task, 1.0)
        total = total + weight * masked_mean
        logs[task] = float(masked_mean.detach())

    return total, logs
