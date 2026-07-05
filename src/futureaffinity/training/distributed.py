"""Minimal, correct DistributedDataParallel scaffolding.

Single-process by default; when launched under `torchrun` (RANK/WORLD_SIZE/LOCAL_RANK set), it
initializes the process group and picks the right device. The subtlety this gets right: DDP only
all-reduces gradients across ranks for parameters touched by the wrapped module's `forward`. The
model's training entry point is `compute_losses`, not `forward`, so `TrainStep` below wraps the
whole loss computation into a `forward` -- wrap *that* in DDP and gradients sync correctly.
"""
from __future__ import annotations

import os

import torch
from torch import nn

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.datatypes import Batch
from futureaffinity.losses.multitask import aggregate_losses
from futureaffinity.model.model import FutureAffinityModel


class TrainStep(nn.Module):
    """Wraps model + multi-task aggregation so `forward(batch) -> (total_loss, logs)`.

    Exists so DDP has a `forward` that touches every trained parameter -- see module docstring.
    """

    def __init__(self, model: FutureAffinityModel, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config

    def forward(self, batch: Batch, generator: torch.Generator | None = None):
        losses, _ = self.model.compute_losses(batch, generator=generator)
        return aggregate_losses(losses, batch, self.config.task_weights)


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def maybe_init_distributed() -> torch.device:
    """Initialize the process group if launched distributed; return this rank's device."""
    if not is_distributed():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def cleanup_distributed() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
