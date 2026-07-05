from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from futureaffinity.config import FutureAffinityConfig


class ContactHead(nn.Module):
    """Predicts a symmetric per-pair contact/interface probability from the pair representation."""

    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(config.pair_dim), nn.Linear(config.pair_dim, 1))

    def forward(self, pair_repr: torch.Tensor) -> torch.Tensor:
        logits = self.proj(pair_repr).squeeze(-1)
        return (logits + logits.transpose(-1, -2)) / 2.0

    def loss(self, logits: torch.Tensor, target_contacts: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        per_pair = F.binary_cross_entropy_with_logits(logits, target_contacts, reduction="none")
        mask = pair_mask.to(per_pair.dtype)
        return (per_pair * mask).sum(dim=(-1, -2)) / mask.sum(dim=(-1, -2)).clamp(min=1)
