from __future__ import annotations

import torch


def masked_mae(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.to(predicted.dtype)
    denom = mask.sum().clamp(min=1.0)
    return float(((predicted - target).abs() * mask).sum() / denom)


def contact_accuracy(logits: torch.Tensor, target: torch.Tensor, pair_mask: torch.Tensor, threshold: float = 0.5) -> float:
    predicted = (torch.sigmoid(logits) > threshold).to(target.dtype)
    correct = (predicted == target).to(torch.float32)
    mask = pair_mask.to(torch.float32)
    return float((correct * mask).sum() / mask.sum().clamp(min=1.0))
