from __future__ import annotations

import torch


def structural_uncertainty(ensemble_coords: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Per-token positional uncertainty (an RMSF-style spread) across a structural ensemble.

    `ensemble_coords`: (B, S, N, 3), the S independent samples produced by
    `DiffusionModule.sample_ensemble`. This is the same computation used to
    turn an MD trajectory into per-residue fluctuation estimates -- applied
    here to a diffusion ensemble instead of a physical trajectory (idea:
    self-distillation from ensembles gives uncertainty "for free").
    """
    mean_coords = ensemble_coords.mean(dim=1, keepdim=True)
    squared_deviation = ((ensemble_coords - mean_coords) ** 2).sum(dim=-1)  # (B, S, N)
    mean_squared_deviation = squared_deviation.mean(dim=1)  # (B, N)
    rmsf = mean_squared_deviation.clamp(min=0).sqrt()
    return rmsf * token_mask.to(rmsf.dtype)


def affinity_uncertainty(ensemble_affinity_predictions: torch.Tensor) -> torch.Tensor:
    """Std-dev of an affinity head's prediction across ensemble members. `(B, S) -> (B,)`."""
    return ensemble_affinity_predictions.std(dim=1, unbiased=False)
