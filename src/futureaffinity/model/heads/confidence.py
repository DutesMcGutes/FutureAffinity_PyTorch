from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from futureaffinity.config import FutureAffinityConfig

NUM_LDDT_BINS = 50
NUM_PDE_BINS = 64
MAX_PDE_ERROR = 32.0  # angstrom, clamp for the predicted-distance-error target


def _discretize(values: torch.Tensor, num_bins: int, min_value: float, max_value: float) -> torch.Tensor:
    clipped = values.clamp(min_value, max_value - 1e-6)
    bucket_width = (max_value - min_value) / num_bins
    return ((clipped - min_value) / bucket_width).floor().long()


def compute_lddt(
    pred_coords: torch.Tensor,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Per-token lDDT-Ca: fraction of nearby-pair distances preserved within tolerance.

    Standard local distance difference test (the same metric AlphaFold's
    pLDDT is trained to predict), computed directly from coordinates so it
    can serve as a ground-truth training target -- no learned components.
    """
    pred_dist = torch.cdist(pred_coords, pred_coords)
    true_dist = torch.cdist(true_coords, true_coords)

    pair_mask = mask[:, :, None] & mask[:, None, :]
    num_tokens = mask.shape[1]
    not_self = ~torch.eye(num_tokens, dtype=torch.bool, device=mask.device)
    within_radius = (true_dist < radius) & pair_mask & not_self[None, :, :]

    diff = (pred_dist - true_dist).abs()
    preserved_fraction = sum((diff < threshold).float() for threshold in thresholds) / len(thresholds)

    numerator = (preserved_fraction * within_radius).sum(dim=-1)
    denominator = within_radius.sum(dim=-1).clamp(min=1)
    return (numerator / denominator) * 100.0


class ConfidenceHead(nn.Module):
    """Predicts per-token lDDT (pLDDT) and per-pair distance error (a PAE-style proxy).

    True AlphaFold PAE requires per-residue reference frames to define an
    aligned error; FutureAffinity's token representation only carries one
    coordinate per token, so this head predicts *predicted distance error*
    (|pred_dist - true_dist| between tokens) instead -- a real, well-defined
    quantity, just a simplification of full frame-aligned PAE. See
    docs/limitations.md.
    """

    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.token_head = nn.Sequential(
            nn.LayerNorm(config.token_dim), nn.Linear(config.token_dim, NUM_LDDT_BINS)
        )
        self.pair_head = nn.Sequential(nn.LayerNorm(config.pair_dim), nn.Linear(config.pair_dim, NUM_PDE_BINS))
        bin_width = 100.0 / NUM_LDDT_BINS
        self.register_buffer("lddt_bin_centers", torch.arange(NUM_LDDT_BINS) * bin_width + bin_width / 2)

    def forward(self, token_repr: torch.Tensor, pair_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        plddt_logits = self.token_head(token_repr)
        pde_logits = self.pair_head(pair_repr)
        plddt = torch.softmax(plddt_logits, dim=-1) @ self.lddt_bin_centers
        return {"plddt_logits": plddt_logits, "pde_logits": pde_logits, "plddt": plddt}

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        token_mask: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_lddt = compute_lddt(pred_coords.detach(), true_coords, token_mask)
        target_lddt_bins = _discretize(target_lddt, NUM_LDDT_BINS, 0.0, 100.0)
        lddt_loss = F.cross_entropy(
            outputs["plddt_logits"].reshape(-1, NUM_LDDT_BINS), target_lddt_bins.reshape(-1), reduction="none"
        ).reshape(target_lddt_bins.shape)
        lddt_loss = (lddt_loss * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1)

        pred_dist = torch.cdist(pred_coords.detach(), pred_coords.detach())
        true_dist = torch.cdist(true_coords, true_coords)
        target_pde = (pred_dist - true_dist).abs()
        target_pde_bins = _discretize(target_pde, NUM_PDE_BINS, 0.0, MAX_PDE_ERROR)
        pde_loss = F.cross_entropy(
            outputs["pde_logits"].reshape(-1, NUM_PDE_BINS), target_pde_bins.reshape(-1), reduction="none"
        ).reshape(target_pde_bins.shape)
        pde_loss = (pde_loss * pair_mask).sum(dim=(-1, -2)) / pair_mask.sum(dim=(-1, -2)).clamp(min=1)

        return lddt_loss + pde_loss
