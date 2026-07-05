from __future__ import annotations

import torch
from torch import nn

from futureaffinity.config import FutureAffinityConfig


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(x.dtype)
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


class AffinityHead(nn.Module):
    """E(protein, ligand, conformation): a learned energy/affinity head.

    Deliberately *not* a function of sequence identity alone -- it pools the
    interface region of the pair representation (which encodes the trunk's
    structural hypothesis) and a simple physical feature (nearest
    protein-ligand distance) computed from the actual coordinates passed in.
    Call it once per sampled conformation (e.g. once per member of a
    diffusion ensemble) and every call is one observation of the same
    underlying energy landscape, per the "learn E(...), not Kd directly"
    framing in the project brief.
    """

    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        interface_dim = config.pair_dim
        token_dim = config.token_dim
        hidden_dim = config.token_dim
        self.interface_proj = nn.Sequential(nn.LayerNorm(interface_dim), nn.Linear(interface_dim, hidden_dim))
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3 + 1),
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim

    def forward(
        self,
        token_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        coords: torch.Tensor,
        is_ligand: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        protein_mask = token_mask & ~is_ligand
        ligand_mask = token_mask & is_ligand

        cross_pair_mask = protein_mask[:, :, None] & ligand_mask[:, None, :]
        interface_pair = self.interface_proj(pair_repr)
        interface_pooled = (interface_pair * cross_pair_mask.unsqueeze(-1)).sum(dim=(1, 2))
        interface_pooled = interface_pooled / cross_pair_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1.0).squeeze(-1)

        protein_pooled = _masked_mean(token_repr, protein_mask)
        ligand_pooled = _masked_mean(token_repr, ligand_mask)

        has_both = (protein_mask.any(dim=1)) & (ligand_mask.any(dim=1))
        cross_dist = torch.cdist(coords, coords)
        cross_dist = cross_dist.masked_fill(~cross_pair_mask, float("inf"))
        min_dist = cross_dist.amin(dim=(1, 2))
        min_dist = torch.where(has_both, min_dist, torch.full_like(min_dist, 10.0))
        min_dist = min_dist.clamp(max=1e4).nan_to_num(nan=10.0, posinf=10.0)

        features = torch.cat(
            [interface_pooled, protein_pooled, ligand_pooled, min_dist[:, None]], dim=-1
        )
        return self.mlp(features).squeeze(-1)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (predicted - target) ** 2
