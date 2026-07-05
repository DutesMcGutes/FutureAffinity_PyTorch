from __future__ import annotations

import torch
from torch import nn

from futureaffinity.config import FutureAffinityConfig

_DIFF_EMBED_DIM = 32


class DDGHead(nn.Module):
    """Predicts DeltaDeltaG (mutant - wildtype) from the wildtype trunk representation.

    Re-running the full trunk on every candidate mutant sequence is
    expensive, so this head approximates the mutation effect cheaply: it
    embeds wildtype and mutant identity at every token, zeroes the
    difference at unchanged positions, and predicts one additive
    contribution per *changed* position from the wildtype token context --
    an explicit, documented simplification (a natural place to plug in full
    mutant re-embedding once compute allows).
    """

    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.wildtype_embedding = nn.Embedding(config.vocab_size, _DIFF_EMBED_DIM)
        self.mutant_embedding = nn.Embedding(config.vocab_size, _DIFF_EMBED_DIM)
        self.mlp = nn.Sequential(
            nn.LayerNorm(config.token_dim + _DIFF_EMBED_DIM),
            nn.Linear(config.token_dim + _DIFF_EMBED_DIM, config.token_dim),
            nn.GELU(),
            nn.Linear(config.token_dim, 1),
        )

    def forward(
        self,
        token_repr: torch.Tensor,
        token_type: torch.Tensor,
        mutant_token_type: torch.Tensor,
        is_ligand: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        protein_mask = token_mask & ~is_ligand
        changed = (mutant_token_type != token_type) & protein_mask

        diff_embed = self.mutant_embedding(mutant_token_type) - self.wildtype_embedding(token_type)
        diff_embed = diff_embed * changed.unsqueeze(-1).to(diff_embed.dtype)

        combined = torch.cat([token_repr, diff_embed], dim=-1)
        per_token_contribution = self.mlp(combined).squeeze(-1) * changed.to(token_repr.dtype)
        return per_token_contribution.sum(dim=-1)

    def loss(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (predicted - target) ** 2
