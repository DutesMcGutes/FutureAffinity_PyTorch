from __future__ import annotations

import torch
from torch import nn

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.datatypes import Batch

_MAX_RELATIVE_DISTANCE = 32


class RelativePositionEncoding(nn.Module):
    """Bucketed relative-position bias for the pair representation.

    Tokens in the same chain get a bucket for `clip(residue_index[i] -
    residue_index[j], -32, 32)`; tokens in different chains (e.g. any
    protein-ligand pair, or two different protein chains) all share one
    extra "different chain" bucket, since relative sequence position is
    meaningless across chains.
    """

    def __init__(self, pair_dim: int, max_relative_distance: int = _MAX_RELATIVE_DISTANCE) -> None:
        super().__init__()
        self.max_relative_distance = max_relative_distance
        num_buckets = 2 * max_relative_distance + 2
        self.embedding = nn.Embedding(num_buckets, pair_dim)

    def forward(self, residue_index: torch.Tensor, chain_id: torch.Tensor) -> torch.Tensor:
        relative = residue_index[:, :, None] - residue_index[:, None, :]
        relative = relative.clamp(-self.max_relative_distance, self.max_relative_distance)
        relative = relative + self.max_relative_distance

        same_chain = chain_id[:, :, None] == chain_id[:, None, :]
        different_chain_bucket = 2 * self.max_relative_distance + 1
        bucket = torch.where(same_chain, relative, torch.full_like(relative, different_chain_bucket))
        return self.embedding(bucket)


class InputEmbedder(nn.Module):
    """Builds initial token and pair representations from a Batch.

    Token representation: vocab lookup (residues + ligand heavy atoms share
    one table, see FutureAffinityConfig) + a ligand/polymer flag + an optional
    projected protein-language-model embedding (masked to zero wherever
    `has_esm` is False, so ESM-less batches are unaffected).

    Pair representation: outer sum of two linear projections of the token
    representation (the standard AlphaFold-style pair init) plus the
    relative-position bias above.
    """

    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.config = config
        self.token_type_embedding = nn.Embedding(config.vocab_size, config.token_dim)
        self.is_ligand_embedding = nn.Embedding(2, config.token_dim)
        self.esm_proj = (
            nn.Linear(config.esm_embedding_dim, config.token_dim) if config.use_esm_embeddings else None
        )
        self.left_proj = nn.Linear(config.token_dim, config.pair_dim)
        self.right_proj = nn.Linear(config.token_dim, config.pair_dim)
        self.relative_position = RelativePositionEncoding(config.pair_dim)

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.token_type_embedding(batch.token_type) + self.is_ligand_embedding(batch.is_ligand.long())

        if self.esm_proj is not None:
            esm_contribution = self.esm_proj(batch.esm_embedding)
            esm_contribution = esm_contribution * batch.has_esm[:, None, None].to(esm_contribution.dtype)
            token = token + esm_contribution

        token = token * batch.token_mask[:, :, None].to(token.dtype)

        pair = self.left_proj(token)[:, :, None, :] + self.right_proj(token)[:, None, :, :]
        pair = pair + self.relative_position(batch.residue_index, batch.chain_id)
        pair = pair * batch.pair_mask[:, :, :, None].to(pair.dtype)
        return token, pair
