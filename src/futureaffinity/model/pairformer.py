from __future__ import annotations

import torch
from torch import nn

from futureaffinity.config import FutureAffinityConfig

_NEG_INF = -1e9


class Transition(nn.Module):
    """Pre-norm 2-layer MLP with a 4x expansion, used after every mixing step."""

    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Linear(dim * expansion, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class TriangleMultiplicativeUpdate(nn.Module):
    """AlphaFold-style triangular multiplicative update (Algorithms 11/12).

    Lets edge (i, j) gather evidence from every third node k via the product
    of edges (i, k) and (k, j) ("outgoing") or (k, i) and (k, j) ("incoming").
    This is the mechanism that lets distant tokens influence each other
    through a shared neighbor in a single trunk block.
    """

    def __init__(self, pair_dim: int, hidden_dim: int | None = None, mode: str = "outgoing") -> None:
        super().__init__()
        if mode not in ("outgoing", "incoming"):
            raise ValueError(f"mode must be 'outgoing' or 'incoming', got {mode!r}")
        self.mode = mode
        hidden_dim = hidden_dim or pair_dim

        self.norm_in = nn.LayerNorm(pair_dim)
        self.left_proj = nn.Linear(pair_dim, hidden_dim)
        self.left_gate = nn.Linear(pair_dim, hidden_dim)
        self.right_proj = nn.Linear(pair_dim, hidden_dim)
        self.right_gate = nn.Linear(pair_dim, hidden_dim)

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, pair_dim)
        self.out_gate = nn.Linear(pair_dim, pair_dim)

    def forward(self, pair: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        mask = pair_mask[..., None].to(pair.dtype)
        z = self.norm_in(pair)

        left = torch.sigmoid(self.left_gate(z)) * self.left_proj(z) * mask
        right = torch.sigmoid(self.right_gate(z)) * self.right_proj(z) * mask

        if self.mode == "outgoing":
            mixed = torch.einsum("bikc,bjkc->bijc", left, right)
        else:
            mixed = torch.einsum("bkic,bkjc->bijc", left, right)

        update = self.out_proj(self.norm_out(mixed))
        gate = torch.sigmoid(self.out_gate(z))
        return gate * update


class TriangleAttention(nn.Module):
    """AlphaFold-style triangular self-attention (Algorithms 13/14).

    "Starting node" attention: for a fixed row i, token j attends over token
    k using query/key/value drawn from pair[i, :], with an extra bias term
    pulled from pair[j, k] (broadcast across i) -- so the attention pattern
    at edge (i, j) is informed by the *opposite* edge (j, k). "Ending node"
    attention is the same operation applied to the transposed pair tensor.
    """

    def __init__(self, pair_dim: int, num_heads: int, node: str = "starting") -> None:
        super().__init__()
        if node not in ("starting", "ending"):
            raise ValueError(f"node must be 'starting' or 'ending', got {node!r}")
        if pair_dim % num_heads != 0:
            raise ValueError(f"pair_dim ({pair_dim}) must be divisible by num_heads ({num_heads})")

        self.node = node
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads

        self.norm = nn.LayerNorm(pair_dim)
        self.to_q = nn.Linear(pair_dim, pair_dim, bias=False)
        self.to_k = nn.Linear(pair_dim, pair_dim, bias=False)
        self.to_v = nn.Linear(pair_dim, pair_dim, bias=False)
        self.to_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.to_gate = nn.Linear(pair_dim, pair_dim)
        self.to_out = nn.Linear(pair_dim, pair_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, m, _ = x.shape
        return x.view(b, n, m, self.num_heads, self.head_dim)

    def forward(self, pair: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if self.node == "ending":
            pair = pair.transpose(1, 2)

        z = self.norm(pair)
        q, k, v = self._split_heads(self.to_q(z)), self._split_heads(self.to_k(z)), self._split_heads(self.to_v(z))
        bias = self.to_bias(z)  # (B, N_j, N_k, H) -- uses the opposite edge, independent of row i

        logits = torch.einsum("bijhd,bikhd->bijkh", q, k) / (self.head_dim**0.5)
        logits = logits + bias[:, None, :, :, :]

        key_mask = token_mask[:, None, None, :, None]
        logits = logits.masked_fill(~key_mask, _NEG_INF)

        attn = torch.softmax(logits, dim=3)
        out = torch.einsum("bijkh,bikhd->bijhd", attn, v)
        out = out.reshape(*out.shape[:3], -1)

        gate = torch.sigmoid(self.to_gate(z))
        out = gate * self.to_out(out)

        if self.node == "ending":
            out = out.transpose(1, 2)
        return out


class AttentionWithPairBias(nn.Module):
    """Token self-attention biased by the pair representation.

    This is how evidence flows from pair back to token each block: the
    attention logits between tokens i and j get an additive term derived
    from pair[i, j], so the trunk's structural hypothesis directly shapes
    how token information mixes.
    """

    def __init__(self, token_dim: int, pair_dim: int, num_heads: int) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads

        self.norm = nn.LayerNorm(token_dim)
        self.to_q = nn.Linear(token_dim, token_dim, bias=False)
        self.to_k = nn.Linear(token_dim, token_dim, bias=False)
        self.to_v = nn.Linear(token_dim, token_dim, bias=False)
        self.pair_bias = nn.Linear(pair_dim, num_heads, bias=False)
        self.to_out = nn.Linear(token_dim, token_dim)

    def forward(self, token: torch.Tensor, pair: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = token.shape
        z = self.norm(token)

        def split(x: torch.Tensor) -> torch.Tensor:
            return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(self.to_q(z)), split(self.to_k(z)), split(self.to_v(z))

        logits = torch.einsum("bhid,bhjd->bhij", q, k) / (self.head_dim**0.5)
        bias = self.pair_bias(pair).permute(0, 3, 1, 2)  # (B, H, N, N)
        logits = logits + bias

        key_mask = token_mask[:, None, None, :]
        logits = logits.masked_fill(~key_mask, _NEG_INF)

        attn = torch.softmax(logits, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class PairformerBlock(nn.Module):
    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.triangle_mult_outgoing = TriangleMultiplicativeUpdate(config.pair_dim, mode="outgoing")
        self.triangle_mult_incoming = TriangleMultiplicativeUpdate(config.pair_dim, mode="incoming")
        self.triangle_attn_starting = TriangleAttention(config.pair_dim, config.num_attn_heads, node="starting")
        self.triangle_attn_ending = TriangleAttention(config.pair_dim, config.num_attn_heads, node="ending")
        self.pair_transition = Transition(config.pair_dim)

        self.token_attention = AttentionWithPairBias(config.token_dim, config.pair_dim, config.num_attn_heads)
        self.token_transition = Transition(config.token_dim)
        self.dropout = nn.Dropout(config.trunk_dropout)

    def forward(
        self, token: torch.Tensor, pair: torch.Tensor, token_mask: torch.Tensor, pair_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = pair + self.dropout(self.triangle_mult_outgoing(pair, pair_mask))
        pair = pair + self.dropout(self.triangle_mult_incoming(pair, pair_mask))
        pair = pair + self.dropout(self.triangle_attn_starting(pair, token_mask))
        pair = pair + self.dropout(self.triangle_attn_ending(pair, token_mask))
        pair = pair + self.pair_transition(pair)
        pair = pair * pair_mask[..., None].to(pair.dtype)

        token = token + self.dropout(self.token_attention(token, pair, token_mask))
        token = token + self.token_transition(token)
        token = token * token_mask[..., None].to(token.dtype)
        return token, pair


class PairformerTrunk(nn.Module):
    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(PairformerBlock(config) for _ in range(config.num_trunk_blocks))

    def forward(
        self, token: torch.Tensor, pair: torch.Tensor, token_mask: torch.Tensor, pair_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            token, pair = block(token, pair, token_mask, pair_mask)
        return token, pair
