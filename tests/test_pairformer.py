import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.model.embedding import InputEmbedder
from futureaffinity.model.pairformer import PairformerTrunk
from futureaffinity.data.synthetic import make_synthetic_batch


def test_pairformer_trunk_preserves_shapes_and_respects_mask():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=3, protein_length=8, ligand_size=3, seed=1)

    embedder = InputEmbedder(config)
    trunk = PairformerTrunk(config)

    token, pair = embedder(batch)
    out_token, out_pair = trunk(token, pair, batch.token_mask, batch.pair_mask)

    assert out_token.shape == token.shape
    assert out_pair.shape == pair.shape
    assert torch.isfinite(out_token).all()
    assert torch.isfinite(out_pair).all()

    # padded positions should stay exactly zero (masked at every block)
    pad_mask = ~batch.token_mask
    if pad_mask.any():
        assert torch.allclose(out_token[pad_mask], torch.zeros_like(out_token[pad_mask]))


def test_triangle_multiplicative_update_outgoing_vs_incoming_differ():
    from futureaffinity.model.pairformer import TriangleMultiplicativeUpdate

    torch.manual_seed(0)
    pair = torch.randn(2, 6, 6, 8)
    mask = torch.ones(2, 6, 6, dtype=torch.bool)

    outgoing = TriangleMultiplicativeUpdate(8, mode="outgoing")
    incoming = TriangleMultiplicativeUpdate(8, mode="incoming")

    out_result = outgoing(pair, mask)
    in_result = incoming(pair, mask)

    assert out_result.shape == pair.shape
    assert in_result.shape == pair.shape
    assert not torch.allclose(out_result, in_result)


def test_gradients_flow_through_trunk():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=2, protein_length=6, ligand_size=2, seed=2)

    embedder = InputEmbedder(config)
    trunk = PairformerTrunk(config)
    token, pair = embedder(batch)
    out_token, out_pair = trunk(token, pair, batch.token_mask, batch.pair_mask)

    loss = out_token.sum() + out_pair.sum()
    loss.backward()

    grads = [p.grad for p in list(embedder.parameters()) + list(trunk.parameters())]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
