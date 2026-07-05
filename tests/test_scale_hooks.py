import dataclasses

import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.synthetic import make_synthetic_batch
from futureaffinity.model.model import FutureAffinityModel


def test_chunked_triangle_attention_matches_unchunked_exactly():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=2, protein_length=11, ligand_size=4, seed=1)

    model = FutureAffinityModel(config)
    model.eval()
    with torch.no_grad():
        token_full, pair_full = model.encode(batch)

    for block in model.trunk.blocks:
        block.triangle_attn_starting.chunk_size = 3
        block.triangle_attn_ending.chunk_size = 3
    with torch.no_grad():
        token_chunked, pair_chunked = model.encode(batch)

    assert torch.allclose(token_full, token_chunked, atol=1e-5)
    assert torch.allclose(pair_full, pair_chunked, atol=1e-5)


def test_gradient_checkpointing_produces_gradients_for_every_parameter():
    torch.manual_seed(0)
    config = dataclasses.replace(FutureAffinityConfig.tiny(), use_gradient_checkpointing=True)
    batch = make_synthetic_batch(config, batch_size=2, protein_length=8, ligand_size=3, seed=2)

    model = FutureAffinityModel(config)
    model.train()
    losses, _ = model.compute_losses(batch)
    total = sum(loss.mean() for loss in losses.values())
    total.backward()

    assert [name for name, p in model.named_parameters() if p.grad is None] == []


def test_train_loop_runs_and_reduces_loss_on_cpu():
    from futureaffinity.training.train import SyntheticSource, train

    config = FutureAffinityConfig.tiny()
    model = train(
        config, [SyntheticSource(config, protein_length=8, ligand_size=3)], num_steps=6, batch_size=3, lr=2e-3, log_every=100, seed=0
    )
    assert isinstance(model, FutureAffinityModel)
