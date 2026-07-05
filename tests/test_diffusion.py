import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.model.diffusion import DiffusionModule, karras_sigma_schedule
from futureaffinity.model.embedding import InputEmbedder
from futureaffinity.model.pairformer import PairformerTrunk
from futureaffinity.data.synthetic import make_synthetic_batch


def _encode(config, batch):
    embedder = InputEmbedder(config)
    trunk = PairformerTrunk(config)
    token, pair = embedder(batch)
    return trunk(token, pair, batch.token_mask, batch.pair_mask)


def test_karras_schedule_is_decreasing_and_ends_at_zero():
    sigmas = karras_sigma_schedule(8, sigma_min=0.002, sigma_max=80.0)
    assert sigmas.shape[0] == 9
    assert sigmas[-1].item() == 0.0
    assert torch.all(sigmas[:-1] > sigmas[1:])


def test_training_loss_is_finite_and_per_example():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=3, protein_length=8, ligand_size=3, seed=3)
    token, pair = _encode(config, batch)

    diffusion = DiffusionModule(config)
    loss = diffusion.training_loss(batch.coords, token, pair, batch.token_mask)

    assert loss.shape == (3,)
    assert torch.isfinite(loss).all()
    assert (loss >= 0).all()


def test_sample_ensemble_shape_and_variation():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=2, protein_length=6, ligand_size=2, seed=4)
    token, pair = _encode(config, batch)

    diffusion = DiffusionModule(config)
    ensemble = diffusion.sample_ensemble(token, pair, batch.token_mask, num_samples=4, num_steps=5)

    assert ensemble.shape == (2, 4, batch.num_tokens, 3)
    assert torch.isfinite(ensemble).all()
    # independent samples should not be identical
    assert not torch.allclose(ensemble[:, 0], ensemble[:, 1])


def test_diffusion_loss_backpropagates_into_trunk():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=2, protein_length=6, ligand_size=2, seed=5)

    embedder = InputEmbedder(config)
    trunk = PairformerTrunk(config)
    diffusion = DiffusionModule(config)

    token, pair = embedder(batch)
    token, pair = trunk(token, pair, batch.token_mask, batch.pair_mask)
    loss = diffusion.training_loss(batch.coords, token, pair, batch.token_mask).mean()
    loss.backward()

    grads = [p.grad for p in trunk.parameters()]
    assert all(g is not None for g in grads)
