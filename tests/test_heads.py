import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.synthetic import make_synthetic_batch
from futureaffinity.model.model import FutureAffinityModel
from futureaffinity.model.heads.confidence import compute_lddt
from futureaffinity.model.heads.uncertainty import affinity_uncertainty, structural_uncertainty


def test_compute_lddt_is_100_for_identical_structures():
    torch.manual_seed(0)
    coords = torch.randn(2, 10, 3) * 5.0
    mask = torch.ones(2, 10, dtype=torch.bool)
    lddt = compute_lddt(coords, coords, mask)
    assert torch.allclose(lddt, torch.full_like(lddt, 100.0), atol=1e-3)


def test_compute_lddt_drops_when_structure_is_perturbed():
    torch.manual_seed(0)
    coords = torch.randn(1, 10, 3) * 5.0
    mask = torch.ones(1, 10, dtype=torch.bool)
    perturbed = coords + torch.randn_like(coords) * 10.0

    identical_score = compute_lddt(coords, coords, mask)
    perturbed_score = compute_lddt(perturbed, coords, mask)
    assert perturbed_score.mean() < identical_score.mean()


def test_model_compute_losses_shapes_and_gradients():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=4, protein_length=10, ligand_size=4, seed=6)
    model = FutureAffinityModel(config)

    losses, predictions = model.compute_losses(batch)
    for name, loss in losses.items():
        assert loss.shape == (4,), name
        assert torch.isfinite(loss).all(), name

    total = sum(loss.mean() for loss in losses.values())
    total.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert missing == []

    assert predictions["contact_logits"].shape == (4, batch.num_tokens, batch.num_tokens)
    assert predictions["affinity"].shape == (4,)
    assert predictions["ddg"].shape == (4,)


def test_ddg_head_only_activates_on_changed_protein_positions():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=2, protein_length=10, ligand_size=3, seed=7)
    model = FutureAffinityModel(config)
    token, pair = model.encode(batch)

    identical_mutant = batch.token_type.clone()
    ddg_no_mutation = model.ddg_head(token, batch.token_type, identical_mutant, batch.is_ligand, batch.token_mask)
    assert torch.allclose(ddg_no_mutation, torch.zeros_like(ddg_no_mutation))


def test_structural_uncertainty_zero_for_identical_ensemble():
    coords = torch.randn(1, 8, 3)
    ensemble = coords[:, None, :, :].expand(-1, 5, -1, -1)
    mask = torch.ones(1, 8, dtype=torch.bool)
    rmsf = structural_uncertainty(ensemble, mask)
    assert torch.allclose(rmsf, torch.zeros_like(rmsf), atol=1e-5)


def test_affinity_uncertainty_matches_std():
    predictions = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = affinity_uncertainty(predictions)
    expected = predictions.std(dim=1, unbiased=False)
    assert torch.allclose(result, expected)
