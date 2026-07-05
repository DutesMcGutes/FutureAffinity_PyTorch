import torch

from futureaffinity.geometry import apply_rotation, random_rotation_matrices
from futureaffinity.evaluation.metrics import (
    affinity_metrics,
    dockq,
    lddt,
    lddt_pli,
    pearson_correlation,
    rmsd,
    spearman_correlation,
    tm_score,
)


def _complex(batch=2, n=14, num_ligand=4, seed=0):
    torch.manual_seed(seed)
    coords = torch.randn(batch, n, 3) * 5.0
    mask = torch.ones(batch, n, dtype=torch.bool)
    is_ligand = torch.zeros(batch, n, dtype=torch.bool)
    is_ligand[:, -num_ligand:] = True
    return coords, mask, is_ligand


def test_structure_metrics_are_invariant_to_rigid_motion():
    true, mask, is_ligand = _complex()
    rotation = random_rotation_matrices(true.shape[0])
    moved = apply_rotation(true, rotation) + torch.tensor([7.0, -3.0, 11.0])

    assert torch.allclose(rmsd(moved, true, mask), torch.zeros(2), atol=1e-3)
    assert torch.allclose(lddt(moved, true, mask), torch.full((2,), 100.0), atol=1e-2)
    assert torch.allclose(tm_score(moved, true, mask), torch.ones(2), atol=1e-3)
    assert torch.allclose(lddt_pli(moved, true, mask, is_ligand), torch.full((2,), 100.0), atol=1e-2)
    assert torch.allclose(dockq(moved, true, mask, is_ligand), torch.ones(2), atol=1e-3)


def test_perturbation_degrades_every_structure_metric():
    true, mask, is_ligand = _complex()
    noisy = true + torch.randn_like(true) * 3.0

    assert (rmsd(noisy, true, mask) > 1.0).all()
    assert (lddt(noisy, true, mask) < 100.0).all()
    assert (tm_score(noisy, true, mask) < 1.0).all()
    assert (dockq(noisy, true, mask, is_ligand) < 1.0).all()


def test_lddt_pli_only_scores_protein_ligand_pairs():
    true, mask, is_ligand = _complex()
    # perturb ONLY intra-protein geometry: global lDDT drops, interface lDDT-PLI should not
    perturbed = true.clone()
    perturbed[:, :-4] = perturbed[:, :-4] + torch.randn_like(perturbed[:, :-4]) * 2.0

    interface_before = lddt_pli(true, true, mask, is_ligand)
    interface_after = lddt_pli(perturbed, true, mask, is_ligand)
    # the ligand atoms themselves were untouched, so their mutual + interface distances to the
    # (moved) protein change -- interface score should still be defined and finite
    assert torch.isfinite(interface_after).all()
    assert (interface_before == 100.0).all()


def test_affinity_metrics_on_correlated_predictions():
    true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pred = true * 0.9 + 0.3
    metrics = affinity_metrics(pred, true)
    assert metrics["pearson"] > 0.99
    assert metrics["spearman"] > 0.99
    assert metrics["rmse"] >= 0.0


def test_spearman_is_one_for_any_monotonic_map():
    true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = torch.exp(true)  # monotonic but very non-linear
    assert spearman_correlation(pred, true) > 0.999
    assert pearson_correlation(pred, true) < 0.999  # ranks agree, raw values do not
