import pytest
import torch

from futureaffinity.datasources.analytical_docking import (
    AnalyticalDockingOracle,
    _axis_angle_to_matrix,
    _local_ligand_template,
    analytical_energy,
)


def test_analytical_energy_favors_moderate_separation_over_clash():
    receptor = torch.tensor([[0.0, 0.0, 0.0]])
    ligand_clash = torch.tensor([[0.6, 0.0, 0.0]])  # far inside the repulsive wall
    ligand_ok = torch.tensor([[3.5, 0.0, 0.0]])  # near the LJ minimum

    assert analytical_energy(receptor, ligand_clash) > analytical_energy(receptor, ligand_ok)


def test_axis_angle_to_matrix_is_a_proper_rotation():
    rotation = _axis_angle_to_matrix(torch.tensor([0.3, -1.1, 0.7]))
    assert torch.allclose(rotation @ rotation.T, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-5)


def test_docking_optimizer_actually_lowers_the_energy():
    """The oracle minimizes, so its docked pose must beat a random initial placement in energy."""
    torch.manual_seed(0)
    receptor_coords = torch.randn(12, 3) * 4.0
    ligand_types = torch.randint(20, 30, (5,))

    template = _local_ligand_template(5)
    pocket_center = receptor_coords.mean(dim=0)
    random_pose = template + pocket_center
    random_energy = float(analytical_energy(receptor_coords, random_pose, ligand_types=ligand_types))

    oracle = AnalyticalDockingOracle(num_restarts=4, num_steps=50, seed=0)
    best = oracle.dock(receptor_coords, ligand_types, num_poses=1)[0]
    assert best.energy < random_energy
    assert best.source == "analytical"


def test_docking_returns_requested_poses_sorted_by_energy():
    torch.manual_seed(0)
    receptor_coords = torch.randn(10, 3) * 5.0
    ligand_types = torch.randint(20, 30, (4,))

    poses = AnalyticalDockingOracle(num_restarts=5, num_steps=30, seed=0).dock(receptor_coords, ligand_types, num_poses=3)
    assert len(poses) == 3
    assert [p.energy for p in poses] == sorted(p.energy for p in poses)
    for pose in poses:
        assert pose.coords.shape == (4, 3)


def test_vina_adapter_raises_clear_error_without_the_optional_dependency():
    from futureaffinity.datasources.vina_adapter import VinaDockingSource

    with pytest.raises(RuntimeError, match="vina"):
        VinaDockingSource().dock_files("receptor.pdbqt", "ligand.pdbqt", (0.0, 0.0, 0.0), (20.0, 20.0, 20.0))


def test_openmm_adapter_raises_clear_error_without_the_optional_dependency():
    from futureaffinity.datasources.openmm_adapter import OpenMMSimulationSource

    with pytest.raises(RuntimeError, match="openmm"):
        OpenMMSimulationSource().simulate_file("system.pdb")
