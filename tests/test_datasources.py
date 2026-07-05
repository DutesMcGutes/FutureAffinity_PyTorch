import pytest
import torch

from futureaffinity.datasources.mock_docking import MockDockingSource, toy_pairwise_energy


def test_toy_pairwise_energy_favors_moderate_separation_over_clash():
    receptor = torch.tensor([[0.0, 0.0, 0.0]])
    ligand_clash = torch.tensor([[0.6, 0.0, 0.0]])  # far inside the repulsive wall
    ligand_ok = torch.tensor([[3.5, 0.0, 0.0]])  # near the LJ minimum

    clash_energy = toy_pairwise_energy(receptor, ligand_clash)
    ok_energy = toy_pairwise_energy(receptor, ligand_ok)

    assert clash_energy > ok_energy


def test_mock_docking_source_returns_requested_number_of_poses_sorted_by_energy():
    torch.manual_seed(0)
    receptor_coords = torch.randn(10, 3) * 5.0
    ligand_types = torch.randint(20, 30, (4,))

    source = MockDockingSource(seed=0)
    poses = source.dock(receptor_coords, ligand_types, num_poses=3)

    assert len(poses) == 3
    energies = [pose.energy for pose in poses]
    assert energies == sorted(energies)
    for pose in poses:
        assert pose.coords.shape == (4, 3)
        assert pose.source == "mock"


def test_vina_adapter_raises_clear_error_without_the_optional_dependency():
    from futureaffinity.datasources.vina_adapter import VinaDockingSource

    with pytest.raises(RuntimeError, match="vina"):
        VinaDockingSource().dock_files("receptor.pdbqt", "ligand.pdbqt", (0.0, 0.0, 0.0), (20.0, 20.0, 20.0))


def test_openmm_adapter_raises_clear_error_without_the_optional_dependency():
    from futureaffinity.datasources.openmm_adapter import OpenMMSimulationSource

    with pytest.raises(RuntimeError, match="openmm"):
        OpenMMSimulationSource().simulate_file("system.pdb")
