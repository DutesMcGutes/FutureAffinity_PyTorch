from __future__ import annotations

import math

import torch

from futureaffinity.datasources.base import PoseResult

_LJ_EPSILON = 0.20  # kcal/mol-scale toy well depth
_LJ_SIGMA = 3.5  # angstrom-scale toy contact distance
_COULOMB_SCALE = 2.0
_MIN_DISTANCE = 0.5


def toy_pairwise_energy(
    receptor_coords: torch.Tensor,
    ligand_coords: torch.Tensor,
    receptor_types: torch.Tensor | None = None,
    ligand_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """A dependency-free Lennard-Jones + toy-electrostatics energy.

    This is NOT a real force field. It exists so the rest of the pipeline
    (synthetic pretraining of the energy head, tests, tutorials) has a
    plausible, fully self-contained energy signal without shelling out to
    AutoDock/OpenMM. See `vina_adapter.py` / `openmm_adapter.py` for real
    physics integration points.
    """
    diff = receptor_coords[:, None, :] - ligand_coords[None, :, :]  # (Nr, Nl, 3)
    dist = diff.norm(dim=-1).clamp(min=_MIN_DISTANCE)

    lj_term = _LJ_EPSILON * ((_LJ_SIGMA / dist) ** 12 - 2.0 * (_LJ_SIGMA / dist) ** 6)

    if receptor_types is not None and ligand_types is not None:
        receptor_charge = (receptor_types.float() % 2.0) * 2.0 - 1.0
        ligand_charge = (ligand_types.float() % 2.0) * 2.0 - 1.0
        coulomb_term = _COULOMB_SCALE * receptor_charge[:, None] * ligand_charge[None, :] / dist
    else:
        coulomb_term = torch.zeros_like(lj_term)

    return (lj_term + coulomb_term).sum()


def _local_ligand_template(num_atoms: int) -> torch.Tensor:
    """A deterministic, fixed internal geometry for a ligand of a given size.

    Real docking varies internal conformation too; this toy generator only
    varies the rigid-body pose (rotation + translation) around this fixed
    template, which is enough to exercise the pipeline end-to-end.
    """
    indices = torch.arange(num_atoms, dtype=torch.float32)
    angle = indices * (2.0 * math.pi / max(num_atoms, 1)) * 2.399963  # golden-angle-ish spiral
    radius = 1.2 * torch.sqrt(indices + 1.0)
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    z = 0.3 * indices
    return torch.stack([x, y, z], dim=-1)


def _random_rotation(generator: torch.Generator) -> torch.Tensor:
    """A random 3x3 rotation matrix, built with torch ops only (no list-of-tensor conversions)."""
    ax, ay, az = (torch.rand(3, generator=generator) * 2.0 * math.pi).unbind()
    one, zero = torch.ones(()), torch.zeros(())

    rx = torch.stack(
        [
            torch.stack([one, zero, zero]),
            torch.stack([zero, torch.cos(ax), -torch.sin(ax)]),
            torch.stack([zero, torch.sin(ax), torch.cos(ax)]),
        ]
    )
    ry = torch.stack(
        [
            torch.stack([torch.cos(ay), zero, torch.sin(ay)]),
            torch.stack([zero, one, zero]),
            torch.stack([-torch.sin(ay), zero, torch.cos(ay)]),
        ]
    )
    rz = torch.stack(
        [
            torch.stack([torch.cos(az), -torch.sin(az), zero]),
            torch.stack([torch.sin(az), torch.cos(az), zero]),
            torch.stack([zero, zero, one]),
        ]
    )
    return rz @ ry @ rx


class MockDockingSource:
    """A pure-PyTorch stand-in for a real docking engine (AutoDock Vina, GNINA, ...).

    Generates candidate rigid-body poses of a ligand around the receptor and
    scores them with `toy_pairwise_energy`. Produces plausible, cheap,
    *synthetic* supervision at arbitrary scale for pretraining the affinity
    head (idea: "generate billions of noisy labels, then fine-tune on scarce
    real affinity data"). It does not model real chemistry.
    """

    def __init__(self, seed: int = 0) -> None:
        self.generator = torch.Generator().manual_seed(seed)

    def dock(
        self,
        receptor_coords: torch.Tensor,
        ligand_atom_types: torch.Tensor,
        num_poses: int = 1,
    ) -> list[PoseResult]:
        num_atoms = ligand_atom_types.shape[0]
        template = _local_ligand_template(num_atoms)
        pocket_center = receptor_coords.mean(dim=0)

        candidates: list[PoseResult] = []
        num_candidates = max(num_poses * 4, num_poses)
        for _ in range(num_candidates):
            rotation = _random_rotation(self.generator)
            jitter = torch.randn(3, generator=self.generator) * 2.0
            translation = pocket_center + jitter
            pose_coords = template @ rotation.T + translation
            energy = toy_pairwise_energy(receptor_coords, pose_coords, ligand_types=ligand_atom_types)
            candidates.append(PoseResult(coords=pose_coords, energy=float(energy), source="mock"))

        candidates.sort(key=lambda pose: pose.energy)
        return candidates[:num_poses]
