from __future__ import annotations

import math

import torch

from futureaffinity.datasources.base import PoseResult

_LJ_EPSILON = 0.20  # kcal/mol-scale well depth
_LJ_SIGMA = 3.5  # angstrom-scale contact distance
_COULOMB_SCALE = 2.0
_MIN_DISTANCE = 0.5


def analytical_energy(
    receptor_coords: torch.Tensor,
    ligand_coords: torch.Tensor,
    receptor_types: torch.Tensor | None = None,
    ligand_types: torch.Tensor | None = None,
) -> torch.Tensor:
    """A differentiable Lennard-Jones (12-6) + toy-electrostatics interaction energy.

    A genuine analytical potential -- a real (if deliberately simple) force field with a
    repulsive wall, an attractive well, and a charge term -- not a random score. It's fully
    differentiable, which is what makes gradient-based pose optimization below possible, and
    it's the same functional form MM force fields use for non-bonded terms. It is NOT
    parameterized to real chemistry (no atom-typed epsilon/sigma, no solvent), so it's a
    synthetic-supervision oracle, not a substitute for Vina/OpenMM -- see those adapters.
    """
    diff = receptor_coords[:, None, :] - ligand_coords[None, :, :]
    dist = diff.norm(dim=-1).clamp(min=_MIN_DISTANCE)

    lj_term = _LJ_EPSILON * ((_LJ_SIGMA / dist) ** 12 - 2.0 * (_LJ_SIGMA / dist) ** 6)

    if receptor_types is not None and ligand_types is not None:
        receptor_charge = (receptor_types.float() % 2.0) * 2.0 - 1.0
        ligand_charge = (ligand_types.float() % 2.0) * 2.0 - 1.0
        coulomb_term = _COULOMB_SCALE * receptor_charge[:, None] * ligand_charge[None, :] / dist
    else:
        coulomb_term = torch.zeros_like(lj_term)

    return (lj_term + coulomb_term).sum()


# backwards-compatible alias for the earlier name
toy_pairwise_energy = analytical_energy


def _local_ligand_template(num_atoms: int) -> torch.Tensor:
    """A deterministic fixed internal geometry for a ligand of a given size (a rigid conformer).

    This oracle optimizes only the rigid-body pose (rotation + translation) of the ligand, not
    its internal torsions -- a documented simplification, the same rigid-docking assumption
    classical docking makes for a single input conformer.
    """
    indices = torch.arange(num_atoms, dtype=torch.float32)
    angle = indices * (2.0 * math.pi / max(num_atoms, 1)) * 2.399963
    radius = 1.2 * torch.sqrt(indices + 1.0)
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle), 0.3 * indices], dim=-1)


def _axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rotation matrix from an axis-angle 3-vector via the matrix exponential of its skew form.

    Fully differentiable in `axis_angle`, so the rotation is a first-class optimization variable.
    """
    zero = axis_angle.new_zeros(())
    wx, wy, wz = axis_angle[0], axis_angle[1], axis_angle[2]
    skew = torch.stack(
        [
            torch.stack([zero, -wz, wy]),
            torch.stack([wz, zero, -wx]),
            torch.stack([-wy, wx, zero]),
        ]
    )
    return torch.matrix_exp(skew)


class AnalyticalDockingOracle:
    """A real (if toy) rigid-body docking engine: analytical force field + gradient optimization.

    Unlike a random-pose sampler, this actually *minimizes* `analytical_energy` over the
    ligand's rigid-body pose (an axis-angle rotation and a translation) with Adam, from several
    random restarts, and returns the lowest-energy local minima it finds. That is the same
    shape as a classical docking run -- a scoring function optimized over pose, with multi-start
    to escape local minima -- just with a simplified potential and rigid ligand. It produces
    cheap, plentiful *synthetic* pose/energy supervision for pretraining the affinity head, and
    shares the `dock(...)` interface with `vina_adapter.py` / `openmm_adapter.py` so a real
    engine can be swapped in unchanged.
    """

    def __init__(self, num_restarts: int = 6, num_steps: int = 60, lr: float = 0.05, seed: int = 0) -> None:
        self.num_restarts = num_restarts
        self.num_steps = num_steps
        self.lr = lr
        self.generator = torch.Generator().manual_seed(seed)

    def _minimize_one(
        self, receptor_coords: torch.Tensor, template: torch.Tensor, ligand_types: torch.Tensor, pocket_center: torch.Tensor
    ) -> PoseResult:
        axis_angle = (torch.rand(3, generator=self.generator) * 2.0 - 1.0) * math.pi
        translation = pocket_center + torch.randn(3, generator=self.generator) * 2.0
        axis_angle = axis_angle.clone().requires_grad_(True)
        translation = translation.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([axis_angle, translation], lr=self.lr)
        for _ in range(self.num_steps):
            optimizer.zero_grad()
            rotation = _axis_angle_to_matrix(axis_angle)
            pose = template @ rotation.T + translation
            energy = analytical_energy(receptor_coords, pose, ligand_types=ligand_types)
            energy.backward()
            optimizer.step()

        with torch.no_grad():
            rotation = _axis_angle_to_matrix(axis_angle)
            pose = template @ rotation.T + translation
            energy = analytical_energy(receptor_coords, pose, ligand_types=ligand_types)
        return PoseResult(coords=pose.detach(), energy=float(energy), source="analytical")

    def dock(
        self, receptor_coords: torch.Tensor, ligand_atom_types: torch.Tensor, num_poses: int = 1
    ) -> list[PoseResult]:
        template = _local_ligand_template(ligand_atom_types.shape[0])
        pocket_center = receptor_coords.mean(dim=0)
        poses = [
            self._minimize_one(receptor_coords, template, ligand_atom_types, pocket_center)
            for _ in range(max(self.num_restarts, num_poses))
        ]
        poses.sort(key=lambda pose: pose.energy)
        return poses[:num_poses]
