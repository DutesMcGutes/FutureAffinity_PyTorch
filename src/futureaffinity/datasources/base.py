from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class PoseResult:
    """A single generated pose plus its (approximate) binding energy."""

    coords: torch.Tensor  # (N, 3), ligand-atom coordinates for this pose
    energy: float  # arbitrary-units binding energy proxy; lower = more favorable
    source: str  # which backend produced this ("mock", "vina", "openmm")


class DockingSource(Protocol):
    """Generates candidate ligand poses + an energy estimate for a receptor/ligand pair."""

    def dock(
        self,
        receptor_coords: torch.Tensor,
        ligand_atom_types: torch.Tensor,
        num_poses: int = 1,
    ) -> list[PoseResult]:
        ...


class MDSource(Protocol):
    """Refines a starting pose into a short trajectory of coordinates + energies."""

    def simulate(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        num_steps: int = 10,
    ) -> list[PoseResult]:
        ...


class FEPSource(Protocol):
    """Estimates a relative free-energy difference between two ligand/complex states."""

    def relative_free_energy(
        self,
        state_a_coords: torch.Tensor,
        state_b_coords: torch.Tensor,
        atom_types: torch.Tensor,
    ) -> float:
        ...
