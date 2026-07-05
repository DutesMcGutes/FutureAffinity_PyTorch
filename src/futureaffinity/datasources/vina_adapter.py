from __future__ import annotations

from dataclasses import dataclass

from futureaffinity.datasources.base import PoseResult

_INSTALL_HINT = (
    "VinaDockingSource requires the 'vina' python package and a working AutoDock Vina "
    "installation (pip install vina). This is a real integration point, not part of the "
    "default install -- it isn't needed to train or run the core model."
)


def _parse_pdbqt_coords(pdbqt_text: str):
    """Extract atom coordinates from a PDBQT/PDB-formatted block (fixed-column format)."""
    import torch

    coords = []
    for line in pdbqt_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((x, y, z))
    return torch.tensor(coords, dtype=torch.float32)


@dataclass
class VinaDockingSource:
    """Real AutoDock Vina docking, operating on prepared receptor/ligand files.

    Unlike `MockDockingSource`, real docking needs actual molecule files
    (receptor PDBQT + ligand PDBQT with correct bonds/charges/torsions) --
    information FutureAffinity's toy per-atom-type tensors don't carry. This
    adapter is therefore file-based rather than tensor-based: it's a real
    integration point for when the data pipeline can serialize a token-level
    example back out to real structure files, not a drop-in replacement for
    `DockingSource.dock()`.
    """

    exhaustiveness: int = 8

    def dock_files(
        self,
        receptor_pdbqt: str,
        ligand_pdbqt: str,
        center: tuple[float, float, float],
        box_size: tuple[float, float, float],
        num_poses: int = 1,
    ) -> list[PoseResult]:
        try:
            from vina import Vina
        except ImportError as error:
            raise RuntimeError(_INSTALL_HINT) from error

        docker = Vina(sf_name="vina")
        docker.set_receptor(receptor_pdbqt)
        docker.set_ligand_from_file(ligand_pdbqt)
        docker.compute_vina_maps(center=list(center), box_size=list(box_size))
        docker.dock(exhaustiveness=self.exhaustiveness, n_poses=num_poses)

        energies = docker.energies(n_poses=num_poses)
        poses_pdbqt = docker.poses(n_poses=num_poses).split("ENDMDL")

        results = []
        for pose_energies, pose_block in zip(energies, poses_pdbqt):
            if not pose_block.strip():
                continue
            results.append(
                PoseResult(coords=_parse_pdbqt_coords(pose_block), energy=float(pose_energies[0]), source="vina")
            )
        return results
