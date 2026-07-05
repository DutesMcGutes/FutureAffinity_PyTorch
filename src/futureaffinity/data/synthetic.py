from __future__ import annotations

import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.datatypes import Batch, Example, collate
from futureaffinity.datasources.analytical_docking import AnalyticalDockingOracle

_CONTACT_THRESHOLD_ANGSTROM = 8.0


def _random_walk_backbone(num_residues: int, generator: torch.Generator, step: float = 3.8) -> torch.Tensor:
    """A crude Ca-trace stand-in: unit steps in random directions, scaled to ~Ca-Ca distance."""
    directions = torch.randn(num_residues, 3, generator=generator)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    steps = directions * step
    return torch.cumsum(steps, dim=0)


def make_synthetic_example(
    config: FutureAffinityConfig,
    protein_length: int = 20,
    ligand_size: int = 6,
    include_ligand: bool = True,
    include_ddg: bool = False,
    include_docking: bool = True,
    name: str = "synthetic",
    seed: int = 0,
) -> Example:
    """Build one fully-labeled (or partially-labeled) synthetic Example.

    Ligand atoms are placed near the tail of the protein backbone to emulate
    a binding pocket, so distance-derived labels (contacts, affinity) carry a
    learnable signal instead of being pure noise.
    """
    generator = torch.Generator().manual_seed(seed)

    protein_types = torch.randint(0, config.residue_vocab_size, (protein_length,), generator=generator)
    backbone = _random_walk_backbone(protein_length, generator)

    token_type = protein_types
    is_ligand = torch.zeros(protein_length, dtype=torch.bool)
    chain_id = torch.zeros(protein_length, dtype=torch.long)
    residue_index = torch.arange(protein_length, dtype=torch.long)
    coords = backbone

    if include_ligand and ligand_size > 0:
        ligand_types = config.residue_vocab_size + torch.randint(
            0, config.ligand_atom_vocab_size, (ligand_size,), generator=generator
        )
        pocket_center = backbone[-min(3, protein_length):].mean(dim=0)
        ligand_coords = pocket_center + torch.randn(ligand_size, 3, generator=generator) * 2.0

        token_type = torch.cat([token_type, ligand_types])
        is_ligand = torch.cat([is_ligand, torch.ones(ligand_size, dtype=torch.bool)])
        chain_id = torch.cat([chain_id, torch.ones(ligand_size, dtype=torch.long)])
        residue_index = torch.cat([residue_index, torch.arange(ligand_size, dtype=torch.long)])
        coords = torch.cat([coords, ligand_coords], dim=0)

    distances = torch.cdist(coords, coords)
    contacts = (distances < _CONTACT_THRESHOLD_ANGSTROM).float()

    example = Example(
        name=name,
        token_type=token_type,
        is_ligand=is_ligand,
        chain_id=chain_id,
        residue_index=residue_index,
        coords=coords,
        has_structure=True,
        contacts=contacts,
        has_contacts=True,
    )

    if include_ligand and ligand_size > 0:
        protein_coords, ligand_coords_only = coords[:protein_length], coords[protein_length:]
        cross_distances = torch.cdist(protein_coords, ligand_coords_only)
        mean_contact_distance = cross_distances.min(dim=0).values.mean()
        noise = torch.randn((), generator=generator) * 0.3
        example.affinity = float(10.0 - mean_contact_distance / 2.0 + noise)
        example.has_affinity = True

        if include_docking:
            # cheap settings here (few restarts/steps): synthetic-example generation just needs a
            # plausible energy label, not a fully converged pose. The real docking demo uses the
            # oracle's stronger defaults.
            docking_source = AnalyticalDockingOracle(num_restarts=2, num_steps=20, seed=seed)
            best_pose = docking_source.dock(protein_coords, token_type[protein_length:], num_poses=1)[0]
            example.docking_energy = best_pose.energy
            example.has_docking = True

        if include_ddg:
            mutant_token_type = token_type.clone()
            mutation_site = int(torch.randint(0, protein_length, (1,), generator=generator).item())
            mutant_token_type[mutation_site] = int(
                torch.randint(0, config.residue_vocab_size, (1,), generator=generator).item()
            )
            distance_to_pocket = cross_distances[mutation_site].min()
            magnitude = torch.randn((), generator=generator) * (2.0 / (1.0 + distance_to_pocket / 5.0))
            example.mutant_token_type = mutant_token_type
            example.ddg = float(magnitude)
            example.has_ddg = True

    return example


def make_synthetic_batch(
    config: FutureAffinityConfig,
    batch_size: int = 4,
    protein_length: int = 20,
    ligand_size: int = 6,
    seed: int = 0,
) -> Batch:
    examples = [
        make_synthetic_example(
            config,
            protein_length=protein_length,
            ligand_size=ligand_size,
            include_ddg=(i % 2 == 0),
            name=f"synthetic_{i}",
            seed=seed + i,
        )
        for i in range(batch_size)
    ]
    return collate(examples, config)
