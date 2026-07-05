from __future__ import annotations

import torch

from futureaffinity.chem import ligand_atom_index, residue_index_from_one_letter
from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.bindingdb import parse_smiles_elements
from futureaffinity.data.datatypes import Example, collate
from futureaffinity.model.heads.uncertainty import affinity_uncertainty, structural_uncertainty
from futureaffinity.model.model import FutureAffinityModel


def _build_example(config: FutureAffinityConfig, protein_sequence: str, ligand_smiles: str | None, name: str) -> Example:
    protein_types = torch.tensor([residue_index_from_one_letter(c) for c in protein_sequence], dtype=torch.long)
    num_protein = protein_types.shape[0]

    if ligand_smiles:
        elements = parse_smiles_elements(ligand_smiles)
        ligand_types = torch.tensor([ligand_atom_index(e) for e in elements], dtype=torch.long)
        ligand_types = ligand_types + config.residue_vocab_size
    else:
        ligand_types = torch.zeros(0, dtype=torch.long)
    num_ligand = ligand_types.shape[0]

    token_type = torch.cat([protein_types, ligand_types])
    is_ligand = torch.cat([torch.zeros(num_protein, dtype=torch.bool), torch.ones(num_ligand, dtype=torch.bool)])
    chain_id = torch.cat([torch.zeros(num_protein, dtype=torch.long), torch.ones(num_ligand, dtype=torch.long)])
    residue_idx = torch.cat([torch.arange(num_protein, dtype=torch.long), torch.arange(num_ligand, dtype=torch.long)])

    return Example(
        name=name,
        token_type=token_type,
        is_ligand=is_ligand,
        chain_id=chain_id,
        residue_index=residue_idx,
        coords=torch.zeros(num_protein + num_ligand, 3),
        has_structure=False,
    )


@torch.no_grad()
def predict(
    model: FutureAffinityModel,
    protein_sequence: str,
    ligand_smiles: str | None = None,
    mutant_sequence: str | None = None,
    num_samples: int | None = None,
    num_steps: int | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Sequence(+ligand) in, structure ensemble + affinity + uncertainty (+ ddG) out.

    This is the "everything the model can say about one system" entry point:
    a structural ensemble (not one structure), per-token confidence, an
    affinity estimate with an ensemble-derived uncertainty, and -- if
    `mutant_sequence` is given -- a predicted DeltaDeltaG for that mutation.
    """
    config = model.config
    example = _build_example(config, protein_sequence, ligand_smiles, name="query")
    batch = collate([example], config)

    token, pair = model.encode(batch)
    ensemble = model.diffusion.sample_ensemble(
        token, pair, batch.token_mask, num_samples=num_samples, num_steps=num_steps, generator=generator
    )  # (1, S, N, 3)

    confidence = model.confidence_head(token, pair)
    contact_logits = model.contact_head(pair)

    affinity_per_sample = torch.stack(
        [model.affinity_head(token, pair, ensemble[:, s], batch.is_ligand, batch.token_mask) for s in range(ensemble.shape[1])],
        dim=1,
    )  # (1, S)

    result = {
        "structure_ensemble": ensemble[0],  # (S, N, 3)
        "plddt": confidence["plddt"][0],  # (N,)
        "contact_probabilities": torch.sigmoid(contact_logits[0]),  # (N, N)
        "structural_uncertainty": structural_uncertainty(ensemble, batch.token_mask)[0],  # (N,)
        "affinity_mean": float(affinity_per_sample[0].mean()),
        "affinity_std": float(affinity_uncertainty(affinity_per_sample)[0]),
    }

    if mutant_sequence is not None:
        if len(mutant_sequence) != len(protein_sequence):
            raise ValueError("mutant_sequence must be the same length as protein_sequence (point mutations only)")
        mutant_types = torch.tensor([residue_index_from_one_letter(c) for c in mutant_sequence], dtype=torch.long)
        mutant_token_type = torch.cat([mutant_types, batch.token_type[0, mutant_types.shape[0] :]])[None]
        ddg = model.ddg_head(token, batch.token_type, mutant_token_type, batch.is_ligand, batch.token_mask)
        result["ddg"] = float(ddg[0])

    return result
