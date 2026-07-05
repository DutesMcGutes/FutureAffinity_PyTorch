"""Fetch and parse real protein-ligand complexes from the RCSB PDB.

Unlike PDBbind (academic license, no auto-download), individual PDB entries are freely
downloadable from RCSB with no login. This module parses a real .pdb file into a FutureAffinity
`Example` -- protein Ca trace plus a chosen ligand's heavy atoms -- so the pipeline can run on
genuinely real coordinates, not just synthetic ones. The parser is hermetic (unit-tested on a
bundled fixture); `fetch_pdb_complex` is a thin urllib wrapper over it.
"""
from __future__ import annotations

from pathlib import Path

import torch

from futureaffinity.chem import ligand_atom_index, residue_index
from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.datatypes import Example

_RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
_SKIP_LIGANDS = {"HOH", "WAT", "SO4", "PO4", "GOL", "EDO", "CL", "NA", "K", "MG", "ZN", "CA"}


def _element_from_atom_name(atom_name: str, columns: str) -> str:
    element = columns[76:78].strip() if len(columns) >= 78 else ""
    if element:
        return element.upper()
    return "".join(ch for ch in atom_name if ch.isalpha())[:1].upper()


def parse_pdb_complex(pdb_text: str, config: FutureAffinityConfig, ligand_code: str | None = None) -> Example:
    """Parse a .pdb string into an Example: protein Ca trace + one ligand's heavy atoms.

    If `ligand_code` is None, the first non-solvent HETATM residue encountered is used.
    """
    protein_types, protein_coords = [], []
    seen_residues = set()
    ligand_types, ligand_coords = [], []
    chosen_ligand: str | None = ligand_code

    for line in pdb_text.splitlines():
        record = line[:6].strip()
        if record == "ATOM" and line[12:16].strip() == "CA":
            key = (line[21].strip(), line[22:26].strip())
            if key in seen_residues:
                continue
            seen_residues.add(key)
            protein_types.append(residue_index(line[17:20].strip()))
            protein_coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        elif record == "HETATM":
            residue_name = line[17:20].strip()
            if residue_name in _SKIP_LIGANDS:
                continue
            if chosen_ligand is None:
                chosen_ligand = residue_name
            if residue_name != chosen_ligand:
                continue
            if line[16] not in (" ", "A"):  # skip alternate locations beyond the first
                continue
            ligand_types.append(ligand_atom_index(_element_from_atom_name(line[12:16].strip(), line)))
            ligand_coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))

    if not protein_types:
        raise ValueError("No protein CA atoms found in the PDB text.")
    if not ligand_types:
        raise ValueError(f"No ligand heavy atoms found (looked for {chosen_ligand or 'any non-solvent HETATM'}).")

    protein_type_t = torch.tensor(protein_types, dtype=torch.long)
    ligand_type_t = torch.tensor(ligand_types, dtype=torch.long) + config.residue_vocab_size
    num_protein, num_ligand = protein_type_t.shape[0], ligand_type_t.shape[0]

    return Example(
        name=f"rcsb_{chosen_ligand}",
        token_type=torch.cat([protein_type_t, ligand_type_t]),
        is_ligand=torch.cat([torch.zeros(num_protein, dtype=torch.bool), torch.ones(num_ligand, dtype=torch.bool)]),
        chain_id=torch.cat([torch.zeros(num_protein, dtype=torch.long), torch.ones(num_ligand, dtype=torch.long)]),
        residue_index=torch.cat([torch.arange(num_protein), torch.arange(num_ligand)]).long(),
        coords=torch.tensor(protein_coords + ligand_coords, dtype=torch.float32),
        has_structure=True,
        has_contacts=False,
    )


def fetch_pdb_complex(
    pdb_id: str, config: FutureAffinityConfig, ligand_code: str | None = None, cache_dir: str | Path | None = None
) -> Example:
    """Download `{pdb_id}.pdb` from RCSB (caching locally) and parse it into an Example.

    Network access required on first call; cached thereafter. RCSB entries are freely available.
    """
    import urllib.request

    pdb_id = pdb_id.lower()
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"{pdb_id}.pdb"
        if cache_path.exists():
            return parse_pdb_complex(cache_path.read_text(encoding="utf-8"), config, ligand_code)

    with urllib.request.urlopen(_RCSB_URL.format(pdb_id=pdb_id)) as response:
        pdb_text = response.read().decode("utf-8")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(pdb_text, encoding="utf-8")
    return parse_pdb_complex(pdb_text, config, ligand_code)
