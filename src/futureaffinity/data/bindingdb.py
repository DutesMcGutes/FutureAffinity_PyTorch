from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import torch

from futureaffinity.chem import ligand_atom_index, residue_index_from_one_letter
from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.datatypes import Example

# preference order: direct binding measurements (Kd, Ki) before functional-assay potencies
_AFFINITY_COLUMNS = ["Kd (nM)", "Ki (nM)", "IC50 (nM)", "EC50 (nM)"]
_SEQUENCE_COLUMN = "BindingDB Target Chain Sequence"
_SMILES_COLUMN = "Ligand SMILES"

# SMILES "organic subset" atoms: bracket atoms, two-letter halogens, or one-letter
# aliphatic/aromatic organic atoms. Ignores bonds, charges, stereochemistry, and
# aromaticity beyond upper-casing -- a bag-of-elements read, not a real SMILES parser.
_SMILES_ATOM_PATTERN = re.compile(r"\[[^\]]+\]|Br|Cl|[BCNOFPSIbcnops]")


def _parse_affinity_nanomolar(raw_value: str | None) -> float | None:
    if not raw_value:
        return None
    cleaned = raw_value.strip().lstrip("<>~")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def _nanomolar_to_p_affinity(value_nanomolar: float) -> float:
    """-log10(concentration in molar); e.g. 1 nM -> pAffinity 9.0."""
    return 9.0 - math.log10(value_nanomolar)


def parse_smiles_elements(smiles: str) -> list[str]:
    elements = []
    for token in _SMILES_ATOM_PATTERN.findall(smiles):
        if token.startswith("["):
            match = re.match(r"[A-Za-z]{1,2}", token[1:-1])
            element = match.group(0) if match else "C"
        else:
            element = token
        elements.append(element.upper())
    return elements


class BindingDBDataset:
    """Sequence + SMILES + affinity examples, with no structural label at all.

    Reads BindingDB's bulk TSV export directly (tab-separated, header row with
    BindingDB's real column names). Every example has `has_structure=False`:
    the affinity head still scores these rows because it's evaluated against
    the model's own diffusion rollout, not against ground-truth coordinates
    (see `model/model.py`), so sequence-only affinity supervision is usable
    without ever needing a structure for these rows.
    """

    def __init__(self, tsv_path: str | Path, config: FutureAffinityConfig) -> None:
        self.config = config
        self.rows: list[tuple[str, str, float]] = []
        with open(tsv_path, encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                sequence = (row.get(_SEQUENCE_COLUMN) or "").strip()
                smiles = (row.get(_SMILES_COLUMN) or "").strip()
                if not sequence or not smiles:
                    continue

                affinity = None
                for column in _AFFINITY_COLUMNS:
                    parsed = _parse_affinity_nanomolar(row.get(column))
                    if parsed is not None:
                        affinity = _nanomolar_to_p_affinity(parsed)
                        break
                if affinity is None:
                    continue

                self.rows.append((sequence, smiles, affinity))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int) -> Example:
        sequence, smiles, affinity = self.rows[position]

        protein_types = torch.tensor([residue_index_from_one_letter(c) for c in sequence], dtype=torch.long)
        ligand_elements = parse_smiles_elements(smiles)
        ligand_types = torch.tensor([ligand_atom_index(e) for e in ligand_elements], dtype=torch.long)
        ligand_types = ligand_types + self.config.residue_vocab_size

        num_protein, num_ligand = protein_types.shape[0], ligand_types.shape[0]
        return Example(
            name=f"bindingdb_{position}",
            token_type=torch.cat([protein_types, ligand_types]),
            is_ligand=torch.cat(
                [torch.zeros(num_protein, dtype=torch.bool), torch.ones(num_ligand, dtype=torch.bool)]
            ),
            chain_id=torch.cat(
                [torch.zeros(num_protein, dtype=torch.long), torch.ones(num_ligand, dtype=torch.long)]
            ),
            residue_index=torch.cat(
                [torch.arange(num_protein, dtype=torch.long), torch.arange(num_ligand, dtype=torch.long)]
            ),
            coords=torch.zeros(num_protein + num_ligand, 3),
            has_structure=False,
            affinity=affinity,
            has_affinity=True,
        )
