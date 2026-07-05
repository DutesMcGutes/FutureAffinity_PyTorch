from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import torch

from futureaffinity.chem import ligand_atom_index, residue_index
from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.datatypes import Example

_INDEX_LINE = re.compile(
    r"^(?P<pdb_id>\S+)\s+(?P<resolution>\S+)\s+(?P<year>\d+)\s+(?P<neg_log_affinity>\S+)\s+(?P<measure>\S+)"
    r"(?:\s*//\s*(?P<comment>.*))?$"
)
_LIGAND_CODE = re.compile(r"\(([A-Za-z0-9]{2,4})\)")


@dataclass
class PDBbindRecord:
    """One row of a PDBbind general/refined-set index file.

    `neg_log_affinity` is the "-logKd/Ki" column: PDBbind's own name for
    pKd/pKi (higher = tighter binder), already on the scale most affinity
    models train against.
    """

    pdb_id: str
    resolution: float | None
    year: int
    neg_log_affinity: float
    measure: str
    ligand_code: str | None


def parse_pdbbind_index(path: str | Path) -> list[PDBbindRecord]:
    """Parse a PDBbind `INDEX_general_PL_data.<year>`-style file.

    Real format: comment lines start with '#'; data lines are whitespace-
    separated `pdb_id resolution year -logKd/Ki Kd/Ki-string // comment`,
    where the comment often ends with the ligand's 3-4 character HET code in
    parentheses, e.g. `// 3zzf.pdf (0QH)`.
    """
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _INDEX_LINE.match(line)
        if not match:
            continue
        resolution_str = match.group("resolution")
        ligand_match = _LIGAND_CODE.search(match.group("comment") or "")
        records.append(
            PDBbindRecord(
                pdb_id=match.group("pdb_id"),
                resolution=None if resolution_str.upper() == "NMR" else float(resolution_str),
                year=int(match.group("year")),
                neg_log_affinity=float(match.group("neg_log_affinity")),
                measure=match.group("measure"),
                ligand_code=ligand_match.group(1) if ligand_match else None,
            )
        )
    return records


def parse_pdb_ca_trace(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal PDB parser: one entry per residue, taken from its CA atom.

    Reads fixed-column ATOM records (standard PDB format). Only the first
    occurrence of each residue's CA is kept, so alternate locations and
    multi-model files are handled by simply taking whichever comes first --
    fine for a pocket file, not a substitute for a full structure parser.
    """
    token_types, coords = [], []
    seen_residues = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        residue_name = line[17:20].strip()
        chain = line[21].strip()
        residue_seq = line[22:26].strip()
        key = (chain, residue_seq)
        if key in seen_residues:
            continue
        seen_residues.add(key)

        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        token_types.append(residue_index(residue_name))
        coords.append((x, y, z))

    return torch.tensor(token_types, dtype=torch.long), torch.tensor(coords, dtype=torch.float32)


def parse_sdf_ligand(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Minimal SDF (V2000) parser: atom block only (element + xyz), first molecule in the file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    counts_line = lines[3]
    num_atoms = int(counts_line[0:3])

    token_types, coords = [], []
    for line in lines[4 : 4 + num_atoms]:
        x, y, z = float(line[0:10]), float(line[10:20]), float(line[20:30])
        element = line[31:34].strip()
        token_types.append(ligand_atom_index(element))
        coords.append((x, y, z))

    return torch.tensor(token_types, dtype=torch.long), torch.tensor(coords, dtype=torch.float32)


class PDBbindDataset:
    """Loads PDBbind-style structure + affinity examples from a standard on-disk layout.

    Expects `{root}/index/INDEX_general_PL_data.*` and, per complex,
    `{root}/{pdb_id}/{pdb_id}_pocket.pdb` + `{root}/{pdb_id}/{pdb_id}_ligand.sdf`.
    Real PDBbind requires an academic license and is not bundled here --
    see docs/data-and-weights.md and tests/fixtures for a tiny synthetic
    stand-in that exercises this exact code path.
    """

    def __init__(self, root: str | Path, config: FutureAffinityConfig, index_filename: str | None = None) -> None:
        self.root = Path(root)
        self.config = config
        index_path = self.root / "index" / index_filename if index_filename else self._find_index_file()
        self.records = parse_pdbbind_index(index_path)

    def _find_index_file(self) -> Path:
        candidates = sorted((self.root / "index").glob("INDEX_general_PL_data*"))
        if not candidates:
            raise FileNotFoundError(f"No PDBbind index file found under {self.root / 'index'}")
        return candidates[0]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, position: int) -> Example:
        record = self.records[position]
        complex_dir = self.root / record.pdb_id
        protein_types, protein_coords = parse_pdb_ca_trace(complex_dir / f"{record.pdb_id}_pocket.pdb")
        ligand_types, ligand_coords = parse_sdf_ligand(complex_dir / f"{record.pdb_id}_ligand.sdf")

        ligand_types = ligand_types + self.config.residue_vocab_size
        num_protein, num_ligand = protein_types.shape[0], ligand_types.shape[0]

        return Example(
            name=record.pdb_id,
            token_type=torch.cat([protein_types, ligand_types]),
            is_ligand=torch.cat([torch.zeros(num_protein, dtype=torch.bool), torch.ones(num_ligand, dtype=torch.bool)]),
            chain_id=torch.cat([torch.zeros(num_protein, dtype=torch.long), torch.ones(num_ligand, dtype=torch.long)]),
            residue_index=torch.cat(
                [torch.arange(num_protein, dtype=torch.long), torch.arange(num_ligand, dtype=torch.long)]
            ),
            coords=torch.cat([protein_coords, ligand_coords]),
            has_structure=True,
            affinity=record.neg_log_affinity,
            has_affinity=True,
        )
