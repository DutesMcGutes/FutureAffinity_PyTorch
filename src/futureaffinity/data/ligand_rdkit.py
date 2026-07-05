from __future__ import annotations

from dataclasses import dataclass

import torch

from futureaffinity.chem import ligand_atom_index

_INSTALL_HINT = (
    "RDKit ligand featurization requires the 'rdkit' package (pip install rdkit, or the "
    "'physics' extra). It is optional: without it, the bag-of-elements SMILES reader in "
    "data/bindingdb.py is used instead, which reads atom composition but not bonds or 3D geometry."
)


@dataclass
class LigandFeatures:
    """Real ligand featurization: elements, bonds, and (optionally) an embedded 3D conformer.

    This is what the bag-of-elements SMILES reader can't give you -- connectivity and geometry.
    `atom_type_indices` are already offset into the combined vocab (residue + ligand-atom), so
    they can be concatenated straight onto a protein token stream.
    """

    elements: list[str]
    atom_type_indices: torch.Tensor  # (num_atoms,) long, offset by residue_vocab_size elsewhere
    bonds: list[tuple[int, int, float]]  # (i, j, bond_order)
    coords: torch.Tensor | None  # (num_atoms, 3) if a 3D conformer was embedded


def rdkit_available() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        return False


def featurize_smiles(smiles: str, embed_3d: bool = True, seed: int = 0) -> LigandFeatures:
    """Parse a SMILES string into real atom/bond features (and an optional MMFF-optimized conformer)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as error:
        raise RuntimeError(_INSTALL_HINT) from error

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smiles!r}")

    coords = None
    if embed_3d:
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, randomSeed=seed) == 0:
            AllChem.MMFFOptimizeMolecule(mol_h)
            mol = Chem.RemoveHs(mol_h)
            conformer = mol.GetConformer()
            coords = torch.tensor(
                [[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y, conformer.GetAtomPosition(i).z]
                 for i in range(mol.GetNumAtoms())],
                dtype=torch.float32,
            )

    elements = [atom.GetSymbol().upper() for atom in mol.GetAtoms()]
    atom_type_indices = torch.tensor([ligand_atom_index(element) for element in elements], dtype=torch.long)
    bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), float(bond.GetBondTypeAsDouble()))
        for bond in mol.GetBonds()
    ]
    return LigandFeatures(elements=elements, atom_type_indices=atom_type_indices, bonds=bonds, coords=coords)
