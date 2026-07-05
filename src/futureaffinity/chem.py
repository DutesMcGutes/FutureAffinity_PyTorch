from __future__ import annotations

# Index 0..20 of the combined vocab (see FutureAffinityConfig.residue_vocab_size); indices
# 21-25 are reserved for future special tokens (mask, gap, chain-break, ...).
RESIDUE_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "UNK",
]
RESIDUE_TO_INDEX = {name: index for index, name in enumerate(RESIDUE_ORDER)}
UNKNOWN_RESIDUE_INDEX = RESIDUE_TO_INDEX["UNK"]

# Ligand heavy-atom elements, offset by config.residue_vocab_size in the combined vocab.
LIGAND_ELEMENT_ORDER = ["C", "N", "O", "S", "P", "F", "CL", "BR", "I", "B", "UNK"]
LIGAND_ELEMENT_TO_INDEX = {name: index for index, name in enumerate(LIGAND_ELEMENT_ORDER)}
UNKNOWN_LIGAND_ELEMENT_INDEX = LIGAND_ELEMENT_TO_INDEX["UNK"]


def residue_index(three_letter_code: str) -> int:
    return RESIDUE_TO_INDEX.get(three_letter_code.strip().upper(), UNKNOWN_RESIDUE_INDEX)


def ligand_atom_index(element_symbol: str) -> int:
    return LIGAND_ELEMENT_TO_INDEX.get(element_symbol.strip().upper(), UNKNOWN_LIGAND_ELEMENT_INDEX)


ONE_LETTER_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
    "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
    "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def residue_index_from_one_letter(one_letter_code: str) -> int:
    three_letter = ONE_LETTER_TO_THREE.get(one_letter_code.strip().upper(), "UNK")
    return residue_index(three_letter)
