from pathlib import Path

import pytest
import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.ligand_rdkit import featurize_smiles, rdkit_available
from futureaffinity.data.rcsb import parse_pdb_complex

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_real_format_pdb_complex_selects_protein_and_ligand_skipping_water():
    config = FutureAffinityConfig.tiny()
    pdb_text = (FIXTURES / "rcsb_sample.pdb").read_text(encoding="utf-8")
    example = parse_pdb_complex(pdb_text, config)

    assert example.has_structure
    assert example.name == "rcsb_LIG"
    # 5 protein residues (CA only) + 3 ligand atoms; the HOH water is skipped
    assert int((~example.is_ligand).sum()) == 5
    assert int(example.is_ligand.sum()) == 3
    assert example.coords.shape == (8, 3)
    # ligand token types are offset into the ligand block of the combined vocab
    assert (example.token_type[example.is_ligand] >= config.residue_vocab_size).all()


def test_parse_pdb_complex_can_target_a_named_ligand():
    config = FutureAffinityConfig.tiny()
    pdb_text = (FIXTURES / "rcsb_sample.pdb").read_text(encoding="utf-8")
    example = parse_pdb_complex(pdb_text, config, ligand_code="LIG")
    assert int(example.is_ligand.sum()) == 3


def test_parse_pdb_complex_errors_clearly_for_absent_ligand():
    config = FutureAffinityConfig.tiny()
    pdb_text = (FIXTURES / "rcsb_sample.pdb").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="ligand"):
        parse_pdb_complex(pdb_text, config, ligand_code="ZZZ")


@pytest.mark.skipif(not rdkit_available(), reason="rdkit not installed (optional dependency)")
def test_rdkit_featurization_gives_bonds_and_geometry():
    features = featurize_smiles("CC(=O)Oc1ccccc1C(=O)O", embed_3d=True)  # aspirin
    assert len(features.elements) == features.atom_type_indices.shape[0]
    assert len(features.bonds) > 0  # bag-of-elements can't give this; RDKit can
    if features.coords is not None:
        assert features.coords.shape == (len(features.elements), 3)


def test_rdkit_featurizer_raises_clear_error_when_unavailable():
    if rdkit_available():
        pytest.skip("rdkit is installed; the missing-dependency guard cannot be exercised")
    with pytest.raises(RuntimeError, match="rdkit"):
        featurize_smiles("CCO")


def test_esm_guard_reports_missing_dependency_clearly():
    from futureaffinity.data.esm_embeddings import compute_esm_embeddings

    try:
        import esm  # noqa: F401

        pytest.skip("fair-esm is installed; the missing-dependency guard cannot be exercised")
    except ImportError:
        with pytest.raises(RuntimeError, match="fair-esm"):
            compute_esm_embeddings(["MKT"])
