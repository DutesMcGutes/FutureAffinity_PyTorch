from pathlib import Path

import pytest
import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.bindingdb import BindingDBDataset, parse_smiles_elements
from futureaffinity.data.datatypes import collate
from futureaffinity.data.pdbbind import PDBbindDataset, parse_pdb_ca_trace, parse_pdbbind_index, parse_sdf_ligand
from futureaffinity.data.synthetic import make_synthetic_batch, make_synthetic_example

FIXTURES = Path(__file__).parent / "fixtures"


def test_synthetic_example_shapes_are_consistent():
    config = FutureAffinityConfig.tiny()
    example = make_synthetic_example(config, protein_length=10, ligand_size=4, include_ddg=True, seed=0)

    assert example.num_tokens == 14
    assert example.coords.shape == (14, 3)
    assert example.contacts.shape == (14, 14)
    assert example.is_ligand[:10].sum() == 0
    assert example.is_ligand[10:].sum() == 4
    assert example.has_affinity
    assert example.has_docking
    assert example.has_ddg


def test_collate_pads_variable_length_examples_and_masks_correctly():
    config = FutureAffinityConfig.tiny()
    short = make_synthetic_example(config, protein_length=5, ligand_size=2, name="short", seed=1)
    long = make_synthetic_example(config, protein_length=8, ligand_size=3, name="long", seed=2)

    batch = collate([short, long], config)

    assert batch.batch_size == 2
    assert batch.num_tokens == long.num_tokens
    assert batch.token_mask[0].sum().item() == short.num_tokens
    assert batch.token_mask[1].sum().item() == long.num_tokens
    assert not batch.token_mask[0, short.num_tokens :].any()


def test_make_synthetic_batch_end_to_end():
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=5, protein_length=10, ligand_size=3, seed=3)
    assert batch.batch_size == 5
    assert batch.has_affinity.any()
    assert batch.has_ddg.any()


def test_parse_pdbbind_index_fixture():
    records = parse_pdbbind_index(FIXTURES / "pdbbind" / "index" / "INDEX_general_PL_data.2020")
    assert len(records) == 1
    record = records[0]
    assert record.pdb_id == "9xyz"
    assert record.year == 2020
    assert record.neg_log_affinity == pytest.approx(7.50)
    assert record.ligand_code == "LIG"


def test_parse_pdb_ca_trace_fixture():
    token_types, coords = parse_pdb_ca_trace(FIXTURES / "pdbbind" / "9xyz" / "9xyz_pocket.pdb")
    assert token_types.shape == (4,)
    assert coords.shape == (4, 3)


def test_parse_sdf_ligand_fixture():
    token_types, coords = parse_sdf_ligand(FIXTURES / "pdbbind" / "9xyz" / "9xyz_ligand.sdf")
    assert token_types.shape == (3,)
    assert coords.shape == (3, 3)


def test_pdbbind_dataset_builds_a_full_example():
    config = FutureAffinityConfig.tiny()
    dataset = PDBbindDataset(FIXTURES / "pdbbind", config)
    assert len(dataset) == 1

    example = dataset[0]
    assert example.name == "9xyz"
    assert example.has_structure
    assert example.has_affinity
    assert example.affinity == pytest.approx(7.50)
    assert example.num_tokens == 4 + 3
    assert example.is_ligand[:4].sum() == 0
    assert example.is_ligand[4:].sum() == 3


def test_parse_smiles_elements_handles_bracket_and_organic_subset_atoms():
    elements = parse_smiles_elements("CC(=O)Oc1ccccc1C(=O)O")
    assert elements.count("C") >= 5
    assert "O" in elements

    bracket_elements = parse_smiles_elements("[Na+].[Cl-]")
    assert "NA" in bracket_elements
    assert "CL" in bracket_elements


def test_bindingdb_dataset_parses_fixture_and_prefers_direct_measurements():
    config = FutureAffinityConfig.tiny()
    dataset = BindingDBDataset(FIXTURES / "bindingdb_sample.tsv", config)
    assert len(dataset) == 3

    example0 = dataset[0]
    assert not example0.has_structure
    assert example0.has_affinity
    assert example0.affinity == pytest.approx(9.0 - torch.log10(torch.tensor(12.5)).item())

    example2 = dataset[2]
    # Ki="<1" should win over IC50=100.0 and parse to 1.0 nM -> pAffinity 9.0
    assert example2.affinity == pytest.approx(9.0, abs=1e-6)
