# Data and Weights

FutureAffinity does not include any real structure, affinity, or protein-language-model weights or
datasets. Nothing here downloads, mirrors, or redistributes licensed or bulk data automatically.

## PDBbind

Structure + affinity supervision (`data/pdbbind.py`). Requires a free academic registration at
http://www.pdbbind.org.cn -- get it there, not from this repo. `scripts/download_pdbbind_sample.py`
only checks that a local extraction is laid out the way the loader expects; it does not fetch
anything. `tests/fixtures/pdbbind/` is a tiny hand-written, clearly-synthetic stand-in (fake PDB
id `9xyz`) used only to exercise the parsing code in tests.

## BindingDB

Sequence + SMILES + affinity-only supervision, no structure (`data/bindingdb.py`). BindingDB
publishes a bulk TSV export at https://www.bindingdb.org/bind/downloads.jsp with no login
required, but it's large (multi-GB) -- download and filter it yourself; this repo doesn't fetch
it. `tests/fixtures/bindingdb_sample.tsv` is a tiny hand-written fixture with the real column
names, used only for tests.

## ESM2 embeddings

Optional protein-language-model features (`data/esm_embeddings.py`, `FutureAffinityConfig.use_esm_embeddings`).
Uses Meta's `fair-esm` package and downloads the chosen checkpoint via `torch.hub` on first use --
that's Meta's model and Meta's license, not redistributed here. Install with
`pip install -e '.[esm]'` if you want this.

## AutoDock Vina / OpenMM

Optional real physics integration (`datasources/vina_adapter.py`, `datasources/openmm_adapter.py`).
Both are guarded imports: importing the module never fails, but calling their methods raises a
clear `RuntimeError` with install instructions if the underlying tool isn't installed. Neither is
required to train or run the core model -- `datasources/mock_docking.py` is the always-available,
dependency-free fallback used for synthetic pretraining signal.

## Model weights

No trained FutureAffinity weights are shipped. `scripts/run_demo.py` trains a `tiny`-config model from
scratch on synthetic data for 30 steps purely to prove the pipeline runs end to end -- see
docs/limitations.md before reading anything into its output.
