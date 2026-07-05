# FutureAffinity (PyTorch)

A pure PyTorch, multi-task co-folding + affinity foundation model: protein(+ligand) -> structure
ensemble, per-token confidence, binding affinity, ΔΔG for point mutations, and uncertainty --
all from one shared trunk.

This is the "model" repo, analogous to `Alphafold3_PyTorch`. The paired teaching repo,
`FutureAffinity_Repro`, walks through the same ideas from scratch with exercises and reference
solutions.

## What this is (and isn't)

Real architecture, small scale: a genuine AlphaFold2/3-style Pairformer trunk (triangular
multiplicative updates + triangular attention, not an MLP stand-in) and a real EDM-style
diffusion module, sized so the whole thing trains and runs on CPU with synthetic data in seconds.
The same code (`FutureAffinityConfig.base()`) is the real-scale configuration; running it against
real downloaded data on real GPU hardware is a follow-up step, not something this repo does for
you. See docs/limitations.md and docs/roadmap.md before drawing conclusions from any output.

## Why multi-task

High-quality affinity measurements are scarce (10^5-10^6) next to available structures and
sequences (10^7-10^8+). Rather than training one affinity predictor, every head here
(structure, confidence, contacts, affinity, ΔΔG) shares one trunk and one training loop, and
`losses/multitask.py` masks each task's loss by whether a given example actually has that label --
so a batch can freely mix a structure-only PDBbind row, a structure-less BindingDB row (sequence +
SMILES + Kd/Ki/IC50 only), and a synthetic-docking row, and every task only learns from the rows
that actually carry its label.

## Install

```bash
pip install -e .
# optional extras:
pip install -e ".[esm]"       # ESM2 protein-language-model embeddings
pip install -e ".[physics]"   # AutoDock Vina / OpenMM integration
pip install -e ".[dev]"       # pytest, ruff
```

## Quickstart

```bash
python scripts/run_demo.py                 # trains briefly on synthetic data, then predicts
python -m futureaffinity.training.train --steps 50 --preset tiny
python -m pytest tests/ -q
```

```python
from futureaffinity.config import FutureAffinityConfig
from futureaffinity.model.model import FutureAffinityModel
from futureaffinity.inference.predict import predict

model = FutureAffinityModel(FutureAffinityConfig.tiny())
result = predict(model, protein_sequence="MKTAYIAKQ...", ligand_smiles="CC(=O)Oc1ccccc1C(=O)O")
# result: structure_ensemble, plddt, contact_probabilities, structural_uncertainty,
#         affinity_mean, affinity_std
```

## Layout

See docs/architecture.md for the full data-flow diagram. Briefly:

- `src/futureaffinity/model/`: embedding, Pairformer trunk, diffusion, and all five task heads
- `src/futureaffinity/data/`: `Example`/`Batch` datatypes, synthetic data generator, PDBbind/BindingDB/ESM loaders
- `src/futureaffinity/datasources/`: mock (dependency-free) and optional real (Vina/OpenMM) physics adapters
- `src/futureaffinity/losses/`: masked multi-task loss aggregation
- `src/futureaffinity/training/`, `src/futureaffinity/inference/`: the training loop and the predict() API

## Data and weight policy

No real structure, affinity, or protein-language-model data or weights are bundled here -- see
docs/data-and-weights.md for exactly where to get PDBbind, BindingDB, and ESM2 yourself.

## Sources and inspiration

- AlphaFold3 (Abramson et al. 2024) and the AlphaFold2 supplementary algorithms (triangular
  updates/attention)
- Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM)
- PDBbind, BindingDB, ESM2
