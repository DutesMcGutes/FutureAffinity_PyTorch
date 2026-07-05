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
python scripts/evaluate.py                 # runs the metric harness (RMSD/lDDT/lDDT-PLI/TM/DockQ + affinity)
python -m futureaffinity.training.train --steps 50 --preset tiny
python -m futureaffinity.training.train --preset base --amp --grad-checkpoint --triangle-chunk 128   # scale knobs
python -m pytest tests/ -q
```

## What's real vs. documented-and-ready

Cheap things are genuinely real and tested; expensive things (full training, real data at scale)
are documented and wired to run, not executed here.

- **Real & tested now:** the Pairformer trunk (triangle multiplicative update + triangle
  attention), EDM diffusion with ensemble sampling, equivariance-by-augmentation (exact
  translation-invariance + SO(3) rotation augmentation, `geometry.py`), the full metric suite
  (`evaluation/metrics.py`: Kabsch RMSD, lDDT, lDDT-PLI, TM-score, DockQ, affinity Pearson/Spearman/
  RMSE), a **real gradient-descent docking optimizer** on an analytical force field
  (`datasources/analytical_docking.py`, not a random mock), scale hooks (gradient checkpointing,
  chunked O(N³) triangle attention verified bit-identical, AMP, correct DDP scaffold), and an
  **overfit-one-complex correctness proof** (`tests/test_overfit.py`: loss collapses, samples reach
  ~1 Å RMSD).
- **Real data paths, opt-in:** `data/rcsb.py` fetches genuine PDB complexes (free, no license),
  `data/ligand_rdkit.py` adds real bonds/3D conformers when RDKit is installed, `data/esm_embeddings.py`
  runs ESM2 on CPU. Vina/OpenMM adapters slot into the same docking interface.
- **Documented, not run:** training the `base`/`large` config on real GPUs, real data curation.
  See `docs/scaling.md` (compute budget), `docs/data-engineering.md` (splits/leakage),
  `docs/evaluation.md` (metric failure modes), and `docs/adr/` (why each design choice).

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
- `src/futureaffinity/geometry.py`: centering, SO(3) rotations, Kabsch superposition
- `src/futureaffinity/evaluation/`: the real metric suite (structure + affinity)
- `src/futureaffinity/data/`: `Example`/`Batch` datatypes, synthetic generator, PDBbind/BindingDB/RCSB/RDKit/ESM loaders
- `src/futureaffinity/datasources/`: the analytical docking optimizer + optional real (Vina/OpenMM) adapters
- `src/futureaffinity/losses/`: masked multi-task loss aggregation
- `src/futureaffinity/training/`: training loop, metrics, and DDP scaffold; `inference/`: the predict() API

## Data and weight policy

No real structure, affinity, or protein-language-model data or weights are bundled here -- see
docs/data-and-weights.md for exactly where to get PDBbind, BindingDB, and ESM2 yourself.

## Sources and inspiration

- AlphaFold3 (Abramson et al. 2024) and the AlphaFold2 supplementary algorithms (triangular
  updates/attention)
- Karras et al. (2022), "Elucidating the Design Space of Diffusion-Based Generative Models" (EDM)
- PDBbind, BindingDB, ESM2
