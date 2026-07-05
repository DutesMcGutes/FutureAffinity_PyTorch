# Data engineering: splits, leakage, and weak supervision

Data is where co-folding/affinity models are won or lost, not architecture. This document is the
plan for turning raw public data into training/validation/test sets that measure *generalization*
rather than memorization. None of it is run at scale here; the loaders (`data/pdbbind.py`,
`data/bindingdb.py`, `data/rcsb.py`) parse the real formats, and this is the pipeline they'd feed.

## The sources and what each contributes

| Source | Signal | Label density | Role |
| --- | --- | --- | --- |
| PDB / RCSB | 3D structures | ~200k complexes | structure + contacts + confidence pretraining |
| AlphaFold DB | predicted structures | ~200 M | optional structural distillation targets |
| PDBbind | structure + affinity | ~20k | the affinity gold set (has both) |
| BindingDB | sequence + SMILES + affinity | ~1-2 M usable | affinity supervision *without* structure |
| Deep mutational scans | ΔΔG | ~10^5-10^6 | mutation-effect fine-tuning |
| Synthetic docking (this repo) | pose + energy | unbounded | cheap physics-shaped pretraining signal |

The whole thesis of the project is that affinity labels are scarce (~10^5-10^6) but the *other*
signals are 100-1000x larger, so the masked multi-task loss (`losses/multitask.py`) lets every row
contribute to whatever it has a label for. The data pipeline's job is to assemble those rows
without letting the test set leak into training.

## Leakage is the whole game

Random splits **massively overestimate** performance in this field, because near-duplicate
proteins/ligands end up on both sides. The three leakage axes and how to cut each:

1. **Sequence similarity.** Cluster all protein chains with **MMseqs2** at, e.g., 30% identity;
   assign whole clusters (not individual chains) to a single split. A test protein must have no
   >30%-identity homolog in training.
2. **Ligand/scaffold similarity.** Cluster ligands by Bemis-Murcko scaffold or ECFP Tanimoto;
   keep similar scaffolds out of the test split (a "scaffold split"), so you measure generalization
   to new chemistry, not interpolation within a congeneric series.
3. **Temporal.** For a realistic "will this work on tomorrow's targets" estimate, use a **time
   split**: train on structures deposited before a cutoff date, test on those after (this is how
   CASP/CAMEO evaluate, and PDBbind provides deposition years -- see `data/pdbbind.py`).

The strongest evaluation combines them: a test set that is simultaneously sequence-, scaffold-,
and time-disjoint from training. Report metrics on each split separately -- a model that's great on
a random split and poor on a scaffold split is memorizing.

## Deduplication and curation

- **Exact + near-duplicate removal**: identical sequences, identical (protein, ligand) pairs.
- **Affinity harmonization**: BindingDB mixes Kd/Ki/IC50/EC50 across assays and labs. Prefer direct
  binding (Kd/Ki) over functional (IC50/EC50); convert to a common pKd/pKi scale (already done in
  `data/bindingdb.py`); flag or down-weight rows with wide assay disagreement.
- **Quality filters**: resolution cutoffs for crystal structures, sane affinity ranges, remove
  covalent binders / metals if out of scope.
- **Crop selection**: for the `O(N^3)` reasons in docs/scaling.md, training crops should be centered
  on the binding interface, not random spans, so the model spends its token budget where the
  supervision is.

## Weak / synthetic supervision

`datasources/analytical_docking.py` (and, when installed, the Vina/OpenMM adapters) generate
pose+energy pairs at arbitrary scale. Used as a *pretraining* signal -- teach the energy head
approximate physics on billions of cheap noisy labels, then fine-tune on the ~10^5 real affinities.
The risk to manage: synthetic labels have systematic bias (the toy force field is not real
chemistry), so they must be a pretraining prior that real data corrects, never mixed in at equal
weight with real affinities at fine-tuning time.

## Splits as code

A `make_splits` step (not yet implemented -- a documented next task) would: run MMseqs2 clustering,
compute scaffold clusters, read deposition dates, then emit train/val/test index files that the
`Dataset` classes read. Keeping splits as versioned artifacts (not recomputed on the fly) is what
makes results reproducible and leakage auditable.
