# ADR 0004 — Multi-task masked loss over mixed-label batches

**Status:** accepted

## Context

The core data problem: affinity labels are scarce (~10^5-10^6) while structures (~10^5-10^6) and
sequences (~10^8) are far more abundant, and most examples have *some* labels but not all (a PDBbind
row has structure + affinity; a BindingDB row has affinity but no structure; a PDB row has structure
but no affinity). Training needs to use every signal each example carries.

## Decision

One model, one shared trunk, **five task heads** (structure/diffusion, confidence, contacts,
affinity, ΔΔG), and a **masked multi-task loss** (`losses/multitask.py`): each `Example` carries a
`has_*` presence flag per task, and each task's loss is averaged only over the rows in the batch
that actually have that label. A batch can freely mix structure-only, affinity-only, and
synthetic-docking rows.

## Alternatives considered

- **Separate models per task:** no cross-task regularization, and the affinity model is starved of
  the abundant structural signal -- exactly the failure mode this project argues against.
- **Require fully-labeled examples:** throws away the vast majority of the data (few rows have all
  labels), reintroducing the scarcity problem.

## Consequences

- Every task regularizes the others through the shared trunk; the abundant structural/contact signal
  supports the scarce affinity signal.
- **Loss weighting matters and is now a hyperparameter** (`FutureAffinityConfig.task_weights`).
  Naive equal weighting lets the high-magnitude/abundant tasks dominate; principled options
  (uncertainty weighting, GradNorm) are a documented tuning step.
- Batches are heterogeneous, so per-task effective batch size varies step to step -- something to
  watch when a task has very few labeled rows in a given batch (the loss is still unbiased, just
  higher-variance). Verified in `tests/test_losses.py`, including the zero-labeled-task case.
