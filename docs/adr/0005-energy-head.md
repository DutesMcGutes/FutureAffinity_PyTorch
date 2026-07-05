# ADR 0005 — Learned energy head, not direct Kd regression

**Status:** accepted

## Context

The affinity head has to predict binding strength. The obvious approach is to regress the measured
number (pKd/pKi) directly from a pooled representation of the complex. The project's thesis is that
this wastes the structure: affinity is a property of a *conformation*, and there's far more
structural signal than affinity signal.

## Decision

Frame the head as a **learned energy over a conformation**, `E(protein, ligand, conformation)`
(`heads/affinity.py`): it pools the interface region of the pair representation *and* a physical
feature (nearest protein-ligand distance) computed from a specific sampled structure. Because the
diffusion module produces an ensemble, the head is evaluated per-conformation, so each affinity
measurement becomes one observation constraining a shared energy landscape rather than a lookup
keyed on identity.

## Alternatives considered

- **Direct pooled regression of Kd:** simplest, but conformation-blind -- it can't distinguish a
  bound pose from a clash, and it can't benefit from the ensemble or from synthetic docking energies
  that come as (pose, energy) pairs.
- **Pure physics (score a docked pose with a force field):** interpretable but caps out at the
  force field's accuracy and doesn't learn from data.

## Consequences

- The head consumes the ensemble naturally: affinity mean and its uncertainty come from scoring each
  ensemble member (`inference/predict.py`), and structural uncertainty and affinity uncertainty are
  linked.
- Synthetic docking supervision (`datasources/analytical_docking.py`) fits the same shape --
  (conformation, energy) pairs -- so the head can be pretrained on cheap physics-shaped labels and
  fine-tuned on scarce real affinities (docs/data-engineering.md).
- The ΔΔG head reuses this framing on a wildtype/mutant token difference (`heads/ddg.py`), rather
  than being an independent regressor.
- **Cost:** evaluating per-conformation means sampling structures at inference, which is more
  expensive than a single pooled forward pass -- the accuracy/uncertainty benefit is paid for in
  sampling FLOPs (see ADR 0003).
