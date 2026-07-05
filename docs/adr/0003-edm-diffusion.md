# ADR 0003 — EDM-style diffusion for structure generation

**Status:** accepted

## Context

The structure module must turn trunk representations into 3D coordinates, and ideally into a
*distribution* over structures (for ensembles/uncertainty), not a single point estimate. Options:
regress coordinates directly (AF2-style structure module), or generate them with a diffusion/flow
model (AF3-style).

## Decision

Use a **diffusion** structure module with the **EDM parameterization** (Karras et al. 2022):
- Karras noise schedule (`karras_sigma_schedule`) and preconditioning (`edm_preconditioning`:
  `c_skip`, `c_out`, `c_in`, `c_noise`), which is the numerically well-behaved,
  hyperparameter-light formulation of diffusion.
- Denoising-score-matching training objective (`DiffusionModule.training_loss`), EDM-weighted.
- Ancestral sampling with multiple independent seeds (`sample_ensemble`) to produce a structural
  ensemble.

## Alternatives considered

- **Direct coordinate regression (AF2 structure module):** simpler, deterministic, but gives one
  structure and no natural uncertainty. Getting an ensemble requires extra machinery.
- **Flow matching:** a strong, arguably simpler alternative to diffusion with similar properties;
  a reasonable future swap. EDM diffusion was chosen for its maturity and direct AF3 lineage.

## Consequences

- The ensemble is first-class: uncertainty (`heads/uncertainty.py`) falls out of ensemble spread,
  and the affinity head can be evaluated per-conformation (energy over a distribution, not a point).
- **Training subtlety:** the denoiser is trained directly via the score-matching loss; we do *not*
  backprop through the iterative sampling loop (`sample_ensemble` is `@torch.no_grad()`). The heads
  that need a "current structure" use a cheap partial rollout as a non-differentiable input. This
  matches standard diffusion training and is documented in `model/model.py`.
- Sampling is iterative (N steps), so inference is more expensive than a single-pass regressor --
  the ensemble/uncertainty benefit is paid for in sampling FLOPs.
- The overfit-one-complex test (`tests/test_overfit.py`) is the correctness proof that the whole
  diffusion + sampling + geometry stack is wired right: loss collapses and samples reach ~1 Å RMSD.
