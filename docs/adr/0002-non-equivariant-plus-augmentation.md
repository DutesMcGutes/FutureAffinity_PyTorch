# ADR 0002 — Non-equivariant structure module + augmentation

**Status:** accepted

## Context

A structure predictor's output must not depend on the arbitrary global position/orientation of the
input frame: rotating the inputs should rotate the outputs, and translating them should translate
the outputs. There are two ways to get this: build it into the architecture (SE(3)-equivariant
networks like tensor-field / EGNN layers), or instill it through data (center + rotation
augmentation on a plain transformer).

## Decision

Use a **plain, non-equivariant** transformer denoiser, and get roto-translation robustness from
data:

- **Translation:** every target is mean-centered before the loss (`geometry.center_coords`), so
  absolute position carries no signal. This makes the objective *exactly* translation-invariant
  (verified in `tests/test_equivariance.py`).
- **Rotation:** at train time each target is randomly rotated (uniform over SO(3),
  `geometry.random_rotation_matrices`) before noising. Over training the denoiser learns to handle
  all orientations rather than a canonical frame.

This is the same choice AlphaFold3 makes (it dropped AF2's equivariant frames for a non-equivariant
diffusion module + augmentation).

## Alternatives considered

- **Built-in SE(3)-equivariant layers:** exactly equivariant by construction, no augmentation
  needed, better sample efficiency in principle. But more complex, historically slower, and harder
  to scale -- and AF3's result is strong evidence that augmentation is sufficient at scale.

## Consequences

- Simpler, faster, more scalable layers; the trunk is standard attention.
- Equivariance is *approximate and learned*, not guaranteed -- it holds in expectation over
  augmentation, and better with more training. A short sequence / undertrained model can show
  orientation sensitivity.
- Augmentation is a real train-time cost multiplier on the effective dataset (the model must see
  each structure in many orientations), which the compute budget (docs/scaling.md) accounts for.
- **Testable claims we actually assert:** translation-invariance is exact; augmentation draws are
  proper SO(3) rotations and genuinely vary orientation.
