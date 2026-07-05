"""Overfit-one-complex correctness proof.

The most convincing single check that the whole stack (embedding -> trunk -> diffusion ->
sampling -> geometry) is wired correctly: a small model, trained on ONE fixed complex, should
drive the denoising loss down and sample structures that land close to that target. If any of
the coordinate plumbing, masking, or gradient flow were wrong, this could not happen. This is
the test you point an interviewer at to de-risk a real training run.
"""
import dataclasses

import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.synthetic import make_synthetic_batch
from futureaffinity.evaluation.metrics import rmsd
from futureaffinity.model.model import FutureAffinityModel


def test_model_can_overfit_a_single_complex():
    torch.manual_seed(0)
    # augmentation off: this is a memorization/correctness probe on one fixed target, not a
    # generalization test, so we don't want a fresh random orientation every step.
    config = dataclasses.replace(FutureAffinityConfig.tiny(), augment_rotation=False)
    batch = make_synthetic_batch(config, batch_size=1, protein_length=10, ligand_size=4, seed=3)
    model = FutureAffinityModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)

    def diffusion_loss() -> torch.Tensor:
        token, pair = model.encode(batch)
        return model.diffusion.training_loss(
            batch.coords, token, pair, batch.token_mask, generator=generator
        ).mean()

    initial = float(diffusion_loss())
    for _ in range(200):
        optimizer.zero_grad()
        loss = diffusion_loss()
        loss.backward()
        optimizer.step()
    final = float(diffusion_loss())

    # loss should collapse to a small fraction of where it started
    assert final < 0.4 * initial, f"overfit failed: {initial:.3f} -> {final:.3f}"

    # and sampled structures should land near the memorized target
    model.eval()
    with torch.no_grad():
        token, pair = model.encode(batch)
        ensemble = model.diffusion.sample_ensemble(
            token, pair, batch.token_mask, num_samples=4, num_steps=20, generator=torch.Generator().manual_seed(1)
        )
    best_rmsd = min(float(rmsd(ensemble[:, s], batch.coords, batch.token_mask)[0]) for s in range(ensemble.shape[1]))
    assert best_rmsd < 3.0, f"best sampled RMSD to target too high: {best_rmsd:.2f} A"
