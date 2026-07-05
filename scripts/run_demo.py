"""End-to-end demo: train briefly on synthetic data, then predict on one example.

Runs entirely on CPU with the `tiny` config and no external downloads --
the same code path `--preset base` would run at real scale.
"""
from __future__ import annotations

import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.inference.predict import predict
from futureaffinity.model.model import FutureAffinityModel
from futureaffinity.training.train import SyntheticSource, train

DEMO_PROTEIN = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSR"
DEMO_LIGAND_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin


def main() -> None:
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()

    print("Training FutureAffinityModel (tiny config) on synthetic data for 30 steps...")
    model = train(
        config,
        sources=[SyntheticSource(config, protein_length=20, ligand_size=6)],
        num_steps=30,
        batch_size=4,
        lr=1e-3,
        log_every=10,
        seed=0,
    )

    model.eval()
    print("\nRunning inference on a real protein sequence + ligand SMILES...")
    result = predict(model, protein_sequence=DEMO_PROTEIN, ligand_smiles=DEMO_LIGAND_SMILES, num_samples=5, num_steps=10)

    print(f"tokens: {result['plddt'].shape[0]}")
    print(f"mean predicted lDDT: {result['plddt'].mean().item():.2f}")
    print(f"predicted affinity (arbitrary units, untrained): {result['affinity_mean']:.3f} +/- {result['affinity_std']:.3f}")
    print(f"mean structural uncertainty (RMSF-style, A): {result['structural_uncertainty'].mean().item():.3f}")
    print(
        "\nReminder: this model is untrained beyond the 30-step synthetic smoke test above -- "
        "these numbers demonstrate the pipeline runs, not that they're accurate. See docs/limitations.md."
    )


if __name__ == "__main__":
    main()
