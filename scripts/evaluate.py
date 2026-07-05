"""Run the evaluation harness end to end on synthetic ground-truth complexes.

Demonstrates that the metric pipeline (docs/evaluation.md) works against real predictions: it
samples structures from the model and scores them with RMSD / lDDT / lDDT-PLI / TM-score / DockQ,
and scores the affinity head with Pearson/Spearman/RMSE. On an untrained model the numbers are
poor by construction -- the point is that the harness is real and ready for a trained checkpoint.
"""
from __future__ import annotations

import argparse

import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.synthetic import make_synthetic_batch
from futureaffinity.evaluation.metrics import affinity_metrics, dockq, lddt, lddt_pli, rmsd, tm_score
from futureaffinity.model.model import FutureAffinityModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional trained state_dict to load.")
    parser.add_argument("--num-complexes", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    model = FutureAffinityModel(config)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        print(f"loaded checkpoint {args.checkpoint}")
    model.eval()

    batch = make_synthetic_batch(config, batch_size=args.num_complexes, protein_length=16, ligand_size=5, seed=1)
    with torch.no_grad():
        token, pair = model.encode(batch)
        ensemble = model.diffusion.sample_ensemble(token, pair, batch.token_mask, num_samples=args.samples, num_steps=20)
        affinity_pred = model.affinity_head(token, pair, ensemble[:, 0], batch.is_ligand, batch.token_mask)

    # best-of-ensemble structure metrics (report the best sample per complex, standard practice)
    def best(metric_fn, higher_is_better: bool):
        per_sample = torch.stack(
            [metric_fn(ensemble[:, s]) for s in range(ensemble.shape[1])], dim=1
        )  # (B, S)
        return (per_sample.max(dim=1).values if higher_is_better else per_sample.min(dim=1).values).mean().item()

    print("structure metrics (best-of-ensemble, mean over complexes):")
    print(f"  RMSD      : {best(lambda c: rmsd(c, batch.coords, batch.token_mask), False):.3f} A")
    print(f"  lDDT      : {best(lambda c: lddt(c, batch.coords, batch.token_mask), True):.2f}")
    print(f"  lDDT-PLI  : {best(lambda c: lddt_pli(c, batch.coords, batch.token_mask, batch.is_ligand), True):.2f}")
    print(f"  TM-score  : {best(lambda c: tm_score(c, batch.coords, batch.token_mask), True):.3f}")
    print(f"  DockQ     : {best(lambda c: dockq(c, batch.coords, batch.token_mask, batch.is_ligand), True):.3f}")

    labeled = batch.has_affinity
    if labeled.any():
        metrics = affinity_metrics(affinity_pred[labeled], batch.affinity[labeled])
        print("affinity metrics:")
        for name, value in metrics.items():
            print(f"  {name:8s}: {value:.3f}")

    print("\n(Untrained model -> poor numbers by construction; the harness is what's being shown.)")


if __name__ == "__main__":
    main()
