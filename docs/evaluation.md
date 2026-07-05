# Evaluation: metrics and their failure modes

The metrics in `evaluation/metrics.py` are the yardsticks a co-folding/affinity model is judged on.
All are implemented in pure PyTorch and unit-tested on synthetic inputs (`tests/test_metrics.py`),
ready to run against real predictions. Knowing *when each metric lies* matters as much as computing
it.

## Structure metrics

- **RMSD (Kabsch-superposed)** — root-mean-square atom deviation after optimal rigid alignment.
  Intuitive, but dominated by the worst-placed atoms and undefined across different-length
  structures. A single flipped loop can wreck an otherwise-correct RMSD.
- **lDDT** — superposition-free: the fraction of local interatomic distances preserved within
  tolerance. Robust to domain motions (no global alignment to be fooled by) and what pLDDT
  confidence is trained to predict. Can look high even when the global fold is wrong if local
  geometry is right.
- **lDDT-PLI** — lDDT restricted to protein-ligand interface pairs. The headline pose-quality
  metric for co-folding: it measures whether the *interaction* geometry is right, ignoring how well
  the protein or ligand was modeled alone. This is usually the number that matters for drug design.
- **TM-score** — length-normalized global similarity in (0, 1], with the standard d0(L)
  normalization. Less sensitive to local errors than RMSD; >0.5 conventionally means "same fold."
- **DockQ** — a single [0, 1] complex-quality score combining interface-contact recall (Fnat) with
  interface- and ligand-RMSD. The community standard for ranking docked/predicted complexes.
  (Ours is a faithful-in-spirit simplification -- see the docstring.)

The trap across all structure metrics: **report them on a leakage-controlled test split**
(docs/data-engineering.md), per split type. A great RMSD on a random split and a poor one on a
scaffold/time split means memorization.

## Affinity metrics

`affinity_metrics()` returns all four, because they answer different questions:

- **Pearson r** — linear correlation of predicted vs. measured affinity. Sensitive to getting the
  *scale* right.
- **Spearman ρ** — rank correlation. Often the metric that matters most in practice: for
  hit-ranking / lead prioritization you care about ordering compounds correctly, not absolute Kd.
  A model can have modest Pearson but excellent Spearman and still be useful.
- **RMSE / MAE** — absolute error in affinity units. What you need if you're predicting a number to
  act on (e.g. "is this sub-micromolar?"), not just ranking.

Failure modes to watch: a model that predicts the *mean* affinity for everything scores decent RMSE
but ~zero correlation; a model tuned on a congeneric series can have high within-series Spearman and
fail completely across series (again, the scaffold-split point).

## Confidence calibration

The confidence head predicts pLDDT and a distance-error proxy. The evaluation that matters is
**calibration**: does predicted confidence actually correlate with realized accuracy? A reliability
plot (predicted pLDDT bin vs. mean true lDDT in that bin) should lie near the diagonal. An
overconfident model is worse than a less accurate but well-calibrated one, because downstream users
act on the confidence.

## Uncertainty

`heads/uncertainty.py` derives uncertainty from the spread of the diffusion ensemble. The check:
ensemble spread should be *larger* on examples the model gets wrong (high error) than on ones it
gets right -- i.e. uncertainty should be predictive of error. That's what makes it usable for the
active-learning loop (pick high-uncertainty complexes to label next).
