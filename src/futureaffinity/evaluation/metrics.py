"""Field-standard evaluation metrics for structure and affinity prediction.

These are the yardsticks a co-folding/affinity model is actually judged on:
per-structure RMSD (superposition-aligned), lDDT and its protein-ligand
interface variant lDDT-PLI, TM-score, a DockQ-style complex-quality score,
and the affinity correlation metrics (Pearson/Spearman/RMSE). They're
implemented in pure PyTorch and unit-tested on synthetic inputs here, ready
to run against real predictions once weights and data exist -- see
docs/evaluation.md for definitions and failure modes.
"""
from __future__ import annotations

import torch

from futureaffinity.geometry import center_coords, superpose


def rmsd(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor, superimpose: bool = True) -> torch.Tensor:
    """Root-mean-square deviation over valid tokens. `pred`/`true` (B, N, 3), `mask` (B, N) -> (B,).

    With `superimpose=True` (the default), `pred` is first optimally rigid-body aligned onto
    `true` (Kabsch), giving the pose-invariant RMSD that structure papers report.
    """
    if superimpose:
        aligned = superpose(pred, true, mask)
        target = center_coords(true, mask)
    else:
        aligned, target = pred, true
    sq_dev = ((aligned - target) ** 2).sum(dim=-1)
    mask_f = mask.to(sq_dev.dtype)
    mean_sq = (sq_dev * mask_f).sum(dim=-1) / mask_f.sum(dim=-1).clamp(min=1.0)
    return mean_sq.clamp(min=0).sqrt()


def lddt(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    inclusion_radius: float = 15.0,
    thresholds: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    pair_selector: torch.Tensor | None = None,
) -> torch.Tensor:
    """lDDT: superposition-free fraction of local distances preserved within tolerance. -> (B,).

    `pair_selector` (B, N, N) optionally restricts which (i, j) pairs count -- this is what
    turns lDDT into lDDT-PLI (see `lddt_pli`), by keeping only protein-ligand pairs.
    """
    pred_dist = torch.cdist(pred, pred)
    true_dist = torch.cdist(true, true)

    pair_mask = mask[:, :, None] & mask[:, None, :]
    num_tokens = mask.shape[1]
    not_self = ~torch.eye(num_tokens, dtype=torch.bool, device=mask.device)[None]
    scored = (true_dist < inclusion_radius) & pair_mask & not_self
    if pair_selector is not None:
        scored = scored & pair_selector

    diff = (pred_dist - true_dist).abs()
    preserved = sum((diff < t).to(pred.dtype) for t in thresholds) / len(thresholds)
    scored_f = scored.to(pred.dtype)
    numerator = (preserved * scored_f).sum(dim=(-1, -2))
    denominator = scored_f.sum(dim=(-1, -2)).clamp(min=1.0)
    return (numerator / denominator) * 100.0


def lddt_pli(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    is_ligand: torch.Tensor,
    inclusion_radius: float = 6.0,
) -> torch.Tensor:
    """lDDT-PLI: lDDT restricted to protein-ligand interface pairs. -> (B,).

    The headline metric for pose quality in co-folding: how well the model reproduces the
    protein-ligand contact geometry, ignoring how well it got the protein or ligand alone.
    """
    protein = mask & ~is_ligand
    ligand = mask & is_ligand
    cross = (protein[:, :, None] & ligand[:, None, :]) | (ligand[:, :, None] & protein[:, None, :])
    return lddt(pred, true, mask, inclusion_radius=inclusion_radius, pair_selector=cross)


def tm_score(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """TM-score: length-normalized, superposition-aligned structural similarity in (0, 1]. -> (B,).

    Less sensitive to local errors than RMSD and independent of protein length, via the
    standard d0(L) length normalization.
    """
    aligned = superpose(pred, true, mask)
    target = center_coords(true, mask)
    dist = ((aligned - target) ** 2).sum(dim=-1).clamp(min=0).sqrt()

    lengths = mask.sum(dim=-1).clamp(min=1.0)
    d0 = 1.24 * (lengths - 15.0).clamp(min=0.5) ** (1.0 / 3.0) - 1.8
    d0 = d0.clamp(min=0.5)
    per_token = 1.0 / (1.0 + (dist / d0[:, None]) ** 2)
    mask_f = mask.to(per_token.dtype)
    return (per_token * mask_f).sum(dim=-1) / lengths


def dockq(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    is_ligand: torch.Tensor,
    contact_threshold: float = 5.0,
) -> torch.Tensor:
    """A DockQ-style complex-quality score in [0, 1], combining interface-contact recall (Fnat)
    with interface and ligand RMSD. -> (B,).

    This is a faithful-in-spirit, simplified DockQ: the real metric distinguishes receptor/
    ligand-RMSD with a specific averaging; here we combine Fnat, an interface-RMSD term, and a
    ligand-RMSD term with DockQ's usual scaling constants. See docs/evaluation.md.
    """
    protein = mask & ~is_ligand
    ligand = mask & is_ligand
    cross = protein[:, :, None] & ligand[:, None, :]

    true_dist = torch.cdist(true, true)
    pred_dist = torch.cdist(pred, pred)
    true_contacts = (true_dist < contact_threshold) & cross
    pred_contacts = (pred_dist < contact_threshold) & cross
    shared = (true_contacts & pred_contacts).sum(dim=(-1, -2)).to(pred.dtype)
    total_true = true_contacts.sum(dim=(-1, -2)).clamp(min=1).to(pred.dtype)
    fnat = shared / total_true

    interface_rmsd = rmsd(pred, true, mask, superimpose=True)
    ligand_rmsd = rmsd(pred, true, ligand, superimpose=True)

    scaled_irmsd = 1.0 / (1.0 + (interface_rmsd / 1.5) ** 2)
    scaled_lrmsd = 1.0 / (1.0 + (ligand_rmsd / 5.0) ** 2)
    return (fnat + scaled_irmsd + scaled_lrmsd) / 3.0


def _rank(values: torch.Tensor) -> torch.Tensor:
    order = values.argsort()
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), dtype=values.dtype, device=values.device)
    return ranks


def pearson_correlation(pred: torch.Tensor, true: torch.Tensor) -> float:
    pred, true = pred.flatten().float(), true.flatten().float()
    pred = pred - pred.mean()
    true = true - true.mean()
    denom = (pred.norm() * true.norm()).clamp(min=1e-8)
    return float((pred @ true) / denom)


def spearman_correlation(pred: torch.Tensor, true: torch.Tensor) -> float:
    return pearson_correlation(_rank(pred.flatten().float()), _rank(true.flatten().float()))


def affinity_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    """Pearson, Spearman, RMSE, MAE for a set of predicted vs. measured affinities."""
    pred_f, true_f = pred.flatten().float(), true.flatten().float()
    return {
        "pearson": pearson_correlation(pred_f, true_f),
        "spearman": spearman_correlation(pred_f, true_f),
        "rmse": float(((pred_f - true_f) ** 2).mean().clamp(min=0).sqrt()),
        "mae": float((pred_f - true_f).abs().mean()),
    }
