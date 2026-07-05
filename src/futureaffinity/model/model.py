from __future__ import annotations

import torch
from torch import nn

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.datatypes import Batch
from futureaffinity.model.diffusion import DiffusionModule
from futureaffinity.model.embedding import InputEmbedder
from futureaffinity.model.heads.affinity import AffinityHead
from futureaffinity.model.heads.confidence import ConfidenceHead
from futureaffinity.model.heads.contacts import ContactHead
from futureaffinity.model.heads.ddg import DDGHead
from futureaffinity.model.pairformer import PairformerTrunk


class FutureAffinityModel(nn.Module):
    """Wires embedding -> Pairformer trunk -> diffusion -> every task head.

    This is the "multi-task scaffold" milestone: every head is real and
    shape-correct and rides on the same trunk representation, but none of
    them are pretrained. `compute_losses` is the entry point used by
    `training/train.py`; it returns per-example, per-task losses so the
    caller (via `losses/multitask.py`) can mask out tasks that a given
    example has no label for.
    """

    def __init__(self, config: FutureAffinityConfig | None = None) -> None:
        super().__init__()
        self.config = config or FutureAffinityConfig()
        self.embedder = InputEmbedder(self.config)
        self.trunk = PairformerTrunk(self.config)
        self.diffusion = DiffusionModule(self.config)
        self.confidence_head = ConfidenceHead(self.config)
        self.contact_head = ContactHead(self.config)
        self.affinity_head = AffinityHead(self.config)
        self.ddg_head = DDGHead(self.config)

    def encode(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        token, pair = self.embedder(batch)
        token, pair = self.trunk(token, pair, batch.token_mask, batch.pair_mask)
        return token, pair

    def _rollout_structure(
        self, token: torch.Tensor, pair: torch.Tensor, token_mask: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        # `sample_ensemble` is decorated @torch.no_grad(): we don't backprop through the
        # iterative reverse-diffusion loop (that's not how EDM-style models are trained --
        # the denoiser is trained directly via `training_loss` instead). Confidence/affinity/
        # ddG heads still get gradient through their trunk-derived (token/pair) inputs; only
        # the coordinate-dependent geometric features computed from this specific rollout are
        # non-differentiable, matching common practice for structure-module rollouts elsewhere.
        ensemble = self.diffusion.sample_ensemble(
            token,
            pair,
            token_mask,
            num_samples=1,
            num_steps=self.config.num_train_rollout_steps,
            generator=generator,
        )
        return ensemble[:, 0]

    def forward(self, batch: Batch, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Inference-style forward pass: one rollout structure + every head's prediction."""
        token, pair = self.encode(batch)
        rollout_coords = self._rollout_structure(token, pair, batch.token_mask, generator)

        return {
            "token": token,
            "pair": pair,
            "rollout_coords": rollout_coords,
            "contact_logits": self.contact_head(pair),
            "confidence": self.confidence_head(token, pair),
            "affinity": self.affinity_head(token, pair, rollout_coords, batch.is_ligand, batch.token_mask),
            "ddg": self.ddg_head(token, batch.token_type, batch.mutant_token_type, batch.is_ligand, batch.token_mask),
        }

    def compute_losses(
        self, batch: Batch, generator: torch.Generator | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Per-example, per-task losses (unmasked by task presence -- see losses/multitask.py)."""
        token, pair = self.encode(batch)
        rollout_coords = self._rollout_structure(token, pair, batch.token_mask, generator)

        diffusion_loss = self.diffusion.training_loss(batch.coords, token, pair, batch.token_mask, generator=generator)

        confidence_outputs = self.confidence_head(token, pair)
        confidence_loss = self.confidence_head.loss(
            confidence_outputs, rollout_coords, batch.coords, batch.token_mask, batch.pair_mask
        )

        contact_logits = self.contact_head(pair)
        contact_loss = self.contact_head.loss(contact_logits, batch.contacts, batch.pair_mask)

        affinity_pred = self.affinity_head(token, pair, rollout_coords, batch.is_ligand, batch.token_mask)
        affinity_loss = self.affinity_head.loss(affinity_pred, batch.affinity)

        ddg_pred = self.ddg_head(token, batch.token_type, batch.mutant_token_type, batch.is_ligand, batch.token_mask)
        ddg_loss = self.ddg_head.loss(ddg_pred, batch.ddg)

        losses = {
            "diffusion": diffusion_loss,
            "confidence": confidence_loss,
            "contacts": contact_loss,
            "affinity": affinity_loss,
            "ddg": ddg_loss,
        }
        predictions = {
            "rollout_coords": rollout_coords,
            "confidence": confidence_outputs,
            "contact_logits": contact_logits,
            "affinity": affinity_pred,
            "ddg": ddg_pred,
        }
        return losses, predictions
