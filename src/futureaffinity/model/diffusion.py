from __future__ import annotations

import math

import torch
from torch import nn

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.geometry import apply_rotation, center_coords, random_rotation_matrices
from futureaffinity.model.pairformer import AttentionWithPairBias, Transition


def karras_sigma_schedule(
    num_steps: int, sigma_min: float, sigma_max: float, rho: float = 7.0, device=None
) -> torch.Tensor:
    """The Karras et al. (2022) noise schedule used by EDM-style diffusion.

    Returns `num_steps + 1` decreasing sigmas, the last of which is 0 (the
    clean-data endpoint), so the sampler has `num_steps` integration steps.
    """
    steps = torch.arange(num_steps, dtype=torch.float32, device=device)
    t = steps / max(num_steps - 1, 1)
    sigmas = (sigma_max ** (1 / rho) + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def edm_preconditioning(sigma: torch.Tensor, sigma_data: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """c_skip, c_out, c_in, c_noise from Karras et al. (2022), Table 1."""
    sigma_sq = sigma**2
    data_sq = sigma_data**2
    c_skip = data_sq / (sigma_sq + data_sq)
    c_out = sigma * sigma_data / torch.sqrt(sigma_sq + data_sq)
    c_in = 1.0 / torch.sqrt(sigma_sq + data_sq)
    c_noise = 0.25 * torch.log(sigma.clamp(min=1e-12))
    return c_skip, c_out, c_in, c_noise


class SigmaEmbedding(nn.Module):
    """Random Fourier features of log(sigma), the noise-conditioning embedding EDM uses."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SigmaEmbedding dim must be even")
        self.register_buffer("freqs", torch.randn(dim // 2) * 16.0)
        self.proj = nn.Linear(dim, dim)

    def forward(self, c_noise: torch.Tensor) -> torch.Tensor:
        angles = c_noise[:, None] * self.freqs[None, :] * 2.0 * math.pi
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.proj(features)


class DiffusionDenoiser(nn.Module):
    """F_theta: predicts a coordinate update conditioned on trunk representations + noise level.

    Reuses the trunk's pair-biased attention block so the denoiser reasons
    about the same token/pair evidence the Pairformer built up, rather than
    being a separate un-conditioned network.
    """

    def __init__(self, config: FutureAffinityConfig, num_blocks: int = 2) -> None:
        super().__init__()
        self.coord_proj = nn.Linear(3, config.token_dim)
        self.sigma_embedding = SigmaEmbedding(config.token_dim)
        self.input_norm = nn.LayerNorm(config.token_dim)

        self.attn_blocks = nn.ModuleList(
            AttentionWithPairBias(config.token_dim, config.pair_dim, config.num_attn_heads) for _ in range(num_blocks)
        )
        self.transitions = nn.ModuleList(Transition(config.token_dim) for _ in range(num_blocks))
        self.out_proj = nn.Linear(config.token_dim, 3)

    def forward(
        self,
        noisy_coords: torch.Tensor,
        c_noise: torch.Tensor,
        token_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_norm(self.coord_proj(noisy_coords) + self.sigma_embedding(c_noise)[:, None, :] + token_repr)
        for attn, transition in zip(self.attn_blocks, self.transitions):
            x = x + attn(x, pair_repr, token_mask)
            x = x + transition(x)
        return self.out_proj(x) * token_mask[..., None].to(x.dtype)


class DiffusionModule(nn.Module):
    """EDM-style (Karras et al. 2022) coordinate diffusion, conditioned on the Pairformer trunk.

    `training_loss` implements the denoising-score-matching objective used to
    train this module. `sample_ensemble` draws several independent reverse
    trajectories -- a real structural ensemble, not one point estimate --
    which feeds both `heads/uncertainty.py` and the affinity/ddG heads
    (idea: distill from an ensemble instead of a single structure).
    """

    def __init__(self, config: FutureAffinityConfig) -> None:
        super().__init__()
        self.config = config
        self.denoiser = DiffusionDenoiser(config)

    def denoise(
        self,
        noisy_coords: torch.Tensor,
        sigma: torch.Tensor,
        token_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, self.config.sigma_data)
        raw = self.denoiser(noisy_coords * c_in[:, None, None], c_noise, token_repr, pair_repr, token_mask)
        return c_skip[:, None, None] * noisy_coords + c_out[:, None, None] * raw

    def training_loss(
        self,
        coords: torch.Tensor,
        token_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        token_mask: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Per-batch-element mean squared denoising error, EDM-weighted, masked to valid tokens.

        The structure module is not intrinsically SE(3)-equivariant (see the ADR referenced in
        the config), so equivariance is instilled by data: every target is centered (removing
        translation) and, at train time, randomly rotated (removing any preferred orientation).
        Over training this teaches the denoiser to be robust to global pose rather than baking it
        into the architecture -- the same tradeoff AlphaFold3 makes.
        """
        batch_size, device = coords.shape[0], coords.device

        if self.config.center_coordinates:
            coords = center_coords(coords, token_mask)
        if self.config.augment_rotation:
            rotation = random_rotation_matrices(batch_size, device=device, generator=generator)
            coords = apply_rotation(coords, rotation) * token_mask[..., None].to(coords.dtype)

        log_sigma = torch.randn(batch_size, device=device, generator=generator) * 1.2 - 1.2
        sigma = log_sigma.exp()

        noise = torch.randn(coords.shape, device=device, generator=generator)
        noisy_coords = coords + sigma[:, None, None] * noise

        denoised = self.denoise(noisy_coords, sigma, token_repr, pair_repr, token_mask)

        weight = (sigma**2 + self.config.sigma_data**2) / (sigma * self.config.sigma_data) ** 2
        per_token_error = ((denoised - coords) ** 2).sum(dim=-1)  # (B, N)
        mask = token_mask.to(per_token_error.dtype)
        per_example_error = (per_token_error * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
        return weight * per_example_error

    @torch.no_grad()
    def sample_ensemble(
        self,
        token_repr: torch.Tensor,
        pair_repr: torch.Tensor,
        token_mask: torch.Tensor,
        num_samples: int | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Returns (B, num_samples, N, 3): an ensemble of independently sampled structures."""
        num_samples = num_samples or self.config.num_ensemble_samples
        num_steps = num_steps or self.config.num_diffusion_steps
        batch_size, num_tokens = token_repr.shape[0], token_repr.shape[1]
        device = token_repr.device

        sigmas = karras_sigma_schedule(num_steps, self.config.sigma_min, self.config.sigma_max, device=device)

        samples = []
        for _ in range(num_samples):
            x = torch.randn(batch_size, num_tokens, 3, device=device, generator=generator) * sigmas[0]
            for i in range(num_steps):
                sigma_cur = sigmas[i].expand(batch_size)
                denoised = self.denoise(x, sigma_cur, token_repr, pair_repr, token_mask)
                direction = (x - denoised) / sigma_cur[:, None, None].clamp(min=1e-12)
                x = x + direction * (sigmas[i + 1] - sigmas[i])
            samples.append(x)
        return torch.stack(samples, dim=1)
