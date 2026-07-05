from __future__ import annotations

import torch


def masked_centroid(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Center of mass over valid tokens. `coords` (..., N, 3), `mask` (..., N) -> (..., 1, 3)."""
    mask_f = mask.to(coords.dtype).unsqueeze(-1)
    total = (coords * mask_f).sum(dim=-2, keepdim=True)
    count = mask_f.sum(dim=-2, keepdim=True).clamp(min=1.0)
    return total / count


def center_coords(coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Remove the (masked) center of mass, so absolute translation carries no signal.

    Translation-invariance for free: the structure module never has to learn where in
    space a complex sits, only its internal geometry.
    """
    centered = coords - masked_centroid(coords, mask)
    return centered * mask.to(coords.dtype).unsqueeze(-1)


def random_rotation_matrices(
    batch_size: int, device: torch.device | None = None, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Uniformly-random 3x3 rotation matrices via QR of a Gaussian, sign-corrected to SO(3).

    Returns (batch_size, 3, 3). Uniform over SO(3) (Haar measure), which is what you want
    for rotation augmentation -- not the biased distribution you'd get from random Euler angles.
    """
    a = torch.randn(batch_size, 3, 3, device=device, generator=generator)
    q, r = torch.linalg.qr(a)
    # make the decomposition unique / sign-stable, then force det=+1 (proper rotation)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    q[:, :, 0] = q[:, :, 0] * det.unsqueeze(-1)
    return q


def apply_rotation(coords: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate coordinates. `coords` (B, N, 3), `rotation` (B, 3, 3) -> (B, N, 3)."""
    return torch.einsum("bij,bnj->bni", rotation, coords)


def kabsch_rotation(mobile: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Optimal rotation aligning (centered) `mobile` onto `target` (Kabsch/Umeyama).

    Both inputs must already be centered. Returns (B, 3, 3). Handles the reflection case
    so the result is always a proper rotation (det +1).
    """
    mask_f = mask.to(mobile.dtype).unsqueeze(-1)
    covariance = torch.einsum("bni,bnj->bij", mobile * mask_f, target)
    u, _, vh = torch.linalg.svd(covariance)
    det = torch.linalg.det(torch.einsum("bij,bjk->bik", vh.transpose(-2, -1), u.transpose(-2, -1)))
    correction = torch.eye(3, device=mobile.device, dtype=mobile.dtype).expand(mobile.shape[0], 3, 3).clone()
    correction[:, 2, 2] = det
    return torch.einsum("bij,bjk,bkl->bil", vh.transpose(-2, -1), correction, u.transpose(-2, -1))


def superpose(mobile: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rigid-body align `mobile` onto `target` over valid tokens; returns the aligned `mobile`."""
    mobile_c = center_coords(mobile, mask)
    target_c = center_coords(target, mask)
    rotation = kabsch_rotation(mobile_c, target_c, mask)
    return apply_rotation(mobile_c, rotation)
