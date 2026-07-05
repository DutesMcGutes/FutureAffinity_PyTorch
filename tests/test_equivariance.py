import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.synthetic import make_synthetic_batch
from futureaffinity.geometry import center_coords, masked_centroid, random_rotation_matrices
from futureaffinity.model.model import FutureAffinityModel


def test_random_rotations_are_proper_and_orthonormal():
    rotations = random_rotation_matrices(64)
    identity = torch.eye(3).expand(64, 3, 3)
    assert torch.allclose(rotations @ rotations.transpose(-1, -2), identity, atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotations), torch.ones(64), atol=1e-5)


def test_center_coords_removes_center_of_mass():
    torch.manual_seed(0)
    coords = torch.randn(3, 20, 3) * 4.0 + 100.0
    mask = torch.ones(3, 20, dtype=torch.bool)
    mask[0, 15:] = False  # ragged: centroid must respect the mask

    centered = center_coords(coords, mask)
    assert torch.allclose(masked_centroid(centered, mask), torch.zeros(3, 1, 3), atol=1e-4)


def test_diffusion_loss_is_translation_invariant():
    """Centering makes the training objective exactly invariant to a global shift of the target."""
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=2, protein_length=8, ligand_size=3, seed=1)
    model = FutureAffinityModel(config)
    token, pair = model.encode(batch)

    shift = torch.tensor([25.0, -12.0, 7.0])
    loss_a = model.diffusion.training_loss(
        batch.coords, token, pair, batch.token_mask, generator=torch.Generator().manual_seed(7)
    )
    loss_b = model.diffusion.training_loss(
        batch.coords + shift, token, pair, batch.token_mask, generator=torch.Generator().manual_seed(7)
    )
    assert torch.allclose(loss_a, loss_b, atol=1e-4)


def test_rotation_augmentation_actually_varies_orientation():
    """With augmentation on, two loss evaluations with different generators see different poses,
    so the denoiser is trained across orientations rather than one canonical frame."""
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=2, protein_length=8, ligand_size=3, seed=2)
    model = FutureAffinityModel(config)
    token, pair = model.encode(batch)

    loss_seed_a = model.diffusion.training_loss(
        batch.coords, token, pair, batch.token_mask, generator=torch.Generator().manual_seed(1)
    )
    loss_seed_b = model.diffusion.training_loss(
        batch.coords, token, pair, batch.token_mask, generator=torch.Generator().manual_seed(2)
    )
    # different augmentation draws -> different losses (not a degenerate no-op)
    assert not torch.allclose(loss_seed_a, loss_seed_b)
