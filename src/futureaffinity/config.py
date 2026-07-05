from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FutureAffinityConfig:
    """Sizes and hyperparameters for the whole FutureAffinity model.

    Tokenization follows AlphaFold3's scheme: one token per polymer residue,
    one token per ligand heavy atom. `residue_vocab_size` and
    `ligand_atom_vocab_size` are laid out back to back in a single embedding
    table of size `vocab_size`. Structure supervision/prediction is
    one coordinate per token (Ca-equivalent for residues, the atom position
    for ligand atoms) rather than full atom37 -- see docs/limitations.md.
    """

    # token / pair representation sizes
    token_dim: int = 384
    pair_dim: int = 128
    num_attn_heads: int = 8

    # vocabularies
    residue_vocab_size: int = 26  # 20 amino acids + X/gap/mask/etc.
    ligand_atom_vocab_size: int = 16  # common heavy-atom elements + unknown
    max_chains: int = 8

    # pairformer trunk
    num_trunk_blocks: int = 4
    trunk_dropout: float = 0.0

    # diffusion module (EDM-style: Karras et al. parameterization)
    num_diffusion_steps: int = 20
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 1.0
    num_ensemble_samples: int = 5
    # cheap partial-sampling rollout used to get a "current structure" for the
    # confidence/affinity/ddG heads during training, instead of a full (and, at
    # `base` scale, expensive) reverse diffusion pass on every step
    num_train_rollout_steps: int = 4

    # optional protein language model embeddings
    use_esm_embeddings: bool = False
    esm_embedding_dim: int = 480  # matches esm2_t12_35M_UR50D

    # multi-task loss weights (used by losses/multitask.py)
    task_weights: dict = field(
        default_factory=lambda: {
            "diffusion": 1.0,
            "confidence": 0.25,
            "contacts": 0.5,
            "affinity": 1.0,
            "ddg": 1.0,
        }
    )

    @property
    def vocab_size(self) -> int:
        return self.residue_vocab_size + self.ligand_atom_vocab_size

    @classmethod
    def tiny(cls) -> "FutureAffinityConfig":
        """Small enough to train/test on CPU in seconds. Same code path as `base`."""
        return cls(
            token_dim=64,
            pair_dim=32,
            num_attn_heads=4,
            num_trunk_blocks=2,
            num_diffusion_steps=8,
            num_ensemble_samples=3,
        )

    @classmethod
    def base(cls) -> "FutureAffinityConfig":
        """Roughly OpenFold/AF3-scale dimensions. Needs a real GPU to train at any speed."""
        return cls(
            token_dim=384,
            pair_dim=128,
            num_attn_heads=8,
            num_trunk_blocks=24,
            num_diffusion_steps=200,
            num_ensemble_samples=5,
            use_esm_embeddings=True,
        )
