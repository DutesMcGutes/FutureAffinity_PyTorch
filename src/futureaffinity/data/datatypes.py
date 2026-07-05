from __future__ import annotations

from dataclasses import dataclass, field

import torch

from futureaffinity.config import FutureAffinityConfig


@dataclass
class Example:
    """One training/inference example: a protein (+ optional ligand) system.

    Every label field has a companion `has_*` flag. A single training batch
    can freely mix rows that only have structure, only have affinity, only
    have a synthetic docking energy, etc. -- `losses/multitask.py` masks each
    task's loss by these flags so missing labels never contribute gradient.
    """

    name: str
    token_type: torch.Tensor  # (N,) long, indices into the combined residue+ligand-atom vocab
    is_ligand: torch.Tensor  # (N,) bool
    chain_id: torch.Tensor  # (N,) long
    residue_index: torch.Tensor  # (N,) long, position within its chain (0 for ligand atoms)

    coords: torch.Tensor  # (N, 3) float, ground-truth coordinate per token
    has_structure: bool = False

    esm_embedding: torch.Tensor | None = None  # (N, esm_dim) float, zeros on ligand tokens
    has_esm: bool = False

    contacts: torch.Tensor | None = None  # (N, N) float in [0, 1]
    has_contacts: bool = False

    affinity: float = 0.0  # e.g. pKd or -ddG in kcal/mol; task-defined scale
    has_affinity: bool = False

    mutant_token_type: torch.Tensor | None = None  # (N,) long, same layout as token_type
    ddg: float = 0.0  # kcal/mol, mutant - wildtype
    has_ddg: bool = False

    docking_energy: float = 0.0  # synthetic or real physics-derived energy proxy
    has_docking: bool = False

    @property
    def num_tokens(self) -> int:
        return int(self.token_type.shape[0])


@dataclass
class Batch:
    token_type: torch.Tensor  # (B, N)
    is_ligand: torch.Tensor  # (B, N)
    chain_id: torch.Tensor  # (B, N)
    residue_index: torch.Tensor  # (B, N)
    token_mask: torch.Tensor  # (B, N) bool, True where token is real (not padding)
    pair_mask: torch.Tensor  # (B, N, N) bool

    coords: torch.Tensor  # (B, N, 3)
    has_structure: torch.Tensor  # (B,) bool

    esm_embedding: torch.Tensor  # (B, N, esm_dim)
    has_esm: torch.Tensor  # (B,) bool

    contacts: torch.Tensor  # (B, N, N)
    has_contacts: torch.Tensor  # (B,) bool

    affinity: torch.Tensor  # (B,)
    has_affinity: torch.Tensor  # (B,) bool

    mutant_token_type: torch.Tensor  # (B, N)
    ddg: torch.Tensor  # (B,)
    has_ddg: torch.Tensor  # (B,) bool

    docking_energy: torch.Tensor  # (B,)
    has_docking: torch.Tensor  # (B,) bool

    names: list = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.token_type.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.token_type.shape[1])

    def to(self, device: torch.device | str) -> "Batch":
        moved = {}
        for name, value in vars(self).items():
            moved[name] = value.to(device) if isinstance(value, torch.Tensor) else value
        return Batch(**moved)


def collate(examples: list[Example], config: FutureAffinityConfig) -> Batch:
    """Pad a list of variable-length Examples into a single Batch."""
    if not examples:
        raise ValueError("collate() requires at least one example")

    max_n = max(example.num_tokens for example in examples)
    esm_dim = config.esm_embedding_dim
    device = examples[0].token_type.device

    def pad_1d(values: torch.Tensor, n: int, fill: float | int = 0) -> torch.Tensor:
        pad_len = n - values.shape[0]
        if pad_len == 0:
            return values
        pad = torch.full((pad_len, *values.shape[1:]), fill, dtype=values.dtype, device=values.device)
        return torch.cat([values, pad], dim=0)

    def pad_2d(values: torch.Tensor, n: int, fill: float = 0.0) -> torch.Tensor:
        padded = torch.full((n, n), fill, dtype=values.dtype, device=values.device)
        k = values.shape[0]
        padded[:k, :k] = values
        return padded

    token_type, is_ligand, chain_id, residue_index = [], [], [], []
    token_mask, pair_mask, coords, esm_embedding, contacts = [], [], [], [], []
    mutant_token_type = []
    has_structure, has_esm, has_contacts, has_affinity, has_ddg, has_docking = [], [], [], [], [], []
    affinity, ddg, docking_energy, names = [], [], [], []

    for example in examples:
        n = example.num_tokens
        mask = torch.zeros(max_n, dtype=torch.bool, device=device)
        mask[:n] = True
        token_mask.append(mask)
        pair_mask.append(mask[:, None] & mask[None, :])

        token_type.append(pad_1d(example.token_type, max_n))
        is_ligand.append(pad_1d(example.is_ligand, max_n, fill=False))
        chain_id.append(pad_1d(example.chain_id, max_n))
        residue_index.append(pad_1d(example.residue_index, max_n))
        coords.append(pad_1d(example.coords, max_n, fill=0.0))

        esm = example.esm_embedding
        if esm is None:
            esm = torch.zeros(n, esm_dim, device=device)
        esm_embedding.append(pad_1d(esm, max_n, fill=0.0))

        contact_map = example.contacts if example.contacts is not None else torch.zeros(n, n, device=device)
        contacts.append(pad_2d(contact_map, max_n))

        mutant = example.mutant_token_type if example.mutant_token_type is not None else example.token_type
        mutant_token_type.append(pad_1d(mutant, max_n))

        has_structure.append(example.has_structure)
        has_esm.append(example.has_esm)
        has_contacts.append(example.has_contacts)
        has_affinity.append(example.has_affinity)
        has_ddg.append(example.has_ddg)
        has_docking.append(example.has_docking)

        affinity.append(example.affinity)
        ddg.append(example.ddg)
        docking_energy.append(example.docking_energy)
        names.append(example.name)

    return Batch(
        token_type=torch.stack(token_type),
        is_ligand=torch.stack(is_ligand),
        chain_id=torch.stack(chain_id),
        residue_index=torch.stack(residue_index),
        token_mask=torch.stack(token_mask),
        pair_mask=torch.stack(pair_mask),
        coords=torch.stack(coords),
        has_structure=torch.tensor(has_structure, dtype=torch.bool),
        esm_embedding=torch.stack(esm_embedding),
        has_esm=torch.tensor(has_esm, dtype=torch.bool),
        contacts=torch.stack(contacts),
        has_contacts=torch.tensor(has_contacts, dtype=torch.bool),
        affinity=torch.tensor(affinity, dtype=torch.float32),
        has_affinity=torch.tensor(has_affinity, dtype=torch.bool),
        mutant_token_type=torch.stack(mutant_token_type),
        ddg=torch.tensor(ddg, dtype=torch.float32),
        has_ddg=torch.tensor(has_ddg, dtype=torch.bool),
        docking_energy=torch.tensor(docking_energy, dtype=torch.float32),
        has_docking=torch.tensor(has_docking, dtype=torch.bool),
        names=names,
    )
