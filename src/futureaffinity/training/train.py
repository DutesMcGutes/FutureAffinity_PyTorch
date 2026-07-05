from __future__ import annotations

import argparse

import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.bindingdb import BindingDBDataset
from futureaffinity.data.datatypes import Example, collate
from futureaffinity.data.pdbbind import PDBbindDataset
from futureaffinity.data.synthetic import make_synthetic_example
from futureaffinity.losses.multitask import aggregate_losses
from futureaffinity.model.model import FutureAffinityModel


class SyntheticSource:
    """Infinite synthetic examples -- always available, used for the pretraining/smoke-test signal."""

    def __init__(self, config: FutureAffinityConfig, protein_length: int = 20, ligand_size: int = 6) -> None:
        self.config = config
        self.protein_length = protein_length
        self.ligand_size = ligand_size
        self._counter = 0

    def next_example(self) -> Example:
        self._counter += 1
        return make_synthetic_example(
            self.config,
            protein_length=self.protein_length,
            ligand_size=self.ligand_size,
            include_ddg=(self._counter % 2 == 0),
            seed=self._counter,
        )


class DatasetSource:
    """Cycles through a finite, indexable real dataset (PDBbind, BindingDB, ...)."""

    def __init__(self, dataset) -> None:
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        self.dataset = dataset
        self._index = 0

    def next_example(self) -> Example:
        example = self.dataset[self._index % len(self.dataset)]
        self._index += 1
        return example


def train(
    config: FutureAffinityConfig,
    sources: list,
    num_steps: int,
    batch_size: int,
    lr: float,
    log_every: int,
    seed: int,
    checkpoint_path: str | None = None,
) -> FutureAffinityModel:
    """Runs `num_steps` of multi-task training, round-robining example sources across the batch.

    Mixing sources (e.g. synthetic + PDBbind + BindingDB) inside one batch is exactly what
    `losses/multitask.py`'s masking is for: a structure-only PDBbind row and a
    structure-less BindingDB row can sit in the same batch and each only contributes
    gradient to the tasks it actually has labels for.
    """
    torch.manual_seed(seed)
    model = FutureAffinityModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    for step in range(1, num_steps + 1):
        examples = [sources[i % len(sources)].next_example() for i in range(batch_size)]
        batch = collate(examples, config)

        losses, _ = model.compute_losses(batch, generator=generator)
        total_loss, logs = aggregate_losses(losses, batch, config.task_weights)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0:
            log_str = " ".join(f"{name}={value:.4f}" for name, value in logs.items() if value == value)
            print(f"step {step}/{num_steps} total={total_loss.item():.4f} {log_str}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")

    return model


def _build_sources(config: FutureAffinityConfig, args: argparse.Namespace) -> list:
    sources: list = [SyntheticSource(config)]
    if args.pdbbind_root:
        sources.append(DatasetSource(PDBbindDataset(args.pdbbind_root, config)))
    if args.bindingdb_tsv:
        sources.append(DatasetSource(BindingDBDataset(args.bindingdb_tsv, config)))
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FutureAffinityModel on synthetic and/or real data.")
    parser.add_argument("--preset", choices=["tiny", "base"], default="tiny")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--pdbbind-root", type=str, default=None, help="Root of a real PDBbind-layout directory.")
    parser.add_argument("--bindingdb-tsv", type=str, default=None, help="Path to a BindingDB bulk TSV export.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Where to save the trained state_dict.")
    args = parser.parse_args()

    config = FutureAffinityConfig.tiny() if args.preset == "tiny" else FutureAffinityConfig.base()
    sources = _build_sources(config, args)

    train(
        config,
        sources,
        num_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
