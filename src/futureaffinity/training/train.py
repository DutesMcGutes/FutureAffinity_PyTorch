from __future__ import annotations

import argparse

import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.bindingdb import BindingDBDataset
from futureaffinity.data.datatypes import Example, collate
from futureaffinity.data.pdbbind import PDBbindDataset
from futureaffinity.data.synthetic import make_synthetic_example
from futureaffinity.model.model import FutureAffinityModel
from futureaffinity.training.distributed import (
    TrainStep,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    maybe_init_distributed,
)


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
    device: torch.device | str = "cpu",
    use_amp: bool = False,
) -> FutureAffinityModel:
    """Runs `num_steps` of multi-task training, round-robining example sources across the batch.

    Mixing sources (e.g. synthetic + PDBbind + BindingDB) inside one batch is exactly what
    `losses/multitask.py`'s masking is for: a structure-only PDBbind row and a
    structure-less BindingDB row can sit in the same batch and each only contributes
    gradient to the tasks it actually has labels for.

    Scale knobs (all no-ops on CPU / single process, so the default path is unchanged): `use_amp`
    enables mixed precision on CUDA; if launched under torchrun the model is wrapped in DDP and
    gradients all-reduce across ranks (see training/distributed.py). Gradient checkpointing and
    chunked triangle attention are controlled by the config, not here.
    """
    torch.manual_seed(seed)
    device = torch.device(device)
    model = FutureAffinityModel(config).to(device)
    step_module: torch.nn.Module = TrainStep(model, config)
    if is_distributed():
        step_module = torch.nn.parallel.DistributedDataParallel(step_module)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator(device=device).manual_seed(seed)  # reproducible noise/augmentation
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for step in range(1, num_steps + 1):
        examples = [sources[i % len(sources)].next_example() for i in range(batch_size)]
        batch = collate(examples, config).to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            total_loss, logs = step_module(batch, generator)
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if is_main_process() and (step == 1 or step % log_every == 0):
            log_str = " ".join(f"{name}={value:.4f}" for name, value in logs.items() if value == value)
            print(f"step {step}/{num_steps} total={total_loss.item():.4f} {log_str}")

    if checkpoint_path and is_main_process():
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
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA only).")
    parser.add_argument("--grad-checkpoint", action="store_true", help="Recompute trunk activations to save memory.")
    parser.add_argument("--triangle-chunk", type=int, default=None, help="Chunk size for O(N^3) triangle attention.")
    args = parser.parse_args()

    import dataclasses

    base_config = FutureAffinityConfig.tiny() if args.preset == "tiny" else FutureAffinityConfig.base()
    config = dataclasses.replace(
        base_config,
        use_gradient_checkpointing=args.grad_checkpoint,
        triangle_attention_chunk_size=args.triangle_chunk,
    )

    device = maybe_init_distributed()
    try:
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
            device=device,
            use_amp=args.amp,
        )
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
