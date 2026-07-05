import math

import torch

from futureaffinity.data.datatypes import collate
from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.synthetic import make_synthetic_example
from futureaffinity.losses.multitask import aggregate_losses


def test_aggregate_losses_ignores_examples_without_the_label():
    config = FutureAffinityConfig.tiny()
    with_ddg = make_synthetic_example(config, protein_length=6, ligand_size=2, include_ddg=True, seed=0)
    without_ddg = make_synthetic_example(config, protein_length=6, ligand_size=2, include_ddg=False, seed=1)
    batch = collate([with_ddg, without_ddg], config)

    per_task_losses = {
        "diffusion": torch.tensor([1.0, 1.0]),
        "confidence": torch.tensor([1.0, 1.0]),
        "contacts": torch.tensor([1.0, 1.0]),
        "affinity": torch.tensor([1.0, 1.0]),
        "ddg": torch.tensor([10.0, 999.0]),  # the second value must be excluded
    }

    total, logs = aggregate_losses(per_task_losses, batch, config.task_weights)
    assert logs["ddg"] == 10.0  # only example 0 has has_ddg=True
    assert math.isfinite(total.item())


def test_aggregate_losses_handles_a_task_with_zero_labeled_examples():
    config = FutureAffinityConfig.tiny()
    example = make_synthetic_example(
        config, protein_length=6, ligand_size=2, include_ddg=False, include_docking=False, seed=2
    )
    batch = collate([example], config)

    per_task_losses = {
        "diffusion": torch.tensor([1.0]),
        "confidence": torch.tensor([1.0]),
        "contacts": torch.tensor([1.0]),
        "affinity": torch.tensor([1.0]),
        "ddg": torch.tensor([5.0]),  # has_ddg is False for this example
    }

    total, logs = aggregate_losses(per_task_losses, batch, config.task_weights)
    assert math.isnan(logs["ddg"])
    assert math.isfinite(total.item())


def test_aggregate_losses_applies_task_weights():
    config = FutureAffinityConfig.tiny()
    example = make_synthetic_example(config, protein_length=6, ligand_size=2, seed=3)
    batch = collate([example], config)

    per_task_losses = {"diffusion": torch.tensor([2.0])}
    total, _ = aggregate_losses(per_task_losses, batch, {"diffusion": 3.0})
    assert total.item() == 6.0
