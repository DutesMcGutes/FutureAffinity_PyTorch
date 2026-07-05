import torch

from futureaffinity.config import FutureAffinityConfig
from futureaffinity.data.synthetic import make_synthetic_batch
from futureaffinity.inference.predict import predict
from futureaffinity.model.model import FutureAffinityModel


def test_forward_produces_every_expected_output():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=3, protein_length=8, ligand_size=3, seed=0)
    model = FutureAffinityModel(config)
    model.eval()

    with torch.no_grad():
        outputs = model(batch)

    for key in ("token", "pair", "rollout_coords", "contact_logits", "confidence", "affinity", "ddg"):
        assert key in outputs

    assert outputs["rollout_coords"].shape == (3, batch.num_tokens, 3)
    assert outputs["affinity"].shape == (3,)
    assert outputs["ddg"].shape == (3,)
    assert torch.isfinite(outputs["rollout_coords"]).all()


def test_predict_end_to_end_from_sequence_and_smiles():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    model = FutureAffinityModel(config)
    model.eval()

    result = predict(
        model,
        protein_sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
        ligand_smiles="CC(=O)Oc1ccccc1C(=O)O",
        num_samples=3,
        num_steps=4,
    )

    num_tokens = len(result["plddt"])
    assert result["structure_ensemble"].shape == (3, num_tokens, 3)
    assert result["contact_probabilities"].shape == (num_tokens, num_tokens)
    assert result["structural_uncertainty"].shape == (num_tokens,)
    assert isinstance(result["affinity_mean"], float)
    assert isinstance(result["affinity_std"], float)
    assert result["affinity_std"] >= 0.0


def test_predict_with_mutant_sequence_returns_ddg():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    model = FutureAffinityModel(config)
    model.eval()

    wildtype = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    mutant = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVA"
    result = predict(model, protein_sequence=wildtype, ligand_smiles=None, mutant_sequence=mutant, num_samples=2, num_steps=3)
    assert "ddg" in result
    assert isinstance(result["ddg"], float)


def test_a_few_training_steps_reduce_total_loss_on_a_fixed_batch():
    torch.manual_seed(0)
    config = FutureAffinityConfig.tiny()
    batch = make_synthetic_batch(config, batch_size=4, protein_length=8, ligand_size=3, seed=42)
    model = FutureAffinityModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    from futureaffinity.losses.multitask import aggregate_losses

    def step():
        losses, _ = model.compute_losses(batch)
        total, _ = aggregate_losses(losses, batch, config.task_weights)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        return float(total.item())

    first_loss = step()
    for _ in range(19):
        last_loss = step()

    assert last_loss < first_loss
