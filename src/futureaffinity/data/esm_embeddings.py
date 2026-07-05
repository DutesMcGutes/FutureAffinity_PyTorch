from __future__ import annotations

from pathlib import Path

import torch

from futureaffinity.data.datatypes import Example

_INSTALL_HINT = (
    "ESM embeddings require the 'fair-esm' package (pip install fair-esm), plus a one-time "
    "download of the chosen ESM2 checkpoint the first time it's used. Install via the 'esm' "
    "extra: pip install -e '.[esm]'. This is optional -- FutureAffinityConfig.use_esm_embeddings "
    "defaults to False, and nothing else in the package requires it."
)

DEFAULT_MODEL_NAME = "esm2_t12_35M_UR50D"  # 480-dim; matches FutureAffinityConfig.esm_embedding_dim


def _load_esm_model(model_name: str):
    try:
        import esm
    except ImportError as error:
        raise RuntimeError(_INSTALL_HINT) from error

    model, alphabet = torch.hub.load("facebookresearch/esm", model_name)
    model.eval()
    return model, alphabet


@torch.no_grad()
def compute_esm_embeddings(
    sequences: list[str], model_name: str = DEFAULT_MODEL_NAME, device: str = "cpu"
) -> list[torch.Tensor]:
    """Per-residue ESM2 embeddings for each sequence, as a list of (L_i, dim) tensors."""
    model, alphabet = _load_esm_model(model_name)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers

    _, _, tokens = batch_converter([(str(i), seq) for i, seq in enumerate(sequences)])
    tokens = tokens.to(device)
    result = model(tokens, repr_layers=[num_layers])
    representations = result["representations"][num_layers]

    embeddings = []
    for i, seq in enumerate(sequences):
        # strip the leading <cls> and trailing <eos>/padding tokens ESM adds
        embeddings.append(representations[i, 1 : len(seq) + 1].cpu())
    return embeddings


def cache_embeddings_to_disk(
    ids: list[str], sequences: list[str], output_dir: str | Path, model_name: str = DEFAULT_MODEL_NAME
) -> None:
    """Computes and writes one `{output_dir}/{id}.pt` file per sequence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings = compute_esm_embeddings(sequences, model_name=model_name)
    for identifier, embedding in zip(ids, embeddings):
        torch.save(embedding, output_dir / f"{identifier}.pt")


def load_cached_embedding(output_dir: str | Path, identifier: str) -> torch.Tensor:
    return torch.load(Path(output_dir) / f"{identifier}.pt")


def attach_esm_embedding(example: Example, protein_embedding: torch.Tensor) -> Example:
    """Sets `example.esm_embedding` (zero-padded over ligand tokens) and `has_esm=True`.

    `protein_embedding` must have one row per protein token, in the same
    order as the protein tokens at the front of `example.token_type`.
    """
    num_protein = protein_embedding.shape[0]
    num_ligand = example.num_tokens - num_protein
    if num_ligand < 0:
        raise ValueError("protein_embedding has more rows than the example has protein tokens")

    ligand_padding = torch.zeros(num_ligand, protein_embedding.shape[1])
    example.esm_embedding = torch.cat([protein_embedding, ligand_padding], dim=0)
    example.has_esm = True
    return example
