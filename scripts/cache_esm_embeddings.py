"""Precompute and cache ESM2 embeddings for a list of sequences.

Optional -- FutureAffinityConfig.use_esm_embeddings defaults to False and
nothing else requires this. Needs `pip install -e '.[esm]'` and a one-time
download of the chosen ESM2 checkpoint (done automatically by torch.hub the
first time it runs).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from futureaffinity.data.esm_embeddings import DEFAULT_MODEL_NAME, cache_embeddings_to_disk


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequences_json", type=Path, help='JSON file: {"id1": "SEQUENCE1", "id2": "SEQUENCE2", ...}'
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()

    sequences_by_id = json.loads(args.sequences_json.read_text(encoding="utf-8"))
    ids = list(sequences_by_id.keys())
    sequences = [sequences_by_id[identifier] for identifier in ids]

    print(f"Computing {args.model_name} embeddings for {len(ids)} sequences...")
    cache_embeddings_to_disk(ids, sequences, args.output_dir, model_name=args.model_name)
    print(f"Wrote {len(ids)} .pt files to {args.output_dir}")


if __name__ == "__main__":
    main()
