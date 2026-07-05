"""Instructions + a layout check for real PDBbind data -- this script does not download anything.

PDBbind (http://www.pdbbind.org.cn) requires a free academic registration
before download; FutureAffinity cannot fetch or redistribute it automatically.
This script just verifies a local PDBbind download is laid out the way
`futureaffinity.data.pdbbind.PDBbindDataset` expects, so you can catch a bad
extraction before a training run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def check_layout(root: Path) -> list[str]:
    problems = []
    index_dir = root / "index"
    if not index_dir.is_dir():
        problems.append(f"missing {index_dir} (expected the PDBbind 'index' subdirectory)")
    elif not list(index_dir.glob("INDEX_general_PL_data*")):
        problems.append(f"no INDEX_general_PL_data* file found under {index_dir}")

    complex_dirs = [p for p in root.iterdir() if p.is_dir() and p.name != "index"]
    if not complex_dirs:
        problems.append(f"no per-complex subdirectories found directly under {root}")
    else:
        sample = complex_dirs[0]
        pdb_id = sample.name
        if not (sample / f"{pdb_id}_pocket.pdb").exists():
            problems.append(f"expected {sample / f'{pdb_id}_pocket.pdb'} (checked first complex found)")
        if not (sample / f"{pdb_id}_ligand.sdf").exists():
            problems.append(f"expected {sample / f'{pdb_id}_ligand.sdf'} (checked first complex found)")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Path to an extracted local PDBbind download.")
    args = parser.parse_args()

    if not args.root.is_dir():
        print(
            f"{args.root} does not exist yet.\n\n"
            "To get real PDBbind data:\n"
            "  1. Register for a free academic license at http://www.pdbbind.org.cn\n"
            "  2. Download the 'general set' (structures + INDEX_general_PL_data.<year>)\n"
            "  3. Extract it so this directory contains an 'index/' folder and one "
            "subdirectory per PDB entry (each with '<id>_pocket.pdb' and '<id>_ligand.sdf')\n"
            "  4. Re-run this script against the extracted root to verify the layout\n"
        )
        sys.exit(1)

    problems = check_layout(args.root)
    if problems:
        print(f"{args.root} does not look like a PDBbind download:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print(f"{args.root} looks like a valid PDBbind layout. Try:")
    print(f"  python -m futureaffinity.training.train --pdbbind-root {args.root}")


if __name__ == "__main__":
    main()
