# ADR 0001 — One token per residue / heavy atom

**Status:** accepted

## Context

Co-folding must represent proteins (polymers) and ligands (small molecules) in one model. Proteins
are naturally residue-level; ligands are naturally atom-level. A single representation has to hold
both.

## Decision

Follow AlphaFold3's tokenization: **one token per polymer residue, one token per ligand heavy
atom**, in a single shared token stream, distinguished by an `is_ligand` flag. Both share one
embedding table (`FutureAffinityConfig.vocab_size = residue_vocab_size + ligand_atom_vocab_size`).
Structure is one coordinate per token.

## Alternatives considered

- **All-atom for both** (atom-level protein too): maximally uniform, but blows up sequence length
  (`O(N^3)` triangle attention makes this very expensive) for little gain on the polymer side, where
  residue-level backbones are a strong prior.
- **Residue-level for both** (coarse-grain the ligand): loses the per-atom ligand geometry that
  affinity depends on -- a non-starter for pose/interface prediction.

## Consequences

- Uniform downstream code: embedding, trunk, diffusion, and every head operate on one token stream
  and never special-case ligands beyond reading `is_ligand`.
- **Limitation:** one coordinate per token means no protein side chains and no full atom37 -- the
  model reasons about a Cα-level protein and an all-heavy-atom ligand. Fine for interface/pose and
  affinity; not a full all-atom structure predictor. Lifting this is a documented future step
  (see docs/limitations.md).
- Ligand connectivity isn't in the base featurization (bag-of-elements); the optional RDKit path
  (`data/ligand_rdkit.py`) adds real bonds and 3D conformers when available.
