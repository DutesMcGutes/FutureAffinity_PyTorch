# Limitations

FutureAffinity is not currently a scientifically accurate structure or affinity predictor. This is
the "multi-task scaffold" milestone: every module is real and shape-correct and trains on
synthetic data, but nothing is pretrained on real structure or affinity data at scale.

Specific simplifications, so they're not mistaken for bugs:

- **One coordinate per token, not full atom37.** Protein residues get a single Ca-equivalent
  position; ligand atoms get their own position. There's no side-chain or backbone-frame modeling.
- **PDE instead of full frame-aligned PAE.** Real AlphaFold PAE needs per-residue reference
  frames, which this coordinate-only representation doesn't have. `ConfidenceHead` predicts
  *predicted distance error* between token pairs instead -- a real, well-defined quantity, just a
  simplification of PAE.
- **ΔΔG from a lightweight re-embedding, not a full mutant trunk pass.** `DDGHead` embeds
  wildtype/mutant identity with its own small embedding table rather than rerunning the whole
  Pairformer trunk on the mutant sequence, for speed. Swapping in full mutant re-embedding is a
  natural next step once compute allows.
- **No backprop through the diffusion sampler.** `sample_ensemble` runs under `torch.no_grad()`;
  the trunk is trained via the direct denoising-score-matching loss (`DiffusionModule.training_loss`),
  not by differentiating through the reverse-diffusion loop. This matches how EDM-style models are
  normally trained, but it does mean the affinity/ddG/confidence heads only get gradient through
  their token/pair inputs, not through the specific sampled geometry.
- **Ligand bonds are optional, not default.** The base SMILES/SDF readers (`data/bindingdb.py`,
  `data/pdbbind.py`) read atom composition only. Real connectivity and 3D conformers are available
  through the optional RDKit path (`data/ligand_rdkit.py`) but aren't required.
- **The docking oracle is a real optimizer on a *toy* force field.** `AnalyticalDockingOracle` does
  genuine gradient-descent rigid-body pose minimization (not random sampling), but the
  Lennard-Jones + toy-electrostatics potential is not parameterized to real chemistry and the ligand
  is treated as rigid. It's a synthetic-supervision oracle, not a substitute for Vina/OpenMM (which
  slot into the same interface).

## What is *not* a limitation (things sometimes mistaken for gaps)

- **Equivariance is handled** -- by centering + SO(3) augmentation, not built-in equivariant layers
  (a deliberate choice; see docs/adr/0002). Translation-invariance is exact and tested.
- **Triangle attention can be chunked** to control its `O(N^3)` memory
  (`config.triangle_attention_chunk_size`), verified bit-identical to the unchunked path.
- **Scale hooks exist** -- gradient checkpointing, AMP, and a correct DDP scaffold are implemented
  (off by default); see docs/scaling.md.

None of this should be used for real structural or affinity conclusions, docking decisions, or
publication *until trained* -- the architecture and evaluation are real, the weights are not. See
docs/scaling.md and docs/roadmap.md for what training at scale would take.
