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
- **No chunking in triangle attention.** `TriangleAttention`'s logits tensor is `O(N^3)` in the
  number of tokens -- fine at the `tiny` config's scale, a real memory concern at `base`/`large`
  scale on long sequences. Real AlphaFold implementations chunk this; this pass doesn't.
- **Ligands have no bonds.** Both the SMILES parser (`data/bindingdb.py`) and the SDF parser
  (`data/pdbbind.py`) read atom composition only, not connectivity. `MockDockingSource`'s "ligand"
  is an unconnected point cloud with a fixed template geometry.
- **`MockDockingSource`'s energy is a toy Lennard-Jones + electrostatics term**, not a real force
  field -- useful as cheap, plentiful synthetic pretraining signal, not as ground truth.

None of this should be used for real structural or affinity conclusions, docking decisions, or
publication. See docs/roadmap.md for what closing these gaps would take.
