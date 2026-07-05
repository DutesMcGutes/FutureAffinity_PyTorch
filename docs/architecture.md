# Architecture

FutureAffinity is organized as embedding -> trunk -> diffusion -> multi-task heads, all riding on
one shared token/pair representation:

```
Batch (protein + optional ligand, per-task presence masks)
  -> InputEmbedder            (model/embedding.py)     token, pair
  -> PairformerTrunk          (model/pairformer.py)     token, pair  [N blocks]
  -> DiffusionModule          (model/diffusion.py)      structure ensemble (sampling)
                                                          + denoising loss (training)
  -> ConfidenceHead           (model/heads/confidence.py)  pLDDT + predicted distance error
  -> ContactHead              (model/heads/contacts.py)    interface/contact probabilities
  -> AffinityHead             (model/heads/affinity.py)    E(protein, ligand, conformation)
  -> DDGHead                  (model/heads/ddg.py)         mutant - wildtype effect
  -> uncertainty utilities    (model/heads/uncertainty.py) ensemble variance -> uncertainty
```

## Tokenization

One token per polymer residue, one token per ligand heavy atom (the AlphaFold3 scheme). Both
share one embedding table (`FutureAffinityConfig.vocab_size = residue_vocab_size + ligand_atom_vocab_size`).
Structure is one coordinate per token (a Ca-equivalent for residues), not full atom37 -- see
docs/limitations.md.

## Trunk: real triangular attention, not an MLP stand-in

`model/pairformer.py` implements the actual AlphaFold2/3 communication pattern:
`TriangleMultiplicativeUpdate` (outgoing + incoming, Algorithms 11/12) lets edge (i, j) gather
evidence from every third token k; `TriangleAttention` (starting + ending node, Algorithms 13/14)
does the analogous thing with learned attention instead of a product; `AttentionWithPairBias`
folds pair evidence back into the token representation each block. All of it is real
attention/einsum, sized small by `FutureAffinityConfig.tiny()` but architecturally identical to what
`FutureAffinityConfig.base()` would run at real scale.

## Diffusion: EDM-style, not linear interpolation

`model/diffusion.py` implements Karras et al. (2022) preconditioning (`c_skip`, `c_out`, `c_in`,
`c_noise`), a proper noise schedule, and an ODE-style Euler sampler. `training_loss` is the
denoising-score-matching objective actually used to train the denoiser; `sample_ensemble` draws
several independent reverse trajectories -- a real structural ensemble, which is what
`heads/uncertainty.py` and the affinity/ddG heads consume, not a single point estimate.

During training, the confidence/affinity/ddG heads see a cheap partial-sampling "rollout"
structure (`FutureAffinityConfig.num_train_rollout_steps`) generated under `torch.no_grad()` --
consistent with how diffusion models are normally trained (the denoiser is trained directly via
the score-matching loss; nothing backprops through the iterative sampling loop itself).

## Multi-task heads and the masked loss

Every head is real and shape-correct, but none of the tasks are pretrained yet -- this is the
"multi-task scaffold" milestone. `losses/multitask.py` masks each task's loss by whether a given
batch row actually has that label (`Batch.has_structure`, `has_contacts`, `has_affinity`,
`has_ddg`), which is what lets one batch mix a structure-only PDBbind row, a structure-less
BindingDB row, and a synthetic-docking row without any of them corrupting the others' gradients.

## Data sources

`data/pdbbind.py` and `data/bindingdb.py` parse the real PDBbind index/PDB/SDF and BindingDB TSV
formats (see docs/data-and-weights.md for how to get the real files -- none are bundled).
`datasources/mock_docking.py` is a dependency-free toy physics generator used for synthetic
pretraining signal; `datasources/vina_adapter.py` and `openmm_adapter.py` are real, file-based
integration points for AutoDock Vina / OpenMM, active only if those tools are installed.

## What's deliberately not here yet

See docs/roadmap.md for what's next (real-scale data ingestion, ESM2 caching at scale, active
learning) and docs/limitations.md for the simplifications baked into this pass (single
coordinate per token, PDE instead of full frame-aligned PAE, cheap ddG re-embedding instead of a
full mutant trunk pass, no chunking for triangle attention's O(N^3) memory).
