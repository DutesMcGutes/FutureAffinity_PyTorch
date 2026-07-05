# Roadmap

This pass shipped the full architecture (real Pairformer trunk, EDM-style diffusion, every
multi-task head) plus real dataset loaders, all exercised end to end on synthetic + tiny fixture
data at the `tiny` config. It deliberately did not attempt real-scale training -- that's a
multi-week program on its own. Roughly in priority order:

1. **Download real PDBbind + a filtered BindingDB subset.** `data/pdbbind.py` and
   `data/bindingdb.py` already parse the real formats; what's missing is actually fetching and
   curating them at scale (dedup, train/val/test split by sequence/scaffold similarity to avoid
   leakage, filtering BindingDB down from its full multi-million-row bulk export).
2. **Cache real ESM2 embeddings** for the sequences in that dataset (`scripts/cache_esm_embeddings.py`
   already does the computation; running it over hundreds of thousands of sequences is a
   compute/time decision for whoever runs it).
3. **Train the `base` config on real GPU hardware.** Same code path as `tiny`, but 24 trunk blocks
   and 200 diffusion steps needs real compute and real wall-clock time -- not something to start
   without knowing the budget.
4. **Chunk `TriangleAttention`** so `base`-scale sequences don't blow up memory (see
   docs/limitations.md).
5. **Full mutant re-embedding for ΔΔG**, replacing the current lightweight identity-diff
   approximation, once training compute allows rerunning the trunk per mutant.
6. **Wire real Vina/OpenMM runs** into the synthetic-supervision pipeline as an alternative to
   `MockDockingSource`, for whoever has those tools installed and wants higher-fidelity (if much
   slower) synthetic labels.
7. **Active learning loop**: use `heads/uncertainty.py`'s ensemble variance to flag
   high-uncertainty protein-ligand systems for prioritized real-label collection, closing the loop
   the project brief describes.

None of the above is scheduled -- each is a real time/compute/data decision, not a default next
action.
