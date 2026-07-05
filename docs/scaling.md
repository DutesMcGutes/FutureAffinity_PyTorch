# Scaling plan: from `tiny` to `base`/`large`

This document is the "what would it actually take to train this" plan. The code path is identical
across configs (`FutureAffinityConfig.tiny()` / `.base()`); only sizes and the compute budget
change. Nothing here has been run at scale -- it's the design and the estimate, ready to execute
when compute is available.

## Model sizes

| Config | token_dim | pair_dim | trunk blocks | diff. steps | ~params | fits on |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` | 64 | 32 | 2 | 8 | ~1.5 M | CPU, seconds/step |
| `base` | 384 | 128 | 24 | 200 | ~120 M | 1x A100-80GB (with the knobs below) |
| `large` (sketch) | 512 | 192 | 48 | 200 | ~450 M | 8x A100/H100, sharded |

Parameter count is dominated by the trunk: each Pairformer block is ~`O(token_dim^2 + pair_dim^2)`
in projections, times `num_trunk_blocks`.

## The memory wall, and the three knobs that move it

The dominant activation cost is **triangle attention**: it materializes a
`(B, N, N, N, H)` logits tensor -- `O(N^3)` in sequence length `N`. At `N=384` tokens, `H=8`,
fp32, one such tensor is `384^3 * 8 * 4 bytes ≈ 1.7 GB`, and there are several per block across 24
blocks. This, not parameter count, is what sets the hardware requirement.

Three knobs, all implemented and off by default (see `config.py`):

1. **Chunked triangle attention** (`triangle_attention_chunk_size`): processes the query-row axis
   in chunks, cutting the peak logits tensor by a factor of `N / chunk`. Verified bit-identical to
   the unchunked path (`tests/test_scale_hooks.py`). This is the single biggest lever for long `N`.
2. **Gradient checkpointing** (`use_gradient_checkpointing`): recompute each trunk block in the
   backward pass instead of storing its activations -- ~1 extra forward for a large activation
   saving, which is what lets a 24-48 block trunk fit at all.
3. **Mixed precision** (`--amp`): bf16/fp16 autocast on CUDA halves activation memory and roughly
   doubles throughput on A100/H100 tensor cores.

Crop size (max tokens per training example) is the fourth, blunter lever: AlphaFold-family models
train on cropped regions (e.g. 256-384 tokens) rather than whole complexes precisely because of
the `O(N^3)` wall. The data pipeline should crop; this repo's loaders currently pass whole
complexes (fine at `tiny` scale).

## Distributed training

`training/distributed.py` provides correct DDP scaffolding. The non-obvious part it gets right:
DDP only all-reduces gradients for parameters touched by the wrapped module's `forward`, and this
model's training entry point is `compute_losses`, not `forward` -- so `TrainStep` wraps the whole
loss computation into a `forward`, and DDP wraps `TrainStep`. Launch with `torchrun`:

```bash
torchrun --nproc_per_node=8 -m futureaffinity.training.train --preset base --amp --grad-checkpoint
```

Data-parallel scales throughput linearly to the point where the trunk fits on one device. Beyond
`large`, you'd move to tensor/pipeline parallelism (FSDP is the natural next step for this code --
it sharded-wraps the same `TrainStep`).

## Rough compute budget for a `base` run

These are order-of-magnitude planning numbers, not benchmarks:

- **Dataset**: ~200k PDB structures (co-folding pretraining) + ~20k PDBbind complexes + a filtered
  BindingDB subset (~1-2 M sequence/SMILES/affinity rows). See docs/data-engineering.md.
- **Pretraining** (structure + contacts + confidence, self-distillation from ensembles): the
  compute-dominant phase. AF3-scale pretraining is reported in the low-thousands of TPU/GPU-days;
  a 120 M `base` model on cropped inputs is far cheaper -- plan for **~1-4k A100-hours** for a first
  usable checkpoint (8x A100 for ~1-3 weeks), then iterate.
- **Affinity/ΔΔG fine-tuning**: cheap by comparison -- the labeled data is small, so this is
  **tens to low-hundreds of A100-hours**, and is where most of the iteration happens.
- **Cost**: at ~$2/A100-hour spot, a first `base` checkpoint is a **~$2k-8k** experiment, not a
  moonshot. The expensive part is the data engineering and eval iteration, not the raw FLOPs.

## Order of operations when compute arrives

1. Run the data-engineering pipeline (dedup, cluster, split -- docs/data-engineering.md) and cache
   ESM2 embeddings (`scripts/cache_esm_embeddings.py`).
2. Enable crop + `--grad-checkpoint --amp --triangle-chunk` and confirm one `base` step fits on the
   target GPU.
3. Pretrain on structure/contacts/confidence + synthetic docking energy until the held-out
   structural metrics (docs/evaluation.md) plateau.
4. Fine-tune jointly on affinity/ΔΔG; watch for the affinity head overfitting the small labeled set.
5. Active learning: use the ensemble-uncertainty head to pick the next complexes to label.
