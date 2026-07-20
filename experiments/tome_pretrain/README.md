# Token Merging (ToMe) pretraining experiment

Adds Token Merging (Bolya et al., ICLR 2023, https://arxiv.org/abs/2210.09461)
to the MAE encoder in `src/smri_mae/`. Implementation: `src/smri_mae/tome.py`
plus small additive hooks in `modules.py`
(`Attention.forward_with_metric`, `Block.forward_with_metric`) and
`model_mae.py` (`MaskedEncoder.tome_r`, wired through `MaskedAutoencoderViT`
and all `mae_vit_*` factory functions).

## What it does

Per encoder block, after self-attention, merges the `tome_r` most-similar
token pairs (cosine similarity on the attention keys, reused for free) into
one, weighted by how many original tokens each side already represents. Only
patch tokens are eligible — `cls`/`reg` tokens are never merged. No new
parameters: existing pretrained checkpoints load with `tome_r=0` (identical
to the unmodified encoder) or `tome_r>0` (same weights, fewer tokens flow
through each subsequent block).

## Why

At this codebase's pretraining resolution (`img_size=[208,240,208]`,
`patch_size=8`), the encoder sees ~4,000 visible tokens per volume even after
80% MAE masking — an order of magnitude more than typical 2D ViT token
counts, so attention's O(N^2) cost is real. Token merging is a free
(no-retrain-required) way to cut that down.

## Verified before starting a real training run

Using the real pretrained checkpoint
`checkpoints/pretrain_full_90_10_h100_4gpu_gbs64_int8_fresh_20260624_045811/`
(ViT-L/8, trained on full FOMO300K, 99 epochs):

- `tome_r=0`: checkpoint still loads with `strict=True` (0 missing, 0
  unexpected keys) — confirms the encoder changes are fully backward
  compatible.
- Forward pass at real resolution (batch size 1, inference only, single
  4090), sweeping `tome_r`:

  | tome_r | visible tokens | loss   | time  | peak mem |
  |-------:|----------------:|-------:|------:|---------:|
  |      0 |            4055 | 1.0049 | 0.54s |  2.03 GB |
  |      4 |            3959 | 1.0052 | 0.49s |  2.03 GB |
  |     16 |            3671 | 1.0044 | 0.47s |  2.02 GB |
  |     64 |            2519 | 1.0053 | 0.41s |  1.99 GB |

  Loss stays essentially flat while token count drops substantially, with a
  ~24% runtime reduction at `tome_r=64` even at batch size 1 (where GPU
  memory is dominated by weights, not activations — the real payoff should
  show up more in memory *and* time once training runs at a real batch size
  with gradients enabled). This is not a claim that pretraining quality is
  unaffected — only that the mechanism runs correctly and doesn't blow up.

`config.yaml` here starts at `tome_r=16` as a moderate first pilot value
(cuts ~16*24=384 tokens across the depth-24 encoder, matching the 3671
measured above); adjust up/down once you have a real quality-vs-speed
comparison from an actual training run.

## Known limitation: not yet wired into `forward_visible_ids`

Merging only activates in `MaskedEncoder.forward()` (the masked pretraining
path), because it needs pre-existing per-token position ids (`mask_ids`) to
track which grid position each surviving/merged token should report to the
decoder. `forward_visible_ids` (used for plain feature extraction, e.g. some
downstream eval paths) doesn't track that and will silently skip merging
regardless of `tome_r` — intentional scope limit for this first pass, not a
bug, but worth knowing if downstream eval speed matters too.

## Data: local .pt files instead of real webdataset shards

`default_pretrain.yaml` expects **webdataset tar shards**
(`datasets/FOMO_with_dwi/shard.{000000..001620}.tar`). We only have the
asparagus/fomo26 per-subject `.pt`/`.pkl` tensor format locally
(`/workspace/fomo26/processed_data/PT902_FOMO300K_HF`, 975 files, 5-fold
splits in `split_99_01_00.json`), not real shards, and `main_pretrain.py`'s
`create_data_loaders` hard-requires `webdataset.WebDataset`.

`src/data/local_pt_dataset.py` (`LocalPtDataset` +
`create_local_data_loaders`) reads the `.pt` files directly and packs each
sample into the exact sparse format the training loop already expects
(`mri_data.densify_sparse_image_batch`), so `main_pretrain.py`'s train/eval
loop needed no changes — only `create_data_loaders` gained one branch,
triggered by `datasets.<name>.format: local_pt` in this experiment's
`config.yaml`. Known simplifications of this adapter (documented in the
module docstring): center-crop/pad instead of physical resampling to a
common voxel spacing, a plain intensity threshold instead of a real brain
mask, and a per-volume robust-intensity rescale that isn't necessarily what
the team's own shard-building does. Good enough to get a run launched and
producing a real loss; not a faithful reproduction of the team's pipeline.

**Verified end-to-end** (`debug=true`, 10 batches, no real checkpoint
loaded — fresh random init just to test plumbing):
`.pt` files → `LocalPtDataset` → `collate` → `densify_sparse_image_batch` →
MAE forward (`tome_r=16`) → backward → optimizer step → checkpoint saved →
exited cleanly. Train loss 1.16, eval loss 0.69 on real FOMO300K subset
data. Also checked GPU memory headroom: `batch_size=1` peaked at ~5.4GB,
`batch_size=4` at ~11GB (of 24GB) — config now defaults to `batch_size=4,
accum_iter=4` (effective batch 16); may be able to push batch size higher,
untested.

## Running for real

```sh
uv run python src/smri_mae/main_pretrain.py \
  --cfg-path experiments/tome_pretrain/config.yaml
```

This trains from a fresh random init (no `ckpt` set), for the full
`epochs: 100` default inherited from `default_pretrain.yaml` — that's a long
run on one 4090 at full resolution; consider overriding `epochs` down for a
first real (non-debug) readout, e.g. `--overrides epochs=5`.

For an apples-to-apples baseline comparison, run the same command with
`model_kwargs.tome_r=0` via `--overrides` (same data/seed/epochs
otherwise), and compare train/eval loss curves and wall-clock time per
epoch between the two runs.

Alternative worth considering: instead of training from scratch, continue
training the already-pretrained checkpoint
(`checkpoints/pretrain_full_90_10_h100_4gpu_gbs64_int8_fresh_20260624_045811/checkpoint-last.pth`,
symlinked into `checkpoints/`) with `--overrides ckpt=<path>` (loads model
weights only, starts at epoch 0 with a fresh optimizer — not `resume=true`,
which would also restore optimizer/epoch state and is for resuming an
interrupted run of *this* experiment, not warm-starting from a different
checkpoint). This tests "does turning on merging partway through hurt"
rather than "does merging change what's learned from scratch" — a
different, faster-to-read-out question, since it doesn't need to reach
from-scratch convergence to have signal.

**Also verified end-to-end** (same debug/10-batch smoke test, real
checkpoint loaded via `ckpt=`, `batch_size=2`): loss started at 0.08 and
dropped to ~0.03-0.04 within 10 steps — much lower than the from-scratch run
above (1.16), as expected since these are real trained weights, confirming
checkpoint-loading + the local data adapter + ToMe all work together
correctly:

```sh
uv run python src/smri_mae/main_pretrain.py \
  --cfg-path experiments/tome_pretrain/config.yaml \
  --overrides ckpt=checkpoints/pretrain_full_90_10_h100_4gpu_gbs64_int8_fresh_20260624_045811/checkpoint-last.pth
```
