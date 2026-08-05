# Coarse-to-fine decoding pretraining experiment

Guess each masked patch's overall shape first, then add detail — instead of
predicting all 512 voxels of an 8x8x8 patch in one shot, the decoder head
first predicts a block-averaged, 2x2x2-downsampled ("coarse") version of the
patch, then predicts a fine residual on top of the (upsampled) coarse guess.
Mirrors sketching a face before adding wrinkles, per the hypothesis that
predicting fine detail in one shot is a harder learning target than
predicting shape, then detail. Implementation: `src/smri_mae/`,
an isolated copy of `smri_mae` (only `modules.py`'s new `CoarseToFineHead`,
`avgpool_patch3d`, `upsample_patch3d`; `model_mae.py`'s decoder-head wiring
and `forward_loss`; `config/default_pretrain.yaml`; and `main_pretrain.py`'s
import lines + the `local_pt` data-loader branch differ — everything else is
a byte-identical copy). `coarse_to_fine: false` reproduces the original
unweighted single-head MAE decoder exactly (verified bit-identical against
`smri_mae`, see "Verified" below).

Sibling of `smri-fm-edgeloss/experiments/edgeloss_pretrain/` and
`smri-fm-tome/experiments/tome_pretrain/` — same data adapter
(`src/data/local_pt_dataset.py`), same model (`mae_vit_small`), same
img_size/epochs/batch settings, same dataset/split. See those experiments'
READMEs for the fuller data-quality writeup (all ~965/10 files kept despite
many being low-fill thin-slice scans — a property of the dataset, not this
adapter; not training-corrupting since `img_mask` already excludes
padded/invalid voxels from context and loss).

## What it does

Per masked patch, the decoder's final linear head (`nn.Linear(decoder_dim,
512)` for an 8x8x8x1 patch) is replaced by `CoarseToFineHead`:

1. `coarse_head`: linear projection to a 2x2x2 (`coarse_factor=2`) shape —
   64 values instead of 512 — the "big shape" guess.
2. `detail_head`: linear projection to the full 512 values — the "detail"
   residual.
3. Final prediction = nearest-neighbor-upsampled coarse guess + detail
   residual.

Loss = the original fine-voxel MSE (now scored against the *summed*
prediction) **plus** `coarse_loss_weight * a second MSE term** scoring the
coarse guess alone against a block-averaged (2x2x2-mean-pooled) version of
the same target. This is the "sketch it first" part of the loss — the coarse
head is explicitly graded on getting the overall shape right, not just
riding along as an implicit component of the residual sum.

Both heads are read from the *same* decoder transformer output — this is not
a two-stage architecture with its own second transformer pass, just a
split/residual head design. See the README's "Design choice" note below for
why.

## Design choice: loss/head restructuring, not a literal two-pass pipeline

A more literal reading of "guess the big shape, then refine" would run a
full second decoder pass conditioned on a first (coarse) prediction — e.g.
decode once at low resolution, then feed that back in for a second attention
pass to refine. That's a larger architectural change (a second decoder stack
or a conditioning mechanism) with more moving parts and more ways to get the
comparison wrong.

What's implemented instead keeps the same single decoder pass and single set
of transformer blocks, and only changes the *head* at the very end plus the
*loss*: predict coarse and detail from the same features, sum them, and
score the coarse prediction against a coarse target as an auxiliary
objective. This is the standard "multi-scale / deep supervision" family of
coarse-to-fine methods (cf. Laplacian-pyramid losses, progressive
super-resolution losses) — it tests "does explicitly supervising the model
to get the big shape right, in addition to the fine detail, help" rather
than "does adding a second refinement stage help." Smaller, easier to keep
correct, and — like the edge-loss and ToMe experiments — a single contained
addition to `model_mae.py`/`modules.py` rather than a restructuring of the
training loop. If this pilot looks promising, a genuine two-pass decoder is
a natural follow-up, not a prerequisite.

## Baseline

**Shared baseline across all three sibling experiments** — `coarse_to_fine:
false` here, `edge_loss_weight=0` in `smri-fm-edgeloss`, and `tome_r=0` in
`smri-fm-tome` all reduce to the exact same unmodified `mae_vit_small`
recipe (same seed, same data, same everything else). Run the baseline
**once**, from any one of the three repos, and compare all three experiments
against that single run — don't run it again here.

## Verified before starting a real training run

- `coarse_to_fine=False` reproduces `smri_mae`'s loss bit-for-bit given the
  same weights and seed (synthetic data, small model, CPU/GPU forward +
  backward) — confirms the "off" path is a true no-op change.
- `coarse_to_fine=True` runs forward + backward cleanly, both `coarse_head`
  and `detail_head` receive gradients, and an invalid `coarse_factor` (one
  that doesn't evenly divide `patch_size`) raises a clear `ValueError` at
  construction instead of failing silently or shape-mismatching downstream.
- **End-to-end real-data smoke test** (`debug=true`, 10 train batches + 5
  eval batches, `mae_vit_small`, `coarse_to_fine=True`, fresh random init,
  batch_size=2): real `.pt` files → `LocalPtDataset` → `collate` →
  `densify_sparse_image_batch` → MAE forward (with `CoarseToFineHead`) →
  backward → optimizer step → eval → wandb-style plots → checkpoint saved →
  exited cleanly. Train loss 4.32 → 1.73 over 10 steps, eval loss 1.52 — a
  real, monotonically-dropping loss on real FOMO300K data, not just
  plumbing that doesn't crash.

## Running for real

```sh
uv run python src/smri_mae/main_pretrain.py \
  --cfg-path experiments/coarse2fine_pretrain/config.yaml
```

Trains from a fresh random init (no `ckpt` set). `coarse_factor` (default 2)
and `coarse_loss_weight` (default 1.0) are both worth sweeping — a higher
`coarse_loss_weight` pushes harder on getting the big shape right relative
to fine detail; `coarse_factor=4` makes the "sketch" much blurrier (8x8x8 ->
2x2x2, one value per octant) than `coarse_factor=2` (8x8x8 -> 4x4x4).

For the shared baseline comparison (see "Baseline" above — do this **once**,
not once per experiment), run the same command with
`model_kwargs.coarse_to_fine=false`:

```sh
uv run python src/smri_mae/main_pretrain.py \
  --cfg-path experiments/coarse2fine_pretrain/config.yaml \
  --overrides model_kwargs.coarse_to_fine=false name=pretrain_baseline
```

Compare train/eval loss curves and wall-clock time per epoch across this
experiment, the shared baseline, `edgeloss_pretrain`, and `tome_pretrain`.

Alternative worth considering, same rationale as the sibling experiments:
continue training the already-pretrained checkpoint
(`/workspace/smri-fm/checkpoints/pretrain_full_90_10_h100_4gpu_gbs64_int8_fresh_20260624_045811/checkpoint-last.pth`)
with `--overrides ckpt=<path>` — note that checkpoint's decoder head is a
plain `nn.Linear`, so it only loads cleanly with `coarse_to_fine=false`; the
`coarse_head`/`detail_head` are new parameters with no pretrained
counterpart and would need to train from scratch even when warm-starting the
rest of the model (expect `strict=False`-style loading with those two
reported as missing keys, or load with `coarse_to_fine=false` initially and
treat this as a from-scratch-decoder-head experiment either way).
