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

## Blocker: no local pretraining data in the expected format

`default_pretrain.yaml` (which this config overrides) expects **webdataset
tar shards** (`datasets/FOMO_with_dwi/shard.{000000..001620}.tar`). What we
have locally under `/workspace/fomo26/processed_data/PT902_FOMO300K_HF` is
the asparagus/fomo26 per-subject `.pt`/`.pkl` tensor format — a different
pipeline, not directly readable by `main_pretrain.py`'s
`create_data_loaders` (`src/data/mri_data.py`, hard-requires
`webdataset.WebDataset`).

Before an actual training run can start, need either:
- pre-built webdataset shards for FOMO data (R2 has candidate folders worth
  checking: `medarc/leema/FOMO300`, `medarc/leema/FOMO60k`, and variants —
  not yet confirmed these are the right format/split), or
- a local conversion script from the asparagus tensor format to webdataset
  shards (none currently checked into this repo).

## Running once data is resolved

```sh
uv run python src/smri_mae/main_pretrain.py \
  --cfg-path experiments/tome_pretrain/config.yaml
```

For an apples-to-apples baseline comparison, run the same command without
`--cfg-path` (or with `model_kwargs.tome_r=0` via `--overrides`) using
identical data/seed/epochs.
