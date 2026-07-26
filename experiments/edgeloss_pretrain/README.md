# Edge-aware loss reweighting pretraining experiment

Upweights voxels near high-gradient tissue boundaries relative to flat
interior regions in the MAE reconstruction loss, so the model is pushed to
get anatomical boundaries right instead of coasting on easy flat regions
that dominate by voxel count. Implementation: `src/smri_mae/`, an
isolated copy of `smri_mae` (only `model_mae.py`'s `gradient_magnitude_3d`/
`prepare_edge_weights`/`forward_loss`, `config/default_pretrain.yaml`, and
`main_pretrain.py`'s three import lines differ — everything else is a
byte-identical copy). `edge_loss_weight=0` reproduces the original
unweighted MAE loss exactly (verified bit-identical against `smri_mae`).

Sibling of `smri-fm-tome/experiments/tome_pretrain/` — same data adapter
(`src/data/local_pt_dataset.py`), same model (`mae_vit_small`), same
img_size/epochs/batch settings, same dataset/split. See that experiment's
README for the fuller data-quality writeup (all 965/10 files kept despite
many being low-fill thin-slice scans — a property of the dataset, not this
adapter; not training-corrupting since `img_mask` already excludes
padded/invalid voxels from context and loss).

## Baseline

**There is one shared baseline for both this experiment and the sibling
ToMe experiment** — `edge_loss_weight=0` here and `tome_r=0` there both
reduce to the exact same unmodified `mae_vit_small` recipe (same seed, same
data, same everything else; verified by diffing the two repos'
`default_pretrain.yaml`, the only difference is the `edge_loss_weight` key
existing at all). Run the baseline **once** from either repo — don't run it
again here — and compare both experiments against that single run.

## Running for real

```sh
uv run python src/smri_mae/main_pretrain.py \
  --cfg-path experiments/edgeloss_pretrain/config.yaml
```

Trains from a fresh random init. Compare against the shared baseline (run
from the `smri-fm-tome` repo, see its README) and against `tome_pretrain`'s
own result.
