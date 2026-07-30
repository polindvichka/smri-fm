"""Measure real per-sample valid-patch occupancy over the local FOMO_with_dwi
train shards, and derive the tile-aligned `l_vis`/`q_pred` budget this
implies via `layout.occupancy_percentiles`/`choose_l_vis`/`choose_q_pred`.

This is the M1 gate measurement TENSTORRENT_PORT.md section 1 calls for
("Measure this distribution on real shards before anything else") and that
`memory_probe.py`'s module docstring explicitly says it could *not* do
("No real MRI shards exist on this host") -- they now do
(`datasets/FOMO_with_dwi/shard.000000-000002.tar`, 450 train samples), so
this script does the real measurement instead of the memory probe's
deliberately-conservative full-grid ceiling.

Not a unit test (filename does not match `test_*.py`), same convention as
`memory_probe.py` -- run directly with the tt-train Python interpreter. Does
**not** open a device or touch `ttnn`/`ttml` at all: this is pure host-side
numpy/CPU-torch work (`data.mri_data.make_sparse_wds_dataset`/`collate`,
unmodified, + `packing.compute_valid_patch_counts`, unmodified).

Real ViT-L hyperparameters and mask_ratio (0.80, from
`src/smri_mae/config/default_pretrain.yaml`) mirrored from
`memory_probe.py`'s own module docstring -- not re-derived independently, so
this script's `IMG_SIZE`/`PATCH_SIZE`/`MASK_RATIO` constants intentionally
match that file verbatim.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from data.mri_data import collate, make_sparse_wds_dataset
from smri_mae_tt import real_data
from smri_mae_tt.layout import ShapeContract, choose_l_vis, choose_q_pred, occupancy_percentiles
from smri_mae_tt.packing import compute_valid_patch_counts

IMG_SIZE = (208, 240, 208)
PATCH_SIZE = 8
MASK_RATIO = 0.80


def measure(urls: list[str], *, batch_size: int = 25) -> np.ndarray:
    """One full deterministic pass (`shuffle=False` -> finite iteration,
    exactly `real_data.RealValSource`'s own policy) over `urls`, returning
    the per-sample valid-patch count for every sample seen.
    """
    dataset = make_sparse_wds_dataset(urls, shuffle=False, buffer_size=1)
    counts: list[np.ndarray] = []
    samples: list[dict] = []
    n_seen = 0
    for sample in dataset:
        samples.append(sample)
        if len(samples) == batch_size:
            batch = collate(samples, include_meta=False)
            counts.append(compute_valid_patch_counts(batch["img_mask"], IMG_SIZE, PATCH_SIZE))
            n_seen += len(samples)
            samples = []
    if samples:
        batch = collate(samples, include_meta=False)
        counts.append(compute_valid_patch_counts(batch["img_mask"], IMG_SIZE, PATCH_SIZE))
        n_seen += len(samples)
    if not counts:
        raise RuntimeError(f"measured zero samples from urls={urls} -- shard(s) empty or unreadable")
    return np.concatenate(counts)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "both"), default="train")
    args = parser.parse_args(argv)

    if args.split == "train":
        urls = list(real_data.TRAIN_SHARD_URLS)
    elif args.split == "val":
        urls = list(real_data.VAL_SHARD_URLS)
    else:
        urls = list(real_data.TRAIN_SHARD_URLS) + list(real_data.VAL_SHARD_URLS)

    print(f"measuring occupancy over {len(urls)} shard(s): {urls}", flush=True)
    counts = measure(urls)
    print(f"measured {counts.shape[0]} samples", flush=True)
    print(f"min={counts.min()} max={counts.max()} mean={counts.mean():.1f}", flush=True)

    stats = occupancy_percentiles(counts)
    print(f"occupancy_percentiles: {stats}", flush=True)

    l_vis_raw = choose_l_vis(counts, MASK_RATIO)
    q_pred = choose_q_pred(counts, MASK_RATIO, l_vis_raw)
    print(f"mask_ratio={MASK_RATIO}", flush=True)
    print(f"choose_l_vis -> l_vis_raw={l_vis_raw} (multiple of 32, not yet class-token-adjusted)", flush=True)
    print(f"choose_q_pred -> q_pred={q_pred}", flush=True)

    # `l_vis_raw` is a bare multiple of 32; with class_token=True,
    # encoder_seq_len = l_vis + 1 needs to be the tile multiple instead (see
    # layout.ShapeContract.__post_init__'s long comment -- individual l_vis
    # need not be tile-aligned, only l_vis+1 and l_vis+q_pred+1). Subtract 1,
    # exactly the adjustment memory_probe.py's full_size_contract() makes.
    l_vis = l_vis_raw - 1
    print(f"class-token-adjusted l_vis={l_vis}, q_pred={q_pred} (unchanged)", flush=True)
    print(f"encoder_seq_len={l_vis + 1}  decoder_seq_len={l_vis + q_pred + 1}", flush=True)

    contract = ShapeContract(
        batch=1,
        l_vis=l_vis,
        q_pred=q_pred,
        mask_ratio=MASK_RATIO,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
    )
    print(f"ShapeContract construction OK: {contract}", flush=True)

    out_path = "/tmp/claude-1000/-home-tt-small-projects-smri-fm/a104ecac-862a-465b-8958-98632556d7a3/scratchpad/real_valid_patch_counts.npy"
    np.save(out_path, counts)
    print(f"saved raw counts to {out_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
