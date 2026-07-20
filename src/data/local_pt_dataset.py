"""Local .pt-file adapter for MAE pretraining, standing in for real webdataset
shards (see experiments/tome_pretrain/README.md for why).

main_pretrain.py's train/eval loop unconditionally reconstructs dense images
via `mri_data.densify_sparse_image_batch(batch["image_values"],
batch["img_mask"], ...)`, so instead of touching that loop, this dataset
packs each sample into the exact same sparse format
(`mri_data.extract_sparse_wds_sample`'s output shape: `image_values`
sparse float16 values at masked voxels, `img_mask` bit-packed uint8) and
reuses `mri_data.collate` unchanged. The sparse packing itself is just a
webdataset transport optimization for streaming from cloud storage across
many nodes -- irrelevant on local disk with one GPU, but replicating the
*format* here means create_data_loaders only needs one small branch instead
of forking the whole training loop.

v3: per-axis crop-or-pad, using ~all 965 files. History:
- v1 resized every volume to the full pretraining img_size (208x240x208)
  regardless of native shape, which for most of this dataset meant mostly
  zero-padding -- 82% of a 60-file sample filled <30% of that canvas, every
  DWI/perfusion file (139/965) filled 0% (those are thin-slice sequences,
  not full anatomical volumes). Confirmed visually: the wandb reconstruction
  plot showed sagittal/coronal views that were almost entirely black.
- v2 fixed this by filtering to only volumes with every spatial dim >=
  crop_size, cropping the rest at random (asparagus's own approach). That
  guarantees zero padding, but drops the dataset to 131/965 files -- far
  fewer than the ~1000-scan subset the checkpoints this is meant to be
  comparable to were pretrained on.
- v3 (this version): crop each axis independently -- take a real crop on
  any axis that's already >= crop_size, and pad *only* the axes that are
  actually too small, instead of either resizing everything or discarding
  anything not uniformly big enough. E.g. a [17, 512, 512] DWI file gets a
  fully-real random 160x160 crop on its two large axes and padding on just
  the one 17-voxel axis, rather than being excluded outright (v2) or having
  every axis mangled by a single whole-volume resize (v1). This keeps ~965
  files usable while still fixing v1's bug, since the padding is now
  minimized to only the axes that need it, not applied uniformly.

Whatever padding remains (e.g. the thin axis above) is never a
training-*correctness* problem regardless of version: the model already
excludes padded/invalid voxels from both context and loss via img_mask (see
MaskedAutoencoderViT.prepare_masks). v1's problem was wasting most of the
compute budget on patches that contributed nothing, not corrupting
training with fake targets.

Other known simplifications (still true in v3):
- Foreground mask is a plain intensity threshold (`image > 0`), not a real
  brain mask (no skull-stripping/SynthSeg step was applied when this .pt
  data was saved -- see PT902_FOMO300K_HF/dataset.json, crop_to_nonzero:
  false).
- Intensity scale varies wildly across raw files in this dataset (e.g. peak
  values from ~700 to ~7000 in an 8-file sample), so each volume is
  independently rescaled by a robust (99.5th percentile, foreground-only)
  intensity estimate. This is a per-volume normalization choice made here,
  not necessarily what the team's own pipeline does.
"""

import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import data.mri_data as mri_data


def crop_or_pad(image: Tensor, crop_size: tuple[int, int, int], train: bool, force_random: bool = False) -> tuple[Tensor, Tensor]:
    """image: [C, D, H, W]. Per spatial axis: if the source is already >=
    crop_size, take a real crop (random offset if train, else centered);
    if it's smaller, center-pad just that axis. Returns (image, valid_mask),
    both [C, *crop_size]; valid_mask is True where content is real (as
    opposed to padding introduced on an undersized axis)."""
    C = image.shape[0]
    src_shape = image.shape[1:]
    out = image.new_zeros((C, *crop_size))
    valid = torch.zeros((C, *crop_size), dtype=torch.bool)

    random_offset = train or force_random
    src_slices, dst_slices = [], []
    for s, c in zip(src_shape, crop_size):
        if s >= c:
            max_start = s - c
            start = int(torch.randint(0, max_start + 1, (1,)).item()) if (random_offset and max_start > 0) else max_start // 2
            src_slices.append(slice(start, start + c))
            dst_slices.append(slice(0, c))
        else:
            start = (c - s) // 2
            src_slices.append(slice(0, s))
            dst_slices.append(slice(start, start + s))

    out[(slice(None), *dst_slices)] = image[(slice(None), *src_slices)]
    valid[(slice(None), *dst_slices)] = True
    return out, valid


def pack_sparse_sample(image: Tensor, img_mask: Tensor, meta: dict) -> dict:
    """image, img_mask: [C, D, H, W] (img_mask already applied: image is 0
    outside img_mask). Matches mri_data.extract_sparse_wds_sample's output
    shape exactly, so mri_data.collate/densify_sparse_image_batch need no
    changes."""
    mask_np = img_mask.numpy().astype(bool)
    image_values = image.numpy()[mask_np].astype(np.float16)
    packed_mask = np.packbits(mask_np.reshape(-1), bitorder="big")
    return {"image_values": image_values, "img_mask": packed_mask, "meta": meta}


class LocalPtDataset(Dataset):
    """Reads asparagus-format .pt tensors directly (no webdataset shards)."""

    def __init__(
        self,
        paths: list[str],
        crop_size: tuple[int, int, int],
        in_chans: int = 1,
        train: bool = True,
        intensity_percentile: float = 99.5,
        intensity_sample_size: int = 200_000,
    ):
        self.paths = paths
        self.crop_size = tuple(int(d) for d in crop_size)
        self.in_chans = in_chans
        self.train = train
        self.intensity_percentile = intensity_percentile
        self.intensity_sample_size = intensity_sample_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx % len(self.paths)]
        image_full = torch.load(path, map_location="cpu", weights_only=False).float()

        if image_full.shape[0] != self.in_chans:
            image_full = image_full[:1]
            if self.in_chans > 1:
                image_full = image_full.expand(self.in_chans, *image_full.shape[1:]).contiguous()

        # A crop is picked at a uniformly random offset regardless of where a
        # volume's real content is, so for scans with a small content
        # footprint inside a large field of view (common in the DWI/perfusion
        # files) an unlucky draw can land entirely on background -- an
        # all-invalid img_mask crashes training (forward_loss divides by the
        # per-sample valid-voxel count). Retry a few times with fresh random
        # offsets first (regardless of train/eval) before giving up.
        for attempt in range(8):
            image, valid = crop_or_pad(image_full, self.crop_size, self.train, force_random=attempt > 0)
            img_mask = valid & (image > 0)
            if img_mask.any():
                break

        fg = image[img_mask]
        if fg.numel() > 0:
            if fg.numel() > self.intensity_sample_size:
                idx_sample = torch.randint(0, fg.numel(), (self.intensity_sample_size,))
                fg_sample = fg[idx_sample]
            else:
                fg_sample = fg
            scale = torch.quantile(fg_sample, self.intensity_percentile / 100.0).clamp_min(1e-3)
            image = image / scale

        image = image.masked_fill(~img_mask, 0)
        return pack_sparse_sample(image, img_mask, meta={"path": path})


def filter_usable_paths(paths: list[str], crop_size: tuple[int, int, int]) -> list[str]:
    """Every spatial dim must be >= crop_size for a fully-real (unpadded) crop.

    Not used by default (LocalPtDataset now handles undersized axes via
    per-axis padding instead of exclusion, see crop_or_pad) -- kept as an
    opt-in for a stricter "zero padding ever" mode via local_pt_source, if
    a future run wants to trade data quantity for that guarantee again.
    """
    kept = []
    for p in paths:
        shape = torch.load(p, map_location="cpu", weights_only=False).shape[1:]
        if all(d >= c for d, c in zip(shape, crop_size)):
            kept.append(p)
    return kept


def split_train_val(paths: list[str], val_fraction: float, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministic shuffle-and-split of an already-filtered pool."""
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    n_val = max(1, round(len(paths) * val_fraction))
    return paths[n_val:], paths[:n_val]


def load_paths(spec: str) -> list[str]:
    """spec: path to a JSON file containing either a flat list of .pt paths,
    or an asparagus-style {"train": [...], "val": [...]} split dict combined
    with a #train/#val suffix, e.g. "split_99_01_00.json#train" (fold 0)."""
    file_path, _, key = spec.partition("#")
    data = json.loads(Path(file_path).read_text())
    if key:
        if isinstance(data, list):  # list of per-fold dicts, use fold 0
            data = data[0]
        return data[key]
    return data


def create_local_data_loaders(args):
    """Drop-in analogue of main_pretrain.create_data_loaders, backed by
    LocalPtDataset instead of webdataset shards. Selected per-dataset via
    datasets.<name>.format == "local_pt" in the config (see
    experiments/tome_pretrain/config.yaml). Crop size == args.img_size.

    If args.local_pt_source is set, train/val paths are both drawn from that
    one shared pool: filtered for crop-size fit (filter_usable_paths) once,
    then split ourselves (args.local_pt_val_fraction, default 0.1) -- an
    opt-in stricter mode, see filter_usable_paths' docstring. Otherwise (the
    default) falls back to each dataset's own `paths` spec as-is, with
    LocalPtDataset padding undersized axes per-sample instead of excluding
    files."""
    data_loaders = {}
    dataset_names = [args.train_dataset] + list(args.eval_datasets)

    path_map = None
    shared_source = args.get("local_pt_source")
    if shared_source:
        crop_size = tuple(int(d) for d in args.img_size)
        pool = load_paths(shared_source)
        usable = filter_usable_paths(pool, crop_size)
        val_fraction = float(args.get("local_pt_val_fraction", 0.1))
        train_paths, val_paths = split_train_val(
            usable, val_fraction, seed=int(args.get("seed", 0)) if args.get("seed") else 0
        )
        print(
            f"local_pt_source: {len(usable)}/{len(pool)} usable at crop_size={crop_size} "
            f"-> {len(train_paths)} train / {len(val_paths)} val"
        )
        path_map = {args.train_dataset: train_paths}
        for n in args.eval_datasets:
            path_map[n] = val_paths

    for dataset_name in dataset_names:
        dataset_config = args.datasets[dataset_name]
        is_train = dataset_name == args.train_dataset

        paths = path_map[dataset_name] if path_map else load_paths(dataset_config["paths"])
        dataset = LocalPtDataset(
            paths,
            crop_size=args.img_size,
            in_chans=int(args.get("in_chans", 1)),
            train=is_train,
        )
        print(f"{dataset_name}: {len(dataset)} files, crop_size={args.img_size}")

        num_workers = int(args.num_workers)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=bool(dataset_config.get("shuffle", is_train)),
            collate_fn=partial(mri_data.collate, include_meta=not is_train),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            drop_last=bool(dataset_config.get("drop_last", True)),
            prefetch_factor=int(args.prefetch_factor) if num_workers > 0 else None,
        )
        data_loaders[dataset_name] = loader

    train_loader = data_loaders.pop(args.train_dataset)
    return train_loader, data_loaders
