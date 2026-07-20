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

Known simplifications (this is a pilot adapter, not a faithful
reproduction of the team's real shard-building pipeline):
- Volumes are center-cropped/padded to a fixed img_size with no physical
  resampling to a common voxel spacing, even though source volumes have
  very different native resolutions (some as small as 64x64x32, some as
  anisotropic as 512x20x512). Badly mismatched-shape volumes lose a lot
  of content this way; a real pipeline would resample to isotropic
  spacing first.
- Foreground mask is a plain intensity threshold (`image > 0`) combined
  with "not padding", not a real brain mask (no skull-stripping/SynthSeg
  step was applied when this .pt data was saved -- see
  PT902_FOMO300K_HF/dataset.json, crop_to_nonzero: false).
- Intensity scale varies wildly across raw files in this dataset (e.g.
  peak values from ~700 to ~7000 in an 8-file sample), so each volume is
  independently rescaled by a robust (99.5th percentile, foreground-only)
  intensity estimate. This is a per-volume normalization choice made here,
  not necessarily what the team's own pipeline does.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import data.mri_data as mri_data


def center_crop_or_pad(image: Tensor, target_shape: tuple[int, int, int]) -> tuple[Tensor, Tensor]:
    """image: [C, D, H, W]. Returns (resized_image, valid_mask), both
    [C, *target_shape]; valid_mask is True where content is real data (as
    opposed to padding introduced by this resize)."""
    C = image.shape[0]
    src_shape = image.shape[1:]
    out = image.new_zeros((C, *target_shape))
    valid = torch.zeros((C, *target_shape), dtype=torch.bool)

    src_slices, dst_slices = [], []
    for s, t in zip(src_shape, target_shape):
        if s >= t:
            start = (s - t) // 2
            src_slices.append(slice(start, start + t))
            dst_slices.append(slice(0, t))
        else:
            start = (t - s) // 2
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
        img_size: tuple[int, int, int],
        in_chans: int = 1,
        intensity_percentile: float = 99.5,
        intensity_sample_size: int = 200_000,
    ):
        self.paths = paths
        self.img_size = tuple(int(d) for d in img_size)
        self.in_chans = in_chans
        self.intensity_percentile = intensity_percentile
        self.intensity_sample_size = intensity_sample_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx % len(self.paths)]
        image = torch.load(path, map_location="cpu", weights_only=False).float()

        if image.shape[0] != self.in_chans:
            image = image[:1]
            if self.in_chans > 1:
                image = image.expand(self.in_chans, *image.shape[1:]).contiguous()

        image, valid = center_crop_or_pad(image, self.img_size)
        img_mask = valid & (image > 0)

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
    experiments/tome_pretrain/config.yaml)."""
    data_loaders = {}
    dataset_names = [args.train_dataset] + list(args.eval_datasets)

    for dataset_name in dataset_names:
        dataset_config = args.datasets[dataset_name]
        is_train = dataset_name == args.train_dataset

        paths = load_paths(dataset_config["paths"])
        dataset = LocalPtDataset(
            paths,
            img_size=args.img_size,
            in_chans=int(args.get("in_chans", 1)),
        )

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
