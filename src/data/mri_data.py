import math
from functools import partial
from glob import glob
from typing import Sequence

import braceexpand
import numpy as np
import torch
import webdataset as wds
from torch import Tensor


def collate(
    samples: list[dict],
    *,
    include_meta: bool = True,
) -> dict[str, Tensor]:
    masks = [torch.as_tensor(sample["img_mask"]) for sample in samples]
    batch = {"img_mask": torch.stack(masks)}
    image_values = [
        torch.as_tensor(sample["image_values"], dtype=torch.float16) for sample in samples
    ]
    batch["image_values"] = torch.cat(image_values)

    if include_meta:
        batch["meta"] = [sample["meta"] for sample in samples]
    return batch


def unpack_img_mask_batch(mask: torch.Tensor, image_shape: Sequence[int]) -> torch.Tensor:
    """Return a dense boolean batch mask from bit-packed mask tensors."""
    image_shape = tuple(int(dim) for dim in image_shape)
    mask_numel = math.prod(image_shape)
    packed_numel = math.ceil(mask_numel / 8)
    if mask.dtype != torch.uint8:
        raise ValueError(f"packed img_mask must have dtype uint8, got {mask.dtype}")
    if mask.ndim != 2 or mask.shape[1] != packed_numel:
        raise ValueError(f"expected packed img_mask shape (B, {packed_numel}), got {tuple(mask.shape)}")

    shifts = torch.arange(7, -1, -1, device=mask.device, dtype=torch.uint8)
    bits = (mask.unsqueeze(-1).bitwise_right_shift(shifts) & 1).reshape(mask.shape[0], -1)
    return bits[:, :mask_numel].reshape((mask.shape[0], *image_shape)).bool()


def densify_sparse_image_batch(
    image_values: torch.Tensor,
    packed_img_mask: torch.Tensor,
    image_shape: Sequence[int],
    *,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct dense images from concatenated brain-voxel values and packed masks."""
    image_shape = tuple(int(dim) for dim in image_shape)
    dtype = dtype or image_values.dtype
    masks = unpack_img_mask_batch(packed_img_mask, image_shape)
    batch_size = masks.shape[0]

    images = torch.zeros(
        (batch_size, *image_shape),
        device=image_values.device,
        dtype=dtype,
    )
    images[masks] = image_values.to(dtype=dtype)
    return images, masks


def expand_urls(urls: str | list[str]) -> list[str]:
    """
    Expand wds urls:

    - expand glob patterns
    - expand brace expressions
    - filter files that don't exist

    Adapted from `webdataset.shardlists.expand_urls`.
    """
    if isinstance(urls, str):
        urls = [urls]
    results = []
    for url in urls:
        chars = set(url)
        if chars.intersection("[*?"):
            result = sorted(glob(url))
        elif "{" in chars:
            result = braceexpand.braceexpand(url)
        else:
            result = [url]
        results.extend(result)
    return results



def warn_and_continue(exn):
    print(f"WARNING {repr(exn)}")
    return True


def extract_sparse_wds_sample(sample: dict) -> dict:
    image_values = np.asarray(sample["image_values.npy"], dtype=np.float16)
    img_mask = np.asarray(sample["img_mask.npy"], dtype=np.uint8)

    # img_mask is bit-packed (one bit per voxel); the number of set bits must
    # match image_values' length, or densify_sparse_image_batch's boolean-mask
    # assignment crashes downstream -- past the point where WebDataset's
    # per-sample handler=warn_and_continue can catch it (that only wraps this
    # extraction step, not later batch-level ops in the training loop). Some
    # real samples in the dataset have a mismatched sparse encoding (seen in
    # practice: vitl_rope_targetnorm_50k crashed on one at epoch 77 with a
    # value tensor of shape [23765142] vs an indexing result of [23765018]);
    # catching the mismatch here turns a fatal training crash into a graceful
    # per-sample skip instead, same as any other malformed sample.
    popcount = int(np.unpackbits(img_mask).sum())
    if popcount != image_values.shape[0]:
        raise ValueError(
            f"sparse sample shape mismatch: img_mask has {popcount} set voxels "
            f"but image_values has {image_values.shape[0]} values "
            f"(meta={sample.get('meta.json')!r})"
        )

    return {
        "image_values": image_values,
        "img_mask": img_mask,
        "meta": sample["meta.json"],
    }


def make_sparse_wds_dataset(
    url: str | list[str],
    *,
    shuffle: bool,
    buffer_size: int,
) -> wds.WebDataset:
    dataset = wds.WebDataset(
        expand_urls(url),
        handler=warn_and_continue,
        resampled=shuffle,
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
    )
    dataset = dataset.decode().map(extract_sparse_wds_sample, handler=warn_and_continue)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    return dataset
