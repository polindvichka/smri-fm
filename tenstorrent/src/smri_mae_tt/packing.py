"""Host-side: turn a sparse shard sample (image_values + bit-packed img_mask)
into the packed, statically-shaped patch tensors the TT-NN model consumes.

No device work happens here -- this is pure numpy/CPU-torch control flow
meant to run in dataloader workers, overlapped with device compute. See
TENSTORRENT_PORT.md section 3 ("The data path should never send a dense
volume") and section 1 for the occupancy measurement this feeds into.

The dense reconstruction (`densify_sparse_image_batch`) still happens here,
on CPU -- that's cheap (single sample is ~10 MB). What we avoid is shipping
the dense volume over PCIe: only the packed patch tensors below cross the
host/device boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data.mri_data import densify_sparse_image_batch, unpack_img_mask_batch
from smri_mae.modules import patchify3d

from .layout import ShapeContract


@dataclass
class PackedBatch:
    visible_values: np.ndarray  # [B, L_vis, patch_dim] float32
    visible_ids: np.ndarray  # [B, L_vis] uint32
    pred_targets: np.ndarray  # [B, Q_pred, patch_dim] float32
    pred_voxel_mask: np.ndarray  # [B, Q_pred, patch_dim] bool
    pred_ids: np.ndarray  # [B, Q_pred] uint32
    replaced_samples: int  # samples in this batch that needed sample-with-replacement


def pack_batch(
    image_values: torch.Tensor,
    packed_img_mask: torch.Tensor,
    contract: ShapeContract,
    rng: np.random.Generator,
) -> PackedBatch:
    """Sample exactly `contract.l_vis` visible and `contract.q_pred` prediction
    patches per sample from the sparse batch, and gather their values/validity.

    image_values, packed_img_mask: as produced by `data.mri_data.collate`
    (concatenated brain-voxel values and bit-packed masks for the batch).
    """
    images, masks = densify_sparse_image_batch(image_values, packed_img_mask, contract.img_size)
    images = images.unsqueeze(1)  # [B, 1, T, H, W]
    masks = masks.unsqueeze(1)

    patch_size = (contract.patch_size,) * 3
    value_patches = patchify3d(images.float(), patch_size)  # [B, N, P]
    mask_patches = patchify3d(masks.float(), patch_size).bool()  # [B, N, P]
    valid = mask_patches.any(dim=-1).numpy()  # [B, N]

    B = valid.shape[0]
    visible_ids = np.zeros((B, contract.l_vis), dtype=np.uint32)
    pred_ids = np.zeros((B, contract.q_pred), dtype=np.uint32)
    replaced_samples = 0
    for b in range(B):
        valid_idx = np.flatnonzero(valid[b])
        vis, vis_replaced = _sample_ids(valid_idx, contract.l_vis, rng)
        remaining_idx = np.setdiff1d(valid_idx, vis, assume_unique=False)
        pred, pred_replaced = _sample_ids(remaining_idx, contract.q_pred, rng)
        visible_ids[b] = vis
        pred_ids[b] = pred
        if vis_replaced or pred_replaced:
            replaced_samples += 1

    value_patches_np = value_patches.numpy()
    mask_patches_np = mask_patches.numpy()
    b_idx = np.arange(B)[:, None]

    return PackedBatch(
        visible_values=value_patches_np[b_idx, visible_ids],
        visible_ids=visible_ids,
        pred_targets=value_patches_np[b_idx, pred_ids],
        pred_voxel_mask=mask_patches_np[b_idx, pred_ids],
        pred_ids=pred_ids,
        replaced_samples=replaced_samples,
    )


def _sample_ids(
    valid_idx: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    """Sample k ids from valid_idx without replacement if there are enough,
    else with replacement. Returns (ids, needed_replacement).
    """
    n = valid_idx.shape[0]
    if k == 0:
        return np.zeros(0, dtype=np.uint32), False
    if n == 0:
        raise ValueError("sample has zero valid patches; cannot sample any ids from it")
    if n >= k:
        return rng.choice(valid_idx, size=k, replace=False).astype(np.uint32), False
    return rng.choice(valid_idx, size=k, replace=True).astype(np.uint32), True


def compute_valid_patch_counts(
    packed_img_mask: torch.Tensor,
    img_size: tuple[int, int, int],
    patch_size: int,
) -> np.ndarray:
    """Per-sample count of valid (any-voxel-present) patches.

    Needs only the mask, not the dense values -- this is what
    `layout.choose_l_vis` / `choose_q_pred` are fit against; run it over real
    shards to produce the M1 occupancy report (see TENSTORRENT_PORT.md section 1).
    """
    masks = unpack_img_mask_batch(packed_img_mask, img_size).unsqueeze(1)
    mask_patches = patchify3d(masks.float(), (patch_size,) * 3).bool()
    valid = mask_patches.any(dim=-1)
    return valid.sum(dim=1).numpy()


def round_trip_check(
    image_values: torch.Tensor,
    packed_img_mask: torch.Tensor,
    contract: ShapeContract,
    packed: PackedBatch,
) -> bool:
    """Bit-exact check that packed visible/pred values match the dense
    patchification at the sampled ids -- the M1 gate in TENSTORRENT_PORT.md.
    """
    images, masks = densify_sparse_image_batch(image_values, packed_img_mask, contract.img_size)
    images = images.unsqueeze(1)
    patch_size = (contract.patch_size,) * 3
    value_patches = patchify3d(images.float(), patch_size).numpy()
    b_idx = np.arange(value_patches.shape[0])[:, None]
    vis_ref = value_patches[b_idx, packed.visible_ids]
    pred_ref = value_patches[b_idx, packed.pred_ids]
    return bool(np.array_equal(vis_ref, packed.visible_values)) and bool(
        np.array_equal(pred_ref, packed.pred_targets)
    )
