import numpy as np
import torch

from data.mri_data import densify_sparse_image_batch, make_sparse_wds_dataset
from smri_mae.modules import patchify3d
from smri_mae_tt import real_data
from smri_mae_tt.layout import ShapeContract
from smri_mae_tt.packing import compute_valid_patch_counts, pack_batch, round_trip_check

_REAL_IMG_SIZE = (208, 240, 208)  # actual FOMO_with_dwi shard volume shape


def _contract(l_vis=32, q_pred=32):
    # A small cubic volume, divisible by patch_size, keeps the test fast while
    # exercising the same patchify3d/densify path the real pipeline uses.
    return ShapeContract(
        batch=3,
        l_vis=l_vis,
        q_pred=q_pred,
        mask_ratio=0.8,
        img_size=(64, 64, 64),
        patch_size=8,
        # These tests only exercise host-side numpy packing (packing.py),
        # never the on-device encoder/decoder sequence, so they're
        # independent of the CLS-token-composition question that
        # encoder_seq_len/decoder_seq_len tile-alignment validation cares
        # about. class_token=False keeps l_vis/q_pred themselves the whole
        # story here.
        class_token=False,
    )


def _real_crops(n):
    """The first `n` real samples from the local FOMO_with_dwi corpus
    (deterministic: shuffle=False, so this is always the same `n` samples in
    the same order), cropped to a central `_contract().img_size` region.

    Central crops of real brain scans are effectively 100% occupied (every
    voxel is inside brain tissue, no background border) -- confirmed
    empirically before writing this helper -- so every one of the crop's
    (64/8)**3=512 patches is available as a candidate for `_real_sparse_batch`
    below to select from.
    """
    dataset = make_sparse_wds_dataset([real_data.TRAIN_SHARD_URLS[0]], shuffle=False, buffer_size=4)
    it = iter(dataset)
    samples = [next(it) for _ in range(n)]
    image_values = torch.cat([torch.as_tensor(s["image_values"], dtype=torch.float32) for s in samples])
    packed_img_mask = torch.stack([torch.as_tensor(s["img_mask"]) for s in samples])
    images, masks = densify_sparse_image_batch(image_values, packed_img_mask, _REAL_IMG_SIZE, dtype=torch.float32)

    T, H, W = _contract().img_size
    Ht, Hh, Hw = images.shape[1:]
    t0, h0, w0 = (Ht - T) // 2, (Hh - H) // 2, (Hw - W) // 2
    return images[:, t0 : t0 + T, h0 : h0 + H, w0 : w0 + W], masks[:, t0 : t0 + T, h0 : h0 + H, w0 : w0 + W]


def _real_sparse_batch(rng, contract, valid_patches_per_sample):
    """Build a sparse batch where each sample has exactly
    `valid_patches_per_sample[i]` valid (whole) patches, at random grid
    positions within a real MRI crop -- real voxel intensities and real
    occupancy geometry throughout, no fabricated content anywhere. Patch-level
    validity is controlled exactly (which of the crop's naturally-valid
    patches get kept), the same edge-case control the old synthetic version
    gave, just applied to real data instead of `rng.standard_normal` noise.
    """
    grid = contract.grid_size
    patch = contract.patch_size
    T, H, W = contract.img_size
    n = len(valid_patches_per_sample)

    images, masks = _real_crops(n)
    if images.shape[1:] != (T, H, W):
        raise ValueError(f"_real_crops shape {tuple(images.shape[1:])} != contract.img_size {(T, H, W)}")

    # "Fully valid" per patch (all patch_dim voxels present), same patch id
    # ordering pack_batch/round_trip_check use.
    fully_valid = patchify3d(masks.unsqueeze(1).float(), (patch,) * 3).bool().all(dim=-1).numpy()  # [n, N]

    values_list = []
    mask_bytes = []
    for i, k in enumerate(valid_patches_per_sample):
        eligible = np.flatnonzero(fully_valid[i])
        if eligible.shape[0] < k:
            raise RuntimeError(
                f"real crop only has {eligible.shape[0]} fully-valid patches, need {k} -- "
                "widen _real_crops' window or pick a different real sample"
            )
        chosen = rng.choice(eligible, size=k, replace=False)
        patch_mask = np.zeros(int(np.prod(grid)), dtype=bool)
        patch_mask[chosen] = True
        patch_mask = patch_mask.reshape(grid)
        voxel_mask = np.kron(patch_mask, np.ones((patch, patch, patch), dtype=bool))
        assert voxel_mask.shape == (T, H, W)

        values = images[i].numpy()
        values_list.append(values[voxel_mask])
        mask_bytes.append(np.packbits(voxel_mask.reshape(-1)))

    image_values = torch.from_numpy(np.concatenate(values_list)).float()
    packed_img_mask = torch.from_numpy(np.stack(mask_bytes)).to(torch.uint8)
    return image_values, packed_img_mask


def test_pack_batch_shapes_and_round_trip():
    rng = np.random.default_rng(0)
    contract = _contract()
    image_values, packed_img_mask = _real_sparse_batch(rng, contract, [100, 120, 90])

    packed = pack_batch(image_values, packed_img_mask, contract, np.random.default_rng(1))

    assert packed.visible_values.shape == (contract.batch, contract.l_vis, contract.patch_dim)
    assert packed.visible_ids.shape == (contract.batch, contract.l_vis)
    assert packed.pred_targets.shape == (contract.batch, contract.q_pred, contract.patch_dim)
    assert packed.pred_voxel_mask.shape == (contract.batch, contract.q_pred, contract.patch_dim)
    assert packed.replaced_samples == 0  # 100/120/90 valid patches easily covers 32+32
    assert np.all(packed.pred_voxel_mask)  # every sampled pred patch is fully valid by construction

    assert round_trip_check(image_values, packed_img_mask, contract, packed)


def test_visible_and_pred_ids_disjoint_per_sample():
    rng = np.random.default_rng(0)
    contract = _contract()
    image_values, packed_img_mask = _real_sparse_batch(rng, contract, [100, 120, 90])
    packed = pack_batch(image_values, packed_img_mask, contract, np.random.default_rng(2))

    for b in range(contract.batch):
        assert set(packed.visible_ids[b].tolist()).isdisjoint(packed.pred_ids[b].tolist())


def test_compute_valid_patch_counts_matches_construction():
    rng = np.random.default_rng(0)
    contract = _contract()
    target_counts = [100, 150, 60]
    _, packed_img_mask = _real_sparse_batch(rng, contract, target_counts)

    counts = compute_valid_patch_counts(packed_img_mask, contract.img_size, contract.patch_size)
    assert counts.tolist() == target_counts


def test_replacement_needed_when_occupancy_too_low():
    rng = np.random.default_rng(0)
    contract = _contract(l_vis=32, q_pred=32)
    # 40 valid patches: enough for the 32 visible, but only 8 remain for the
    # 32 prediction queries, so pred sampling must fall back to replacement.
    image_values, packed_img_mask = _real_sparse_batch(rng, contract, [40, 40, 40])

    packed = pack_batch(image_values, packed_img_mask, contract, np.random.default_rng(3))
    assert packed.replaced_samples == 3
