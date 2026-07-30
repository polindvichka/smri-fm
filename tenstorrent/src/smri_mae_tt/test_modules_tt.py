"""Parity test for `modules_tt.py`'s encoder/decoder against the PyTorch
reference (`src/smri_mae/modules.py`, `src/smri_mae/model_mae.py`).

No trained checkpoint exists anywhere in this repo (checked), so this
validates correctness against the PyTorch reference using **shared random
weights**: one `numpy.random.Generator` produces the encoder/decoder
parameter arrays (`modules_tt.init_encoder_params`/`init_decoder_params`),
which are loaded both into a `modules_tt.Encoder`/`Decoder` (device) and
into a matching small-scale `smri_mae.model_mae.MaskedEncoder`/
`MaskedDecoder` (CPU torch) via `.copy_()`. Both then run on the exact same
patch values / position ids, and outputs are compared by Pearson
correlation (PCC), gated at >= 0.999 -- the M2/M3 gate in
TENSTORRENT_PORT.md section 7.

The shared config (dims, seeds, weight-copy/dense-SDPA-reference plumbing,
and the `compare()` PCC harness) now lives in `parity.py`
(TENSTORRENT_PORT.md section 7's "frozen fixture" + "Build this at M2 and
keep it" instruction) -- this file imports it rather than defining its own
copy. See `parity.FrozenFixture`'s docstring for exactly why
`embed_dim=64, num_heads=2` / `decoder_embed_dim=32, decoder_num_heads=1`
(head_dim=32 on both sides) and `l_vis=31, q_pred=32` were chosen.

This test never touches `packing.py`: patch "values" are literally
`patchify3d` applied to a random volume (reusing the reference's own
patchify, imported directly, for a realistic patch_dim/grid_size), and
`visible_ids`/`pred_ids` are host-sampled distinct patch indices -- there is
no ragged batching to worry about at these fixed sizes (identical to the
static-shape contract this whole port is built around), so `MaskedEncoder.
forward_visible_ids` and `MaskedDecoder.forward` (called directly, no
masking) are the exact right reference entry points, per
TENSTORRENT_PORT.md's own guidance on which reference methods this port
matches.

`pytest.importorskip("ttml")` is the one legitimate skip here (environment
availability), matching `ops_tt/test_attention.py`'s pattern -- no other
skip/try-except is used.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae.modules import Patchify3D, SinCosPosEmbed3D, get_3d_sincos_pos_embed, patchify3d  # noqa: E402
from smri_mae.model_mae import MaskedDecoder, MaskedEncoder  # noqa: E402

from smri_mae_tt import modules_tt as M  # noqa: E402
from smri_mae_tt.layout import ShapeContract  # noqa: E402
from smri_mae_tt.loss_tt import per_scan_valid_voxel_mse_loss, pred_targets_to_tensor, voxel_mask_to_tensor  # noqa: E402
from smri_mae_tt.parity import (  # noqa: E402
    compare,
    dense_decoder_forward as _dense_decoder_forward,
    dense_encoder_forward_visible_ids as _dense_encoder_forward_visible_ids,
    load_decoder_into_ref as _load_decoder_into_ref,
    load_encoder_into_ref as _load_encoder_into_ref,
    numpy_reference_forward_loss as _numpy_reference_forward_loss,
    to_numpy_fp32 as _to_numpy_fp32,
)
from smri_mae_tt.parity import FROZEN_FIXTURE

SEED = FROZEN_FIXTURE.param_seed
PCC_GATE = 0.999


def _contract() -> ShapeContract:
    return FROZEN_FIXTURE.contract


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_sincos_pos_embed_matches_reference():
    """`modules_tt.sincos_pos_embed_3d` is a verbatim port of
    `get_3d_sincos_pos_embed`/`get_1d_sincos_pos_embed_from_grid`
    (`uniform_power=True, cls_token=False` specialization) -- check the
    numpy tables are bit-for-bit the same before trusting anything built on
    top of it. Pure host-side numpy; no device needed.
    """
    contract = _contract()
    grid = contract.grid_size  # (T, H, W)
    ours = M.sincos_pos_embed_3d(contract.embed_dim, grid)
    reference = get_3d_sincos_pos_embed(
        embed_dim=contract.embed_dim,
        grid_size=(grid[1], grid[2]),
        grid_depth=grid[0],
        uniform_power=True,
    )
    assert ours.shape == reference.shape
    np.testing.assert_array_equal(ours, reference.astype(np.float32))


def test_encoder_decoder_forward_parity():
    contract = _contract()
    encoder_depth = 2
    decoder_depth = 1

    param_rng = np.random.default_rng(SEED)
    enc_params = M.init_encoder_params(param_rng, contract, depth=encoder_depth)
    dec_params = M.init_decoder_params(param_rng, contract, depth=decoder_depth)

    data_rng = np.random.default_rng(SEED + 1)
    batch = contract.batch
    num_patches = contract.num_patches
    if contract.l_vis + contract.q_pred > num_patches:
        raise ValueError(
            f"test config needs l_vis+q_pred={contract.l_vis + contract.q_pred} distinct patch ids "
            f"but grid only has num_patches={num_patches}"
        )

    vol_np = data_rng.standard_normal((batch, contract.in_chans, *contract.img_size)).astype(np.float32)
    vol_t = torch.from_numpy(vol_np)
    all_patches = patchify3d(vol_t, (contract.patch_size,) * 3).numpy()  # [B, N, patch_dim]
    assert all_patches.shape == (batch, num_patches, contract.patch_dim)

    visible_ids = np.zeros((batch, contract.l_vis), dtype=np.int64)
    pred_ids = np.zeros((batch, contract.q_pred), dtype=np.int64)
    for b in range(batch):
        perm = data_rng.permutation(num_patches)
        visible_ids[b] = perm[: contract.l_vis]
        pred_ids[b] = perm[contract.l_vis : contract.l_vis + contract.q_pred]

    b_idx = np.arange(batch)[:, None]
    visible_values = all_patches[b_idx, visible_ids]  # [B, L_vis, patch_dim]

    # --- Build + load the PyTorch reference ---
    patchify_ref = Patchify3D(contract.img_size, (contract.patch_size,) * 3, in_chans=contract.in_chans)
    patch_embed_ref = nn.Linear(patchify_ref.patch_dim, contract.embed_dim)
    pos_embed_ref = SinCosPosEmbed3D(contract.embed_dim, patchify_ref.grid_size)
    ref_encoder = MaskedEncoder(
        patchify=patchify_ref,
        patch_embed=patch_embed_ref,
        pos_embed=pos_embed_ref,
        depth=encoder_depth,
        embed_dim=contract.embed_dim,
        num_heads=contract.encoder_heads,
        qkv_bias=True,
        proj_bias=True,
        mlp_ratio=4,
        class_token=True,
        reg_tokens=0,
        no_embed_class=False,
        drop_path_rate=0.0,
    )
    _load_encoder_into_ref(ref_encoder, enc_params)
    ref_encoder.eval()

    decoder_pos_embed_ref = SinCosPosEmbed3D(contract.decoder_embed_dim, patchify_ref.grid_size)
    decoder_head_ref = nn.Linear(contract.decoder_embed_dim, patchify_ref.patch_dim)
    ref_decoder = MaskedDecoder(
        pos_embed=decoder_pos_embed_ref,
        head=decoder_head_ref,
        input_dim=contract.embed_dim,
        depth=decoder_depth,
        embed_dim=contract.decoder_embed_dim,
        num_heads=contract.decoder_heads,
        qkv_bias=True,
        proj_bias=True,
        mlp_ratio=4,
        class_token=True,
        no_embed_class=False,
    )
    _load_decoder_into_ref(ref_decoder, dec_params)
    ref_decoder.eval()

    with torch.no_grad():
        cls_ref, reg_ref, patch_ref = _dense_encoder_forward_visible_ids(
            ref_encoder, vol_t, torch.from_numpy(visible_ids)
        )
        assert reg_ref is None  # reg_tokens=0
        pred_ref = _dense_decoder_forward(
            ref_decoder,
            patch_ref,
            torch.from_numpy(visible_ids),
            torch.from_numpy(pred_ids),
        )

    # --- Build + load the TT modules, run on device ---
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        device = ctx.get_device()

        enc = M.Encoder(contract, depth=encoder_depth, params=enc_params)
        dec = M.Decoder(contract, depth=decoder_depth, params=dec_params)

        cls_tt, patch_tt = enc.forward(visible_values, visible_ids.astype(np.uint32))
        ttnn.synchronize_device(device)
        pred_tt = dec.forward(patch_tt, visible_ids.astype(np.uint32), pred_ids.astype(np.uint32))
        ttnn.synchronize_device(device)

        cls_tt_np = _to_numpy_fp32(cls_tt)
        patch_tt_np = _to_numpy_fp32(patch_tt)
        pred_tt_np = _to_numpy_fp32(pred_tt)
    finally:
        ctx.close_device()

    cls_ref_np = cls_ref.numpy()
    patch_ref_np = patch_ref.numpy()
    pred_ref_np = pred_ref.numpy()

    assert cls_tt_np.reshape(cls_ref_np.shape).shape == cls_ref_np.shape
    assert patch_tt_np.reshape(patch_ref_np.shape).shape == patch_ref_np.shape
    assert pred_tt_np.reshape(pred_ref_np.shape).shape == pred_ref_np.shape

    compare("cls_embed (M2 gate)", cls_ref_np, cls_tt_np)
    compare("patch_embeds (M2 gate)", patch_ref_np, patch_tt_np)
    compare("decoder preds (M3 gate, pre-loss)", pred_ref_np, pred_tt_np)


def _numpy_reference_forward_loss(preds: np.ndarray, targets: np.ndarray, voxel_mask: np.ndarray) -> float:
    """Static-shape special case of `model_mae.py::MaskedAutoencoderViT.forward_loss`
    (lines 660-683): every sample contributes exactly `Q_pred` prediction
    slots, so the `scatter_add_`-by-`batch_ids` in the reference degenerates
    to a plain per-sample reduction (same simplification already validated
    against the reference formula in `test_loss_tt.py::_numpy_reference_loss`
    -- duplicated here rather than imported so this test's PyTorch-side math
    has no dependency on `loss_tt.py`/`smri_mae_tt` at all, only on plain
    numpy and the shapes `MaskedDecoder`/`forward_loss` actually produce).
    """
    scan_errors = ((preds - targets) ** 2 * voxel_mask).sum(axis=(1, 2))
    scan_voxels = voxel_mask.sum(axis=(1, 2))
    assert (scan_voxels > 0).all()
    return float((scan_errors / scan_voxels).mean())


def test_full_forward_loss_parity_m3_gate():
    """M3 gate (TENSTORRENT_PORT.md section 7): "loss within 1% of PyTorch on
    the same batch with identical visible_ids/pred_ids". This runs the full
    encoder -> decoder -> loss pipeline on both sides (not just the decoder
    output, which `test_encoder_decoder_forward_parity` already covers) and
    compares the final scalar loss value.

    Targets/voxel_mask are synthesized host-side (real patch values at
    `pred_ids`, an all-ones voxel mask -- this test has no brain mask/
    `packing.py` involved, so "every voxel valid" is the only well-defined
    choice, and matches `loss_tt.py`'s documented structural invariant that
    `packing.py` never produces an all-invalid scan either).
    """
    contract = _contract()
    encoder_depth = 2
    decoder_depth = 1

    param_rng = np.random.default_rng(SEED + 100)
    enc_params = M.init_encoder_params(param_rng, contract, depth=encoder_depth)
    dec_params = M.init_decoder_params(param_rng, contract, depth=decoder_depth)

    data_rng = np.random.default_rng(SEED + 101)
    batch = contract.batch
    num_patches = contract.num_patches

    vol_np = data_rng.standard_normal((batch, contract.in_chans, *contract.img_size)).astype(np.float32)
    vol_t = torch.from_numpy(vol_np)
    all_patches = patchify3d(vol_t, (contract.patch_size,) * 3).numpy()  # [B, N, patch_dim]

    visible_ids = np.zeros((batch, contract.l_vis), dtype=np.int64)
    pred_ids = np.zeros((batch, contract.q_pred), dtype=np.int64)
    for b in range(batch):
        perm = data_rng.permutation(num_patches)
        visible_ids[b] = perm[: contract.l_vis]
        pred_ids[b] = perm[contract.l_vis : contract.l_vis + contract.q_pred]

    b_idx = np.arange(batch)[:, None]
    visible_values = all_patches[b_idx, visible_ids]  # [B, L_vis, patch_dim]
    # Targets: the reference's `prepare_targets` with no target normalization
    # (`target_norm=None`, this project's default per `default_pretrain.yaml`)
    # is literally `pred_patchify(images)` -- i.e. the same patch values this
    # test already computed, gathered at `pred_ids`.
    targets = all_patches[b_idx, pred_ids].astype(np.float32)  # [B, Q_pred, patch_dim]
    voxel_mask = np.ones_like(targets, dtype=np.float32)

    # --- PyTorch reference: encoder -> decoder -> forward_loss ---
    patchify_ref = Patchify3D(contract.img_size, (contract.patch_size,) * 3, in_chans=contract.in_chans)
    patch_embed_ref = nn.Linear(patchify_ref.patch_dim, contract.embed_dim)
    pos_embed_ref = SinCosPosEmbed3D(contract.embed_dim, patchify_ref.grid_size)
    ref_encoder = MaskedEncoder(
        patchify=patchify_ref,
        patch_embed=patch_embed_ref,
        pos_embed=pos_embed_ref,
        depth=encoder_depth,
        embed_dim=contract.embed_dim,
        num_heads=contract.encoder_heads,
        qkv_bias=True,
        proj_bias=True,
        mlp_ratio=4,
        class_token=True,
        reg_tokens=0,
        no_embed_class=False,
        drop_path_rate=0.0,
    )
    _load_encoder_into_ref(ref_encoder, enc_params)
    ref_encoder.eval()

    decoder_pos_embed_ref = SinCosPosEmbed3D(contract.decoder_embed_dim, patchify_ref.grid_size)
    decoder_head_ref = nn.Linear(contract.decoder_embed_dim, patchify_ref.patch_dim)
    ref_decoder = MaskedDecoder(
        pos_embed=decoder_pos_embed_ref,
        head=decoder_head_ref,
        input_dim=contract.embed_dim,
        depth=decoder_depth,
        embed_dim=contract.decoder_embed_dim,
        num_heads=contract.decoder_heads,
        qkv_bias=True,
        proj_bias=True,
        mlp_ratio=4,
        class_token=True,
        no_embed_class=False,
    )
    _load_decoder_into_ref(ref_decoder, dec_params)
    ref_decoder.eval()

    with torch.no_grad():
        _, _, patch_ref = _dense_encoder_forward_visible_ids(ref_encoder, vol_t, torch.from_numpy(visible_ids))
        pred_ref = _dense_decoder_forward(
            ref_decoder,
            patch_ref,
            torch.from_numpy(visible_ids),
            torch.from_numpy(pred_ids),
        )
    ref_loss = _numpy_reference_forward_loss(pred_ref.numpy(), targets, voxel_mask)

    # --- TT: encoder -> decoder -> loss_tt ---
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        device = ctx.get_device()

        enc = M.Encoder(contract, depth=encoder_depth, params=enc_params)
        dec = M.Decoder(contract, depth=decoder_depth, params=dec_params)

        _, patch_tt = enc.forward(visible_values, visible_ids.astype(np.uint32))
        ttnn.synchronize_device(device)
        pred_tt = dec.forward(patch_tt, visible_ids.astype(np.uint32), pred_ids.astype(np.uint32))
        ttnn.synchronize_device(device)

        # Decoder output is [B, 1, Q_pred, patch_dim] (this port's leading-
        # singleton-head-axis convention, see modules_tt.py's module
        # docstring); loss_tt.py's PerScanValidVoxelMSELoss expects rank-3
        # [B, Q_pred, patch_dim] (packing.py's PackedBatch contract). Read
        # back to host and re-upload reshaped -- this test checks forward
        # *value* parity end-to-end (the M3 gate), not that gradients flow
        # through a single unbroken device graph across that reshape
        # boundary (that is M4's job, and packing.py's real callers feed
        # loss_tt straight from PackedBatch, never through modules_tt's
        # internal 4D convention).
        pred_tt_np = _to_numpy_fp32(pred_tt).reshape(batch, contract.q_pred, contract.patch_dim)

        preds_t = pred_targets_to_tensor(pred_tt_np)
        targets_t = pred_targets_to_tensor(targets)
        mask_t = voxel_mask_to_tensor(voxel_mask)
        loss_tt_tensor = per_scan_valid_voxel_mse_loss(preds_t, targets_t, mask_t)
        ttnn.synchronize_device(device)
        tt_loss = float(np.asarray(loss_tt_tensor.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])
    finally:
        ctx.close_device()

    rel_err = abs(tt_loss - ref_loss) / abs(ref_loss)
    print(f"[parity] full forward+loss (M3 gate): tt_loss={tt_loss:.6f} ref_loss={ref_loss:.6f} rel_err={rel_err:.4%}")
    assert rel_err < 0.01, f"loss rel_err={rel_err:.4%} exceeds M3 gate of 1%: tt_loss={tt_loss}, ref_loss={ref_loss}"
