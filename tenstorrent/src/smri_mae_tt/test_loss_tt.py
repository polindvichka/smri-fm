"""Tests for `loss_tt.per_scan_valid_voxel_mse_loss`.

Gate (see loss_tt.py's module docstring for the derivation):
  - forward value matches a plain-numpy reimplementation of
    `model_mae.py:forward_loss`'s formula (non-ragged, fixed `[B, Q_pred,
    patch_dim]` special case -- this port's static-shape contract makes the
    scatter-add in the original a plain reduction, see loss_tt.py) to a
    bf16-appropriate relative tolerance. This is exact elementwise
    arithmetic (not an approximate/tuned kernel), so the only source of
    mismatch is bf16 quantization of the inputs on upload + whatever
    reduced-precision math fidelity `ttnn.sum`/`ttnn.divide` use by default
    -- a tight `rtol` is the right gate, not a loose PCC-style one.
  - backward gradient matches both (a) the closed-form analytic gradient and
    (b) central finite differences computed directly against the device
    forward pass, to a bf16-appropriate tolerance.

`pytest.importorskip("ttml")` is the one legitimate skip (environment
availability, not a fallback masking a bug -- see `feedback_fail_fast`
memory), matching `ops_tt/test_attention.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae_tt.loss_tt import (  # noqa: E402
    per_scan_valid_voxel_mse_loss,
    pred_targets_to_tensor,
    voxel_mask_to_tensor,
)

SEED = 2026


def _numpy_reference_loss(preds: np.ndarray, targets: np.ndarray, voxel_mask: np.ndarray) -> np.ndarray:
    """Fixed-shape special case of `model_mae.py:forward_loss` (lines 660-683):
    every sample contributes exactly Q_pred slots (no ragged
    `pred_token_mask`/scatter-add needed), so the scatter-add-by-`batch_ids`
    degenerates to a plain per-sample reduction.
    """
    scan_errors = ((preds - targets) ** 2 * voxel_mask).sum(axis=(1, 2))
    scan_voxels = voxel_mask.sum(axis=(1, 2))
    assert (scan_voxels > 0).all(), "test fixture must not construct a degenerate all-invalid scan"
    return (scan_errors / scan_voxels).mean()


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _make_batch(rng: np.random.Generator, batch: int, q_pred: int, patch_dim: int):
    preds_np = rng.standard_normal((batch, q_pred, patch_dim)).astype(np.float32)
    targets_np = rng.standard_normal((batch, q_pred, patch_dim)).astype(np.float32)
    # >=1 valid voxel per (sample, patch) row so no row -- and hence no
    # sample -- can degenerate to scan_voxels == 0 (mirrors the structural
    # guarantee packing.py provides; see loss_tt.py docstring).
    voxel_mask_np = (rng.random((batch, q_pred, patch_dim)) > 0.3).astype(np.float32)
    voxel_mask_np[:, :, 0] = 1.0
    return preds_np, targets_np, voxel_mask_np


def test_forward_matches_numpy_reference():
    batch, q_pred, patch_dim = 3, 64, 512
    rng = np.random.default_rng(SEED)
    preds_np, targets_np, voxel_mask_np = _make_batch(rng, batch, q_pred, patch_dim)

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        preds_t = pred_targets_to_tensor(preds_np)
        targets_t = pred_targets_to_tensor(targets_np)
        mask_t = voxel_mask_to_tensor(voxel_mask_np)

        # Read back the bf16-quantized inputs so the numpy reference is
        # computed from exactly what the device kernels see -- otherwise
        # bf16 upload rounding shows up as spurious error unrelated to the
        # arithmetic under test.
        preds_bf16 = np.asarray(preds_t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)
        targets_bf16 = np.asarray(targets_t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)
        mask_bf16 = np.asarray(mask_t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)

        loss = per_scan_valid_voxel_mse_loss(preds_t, targets_t, mask_t)
        loss_val = float(np.asarray(loss.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])

        ref = float(_numpy_reference_loss(preds_bf16, targets_bf16, mask_bf16))

        assert np.allclose(loss_val, ref, rtol=1e-2, atol=1e-4), (
            f"device loss {loss_val} vs numpy reference {ref}, "
            f"relative error {abs(loss_val - ref) / abs(ref):.4%}"
        )
    finally:
        ctx.close_device()


def test_backward_matches_analytic_and_finite_difference():
    batch, q_pred, patch_dim = 2, 32, 128
    rng = np.random.default_rng(SEED + 1)
    preds_np, targets_np, voxel_mask_np = _make_batch(rng, batch, q_pred, patch_dim)

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        preds_t = pred_targets_to_tensor(preds_np)
        targets_t = pred_targets_to_tensor(targets_np)
        mask_t = voxel_mask_to_tensor(voxel_mask_np)
        preds_t.set_requires_grad(True)

        preds_bf16 = np.asarray(preds_t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)
        targets_bf16 = np.asarray(targets_t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)
        mask_bf16 = np.asarray(mask_t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)

        loss = per_scan_valid_voxel_mse_loss(preds_t, targets_t, mask_t)
        loss.backward(retain_graph=False)

        assert preds_t.is_grad_initialized(), "preds should have a gradient after backward()"
        grad_device = np.asarray(preds_t.get_grad_tensor().to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)

        # Closed-form analytic gradient (loss_tt.py docstring), evaluated on
        # the exact bf16-quantized inputs the device saw. grad_output is
        # 1.0 (scalar `.backward()` on the loss itself).
        scan_voxels = mask_bf16.sum(axis=(1, 2))
        analytic = 2.0 * (preds_bf16 - targets_bf16) * mask_bf16 / scan_voxels.reshape(batch, 1, 1) / batch

        assert np.allclose(grad_device, analytic, rtol=1e-2, atol=1e-3), (
            f"device grad vs closed-form analytic grad: max abs diff "
            f"{np.abs(grad_device - analytic).max():.6g}"
        )

        # Independent check: central finite differences directly against the
        # device forward pass (not the analytic formula), on a handful of
        # random elements.
        def device_loss(preds_arr: np.ndarray) -> float:
            p = pred_targets_to_tensor(preds_arr)
            out = per_scan_valid_voxel_mse_loss(p, targets_t, mask_t)
            val = float(np.asarray(out.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])
            ttml.autograd.AutoContext.get_instance().reset_graph()
            return val

        eps = 1e-2
        rng2 = np.random.default_rng(SEED + 2)
        for _ in range(4):
            b, q, p = rng2.integers(batch), rng2.integers(q_pred), rng2.integers(patch_dim)
            plus = preds_bf16.copy()
            plus[b, q, p] += eps
            minus = preds_bf16.copy()
            minus[b, q, p] -= eps
            fd_grad = (device_loss(plus) - device_loss(minus)) / (2 * eps)
            assert np.allclose(fd_grad, grad_device[b, q, p], rtol=5e-2, atol=5e-3), (
                f"finite-difference grad at ({b},{q},{p}): fd={fd_grad:.6f} "
                f"device={grad_device[b, q, p]:.6f}"
            )
    finally:
        ctx.close_device()


def test_voxel_mask_to_tensor_accepts_bool():
    rng = np.random.default_rng(SEED + 3)
    mask_np = rng.random((2, 32, 32)) > 0.5  # bool dtype, as packing.py produces
    assert mask_np.dtype == np.bool_

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        t = voxel_mask_to_tensor(mask_np)
        got = np.asarray(t.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)
        np.testing.assert_array_equal(got, mask_np.astype(np.float32))
    finally:
        ctx.close_device()


def test_voxel_mask_to_tensor_rejects_non_binary_values():
    rng = np.random.default_rng(SEED + 4)
    mask_np = rng.random((2, 32, 32)).astype(np.float32)  # soft values, not 0/1
    with pytest.raises(ValueError, match="0/1"):
        voxel_mask_to_tensor(mask_np)


def test_forward_rejects_shape_mismatch():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        preds_t = pred_targets_to_tensor(np.zeros((2, 32, 32), dtype=np.float32))
        targets_t = pred_targets_to_tensor(np.zeros((2, 64, 32), dtype=np.float32))
        mask_t = voxel_mask_to_tensor(np.ones((2, 32, 32), dtype=np.float32))
        with pytest.raises(ValueError, match="shapes must match"):
            per_scan_valid_voxel_mse_loss(preds_t, targets_t, mask_t)
    finally:
        ctx.close_device()
