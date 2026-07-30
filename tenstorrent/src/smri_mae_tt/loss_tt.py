"""Per-scan valid-voxel MSE loss for the TT-NN MAE port.

## What this reproduces

`model_mae.py:forward_loss` (`/home/tt-small/projects/smri-fm/src/smri_mae/model_mae.py:660-683`)
is **not** a plain MSE. Per scan it computes

    (sum of squared error over valid voxels) / (count of valid voxels)

and then means that per-scan quantity over the batch. On the PyTorch side this
needs a `scatter_add_` because samples contribute a ragged number of
prediction slots (`pred_token_mask.nonzero()`, lines 669-682) -- the batch is
packed as a flat `[T, P]` tensor of only the *real* prediction tokens across
all samples, with `batch_ids`/`patch_ids` recovering which sample/patch each
row belongs to.

TENSTORRENT_PORT.md section 3 removes the raggedness: under this port's
static-shape contract every sample contributes **exactly**
`ShapeContract.q_pred` prediction slots (see `layout.py`), sampled on the host
by `packing.pack_batch` (`packing.py:38-84`). There is no `pred_token_mask`
here and no scatter-add -- `preds`/`targets`/`voxel_mask` below are already
the fixed-shape `[B, Q_pred, patch_dim]` tensors (`PackedBatch.pred_targets`,
`PackedBatch.pred_voxel_mask` in `packing.py:32-34`), so the "segment sum"
degenerates to a plain `[B, Q_pred, patch_dim] -> [B]` reduction followed by a
mean over `B` (TENSTORRENT_PORT.md section 4, item 2).

## Why this needs a hand-written backward (Tier 1, not composition)

`ttml.ops.unary.mean` (`nb_ops.cpp:513`, `ttml::ops::mean`,
`unary_ops.hpp:14`) reduces an *entire* tensor to one scalar -- there is no
`dim` argument, so it cannot express the per-sample
`sum(dim=-1) -> sum(dim=-1)` this loss needs while keeping the batch
dimension intact. The natural fix, a dim-aware `ttml.ops.unary.sum`, is
present in the C++ source but explicitly commented out of the Python binding
(`nb_ops.cpp:514-515`: `// py_unary.def("sum", &ttml::ops::sum, ...)`, mirrored
in `unary_ops.hpp:15`) -- it is not reachable from Python at all in this
build. `ttml.ops.binary.div`'s autograd backward additionally throws on any
broadcasted operand (`ops::operator/` in `binary_ops.cpp:197-212`: "Broadcasting
is not supported in the backward pass of operator/"), which the
`[B,Q,P] / [B,1,1]` shape in the closed-form gradient below would trigger.

So, per TENSTORRENT_PORT.md section 4 ("Tier 1 -- compose in Python on
`ttml.autograd.Function` ... this is where the MAE loss ... belongs"), this
whole loss is one hand-written `ttml.autograd.Function`: forward built from
raw `ttnn` primitives (`ttnn.sum(..., dim=-1)` *is* present and dim-aware --
confirmed via `help(ttnn.sum)` on the built wheel, unlike the `ttml` binding),
and an explicit closed-form backward, exactly the `PythonLinearOp` pattern in
`tt-train/tests/python/test_custom_operations.py:84-185`.

Forward (all in fp32, see "Precision" below):

    diff         = preds - targets                          # [B, Q, P]
    masked_sq    = diff * diff * voxel_mask                  # [B, Q, P]
    scan_errors  = sum(sum(masked_sq, dim=-1), dim=-1)        # [B]
    scan_voxels  = sum(sum(voxel_mask, dim=-1), dim=-1)       # [B]
    loss         = mean(scan_errors / scan_voxels)            # scalar [1, 1]

Backward (TENSTORRENT_PORT.md section 4, item 2 -- derived and empirically
verified against finite differences, see `test_loss_tt.py`):

    d(loss)/d(preds) = grad_output * 2 * (preds - targets) * voxel_mask
                       / scan_voxels[:, None, None] / B

`scan_voxels` broadcasts per-sample over that sample's own `[Q, P]` slice.
The outer `1/B` is the standard `mean`-over-batch chain-rule factor -- each
sample's scalar loss contributes `1/B` to the total, so its own gradient
picks up that same factor once (not on top of anything else); this is a
gradient of the *scalar* loss w.r.t. that sample's `preds`, consistent with
`grad_output` being the incoming upstream gradient (1.0 for a direct
`loss.backward()`). `targets`/`voxel_mask` are host-derived constants (no
`get_param_groups` entry could ever reference them) so their gradients are
`None`.

## Precision

TENSTORRENT_PORT.md section 6 mandates "bf16 activations, fp32
accumulation". `preds`/`targets`/`voxel_mask` may be bf16 (decoder output and
host-uploaded constants), but summing ~`Q_pred * patch_dim` (order of
thousands) small squared-error terms in bf16 (~8 bits of mantissa) would lose
real precision. Forward and backward both `ttnn.typecast(..., FLOAT32)` every
input up front and do all elementwise/reduction arithmetic in fp32; the
returned gradient is cast back to `preds`' original dtype (mirrors how
`ttml`'s own `unbroadcast_grad` in `binary_ops.cpp:70-76` asks for
`core::ComputeKernelConfig::precise()` on its internal `moreh_sum` -- same
motivation, applied by literally computing in fp32 instead of relying on a
compute-kernel-config flag from Python, since none is exposed for `ttnn.sum`/
`ttnn.divide` through this binding).

## Structural invariant: `scan_voxels` can never be zero

`packing.pack_batch` (`packing.py:63-71`) draws `pred_ids` for a sample only
from `remaining_idx`, a subset of `valid_idx = np.flatnonzero(valid[b])`
where `valid[b] = mask_patches[b].any(dim=-1)` (`packing.py:57`) -- i.e. every
candidate *patch* id has, by construction, at least one valid voxel before it
can ever be selected into `pred_ids`. `pred_voxel_mask` (`packing.py:81`,
`PackedBatch.pred_voxel_mask`) is `mask_patches_np[b_idx, pred_ids]`: the
per-voxel validity of exactly those already-valid-by-construction patches.
Therefore `scan_voxels = pred_voxel_mask.sum(axis=(1, 2))` is a sum of
`Q_pred` per-patch counts each `>= 1`, so `scan_voxels >= Q_pred >= 1` for
every sample in every batch this port can ever produce -- a
degenerate/all-invalid scan is structurally impossible given how
`pred_voxel_mask` is derived, not merely unlikely. Per the no-fallback
principle this is therefore *not* guarded with a runtime zero-check in
`forward` (that would be defensive code against a case that cannot occur, and
would additionally force a device->host sync on every training step just to
check it) -- it is guarded at the host/numpy boundary instead, in
`voxel_mask_to_tensor`, which fails loudly on the actually-possible failure
mode: a caller passing a `voxel_mask` that didn't come from `packing.py` and
contains values other than 0/1.
"""

from __future__ import annotations

import numpy as np
import ttml
import ttnn
from ttml.autograd import Function, Tensor


def _validate_patch_tensor_shape(name: str, shape: "ttnn.Shape") -> None:
    if len(shape) != 3:
        raise ValueError(
            f"{name} must be rank-3 [B, Q_pred, patch_dim] (packing.py's PackedBatch "
            f"contract), got shape {tuple(shape)}"
        )


def voxel_mask_to_tensor(pred_voxel_mask: np.ndarray) -> "ttml.autograd.Tensor":
    """Host-side convert `PackedBatch.pred_voxel_mask` (`packing.py:33`, a
    `[B, Q_pred, patch_dim]` numpy array) into a device `ttml.autograd.Tensor`.

    `packing.py` produces this mask as `mask_patches_np[b_idx, pred_ids]`
    (`packing.py:81`) where `mask_patches` is `patchify3d(masks.float(), ...).bool()`
    (`packing.py:56`) -- genuinely `bool` dtype. This function accepts `bool`
    directly (cast to float32 is exact and lossless) and, for any other
    dtype, fails loudly if any element is not exactly 0 or 1 rather than
    silently truncating/rounding it (no-fallback principle) -- a caller
    passing e.g. soft probabilities here would otherwise corrupt the loss's
    "valid voxel count" semantics without any error.
    """
    arr = np.asarray(pred_voxel_mask)
    if arr.dtype != np.bool_:
        bad = ~np.isin(arr, (0, 1))
        if bad.any():
            bad_values = np.unique(arr[bad])
            raise ValueError(
                f"voxel_mask must contain only 0/1 (or be bool), found "
                f"{bad.sum()} element(s) with other values, e.g. {bad_values[:8].tolist()}. "
                "This mask must come from packing.pack_batch's "
                "PackedBatch.pred_voxel_mask; a non-0/1 value here means the caller "
                "built it incorrectly upstream."
            )
    return Tensor.from_numpy(arr.astype(np.float32), layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)


def pred_targets_to_tensor(pred_targets: np.ndarray) -> "ttml.autograd.Tensor":
    """Host-side convert `PackedBatch.pred_targets` (`packing.py:32`, a
    `[B, Q_pred, patch_dim]` float32 numpy array) into a device tensor.
    """
    arr = np.asarray(pred_targets, dtype=np.float32)
    return Tensor.from_numpy(arr, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)


class PerScanValidVoxelMSELoss(Function):
    """`ttml.autograd.Function`: per-scan mean valid-voxel MSE (module docstring).

    Forward args:
        preds:       [B, Q_pred, patch_dim] decoder output, requires_grad expected.
        targets:     [B, Q_pred, patch_dim] host-derived constant (no grad).
        voxel_mask:  [B, Q_pred, patch_dim] host-derived constant (no grad),
                     1.0 = valid voxel, 0.0 = outside the brain mask.

    Returns a `[1, 1]` scalar tensor (not rank-0: `ttml::autograd::Tensor::backward`
    seeds its upstream gradient via `ttml::core::ones_like` ->
    `ttnn::moreh_full_like` -> `ttnn::prim::full`, which `TT_FATAL`s with
    "shape.size() > 1" on a rank-0/rank-1 tensor -- confirmed empirically on
    this build. `ttml`'s own losses use the same convention, e.g.
    `cross_entropy_loss`'s `ttnn::Shape({1, 1, 1, 1})` in `losses.cpp:80`).
    """

    @staticmethod
    def forward(ctx, preds: "ttml.autograd.Tensor", targets: "ttml.autograd.Tensor", voxel_mask: "ttml.autograd.Tensor"):
        preds_shape = preds.get_value().shape
        _validate_patch_tensor_shape("preds", preds_shape)
        _validate_patch_tensor_shape("targets", targets.get_value().shape)
        _validate_patch_tensor_shape("voxel_mask", voxel_mask.get_value().shape)
        if tuple(preds_shape) != tuple(targets.get_value().shape) or tuple(preds_shape) != tuple(
            voxel_mask.get_value().shape
        ):
            raise ValueError(
                f"preds/targets/voxel_mask shapes must match exactly, got "
                f"{tuple(preds_shape)} / {tuple(targets.get_value().shape)} / "
                f"{tuple(voxel_mask.get_value().shape)}"
            )
        batch_size = int(preds_shape[0])

        preds_f32 = ttnn.typecast(preds.get_value(), ttnn.DataType.FLOAT32)
        targets_f32 = ttnn.typecast(targets.get_value(), ttnn.DataType.FLOAT32)
        mask_f32 = ttnn.typecast(voxel_mask.get_value(), ttnn.DataType.FLOAT32)

        diff = ttnn.subtract(preds_f32, targets_f32)
        masked_sq = ttnn.multiply(ttnn.multiply(diff, diff), mask_f32)

        patch_errors = ttnn.sum(masked_sq, dim=-1, keepdim=False)  # [B, Q_pred]
        scan_errors = ttnn.sum(patch_errors, dim=-1, keepdim=False)  # [B]

        voxel_counts = ttnn.sum(mask_f32, dim=-1, keepdim=False)  # [B, Q_pred]
        scan_voxels = ttnn.sum(voxel_counts, dim=-1, keepdim=False)  # [B]

        per_scan_loss = ttnn.divide(scan_errors, scan_voxels)  # [B]
        total = ttnn.sum(per_scan_loss, dim=0, keepdim=False)  # rank-0 scalar
        total = ttnn.reshape(total, [1, 1])  # see docstring: backward needs rank >= 2
        loss = ttnn.multiply(total, 1.0 / batch_size)

        ctx.save_for_backward(preds, targets, voxel_mask)
        ctx.scan_voxels = scan_voxels
        ctx.batch_size = batch_size
        ctx.orig_dtype = preds.get_value().dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        preds, targets, voxel_mask = ctx.saved_tensors

        preds_f32 = ttnn.typecast(preds.get_value(), ttnn.DataType.FLOAT32)
        targets_f32 = ttnn.typecast(targets.get_value(), ttnn.DataType.FLOAT32)
        mask_f32 = ttnn.typecast(voxel_mask.get_value(), ttnn.DataType.FLOAT32)

        grad = ttnn.multiply(ttnn.subtract(preds_f32, targets_f32), mask_f32)
        grad = ttnn.multiply(grad, 2.0)

        scan_voxels_bcast = ttnn.reshape(ctx.scan_voxels, [ctx.batch_size, 1, 1])
        grad = ttnn.divide(grad, scan_voxels_bcast)  # broadcast over (Q_pred, patch_dim)
        grad = ttnn.multiply(grad, 1.0 / ctx.batch_size)

        grad_output_f32 = ttnn.typecast(grad_output, ttnn.DataType.FLOAT32)
        grad = ttnn.multiply(grad, grad_output_f32)  # chain rule w/ upstream grad, broadcasts from [1,1]

        grad = ttnn.typecast(grad, ctx.orig_dtype)
        return grad, None, None


def per_scan_valid_voxel_mse_loss(
    preds: "ttml.autograd.Tensor",
    targets: "ttml.autograd.Tensor",
    voxel_mask: "ttml.autograd.Tensor",
) -> "ttml.autograd.Tensor":
    """Per-scan mean valid-voxel MSE (module docstring). `preds` is the
    decoder's `[B, Q_pred, patch_dim]` output (should already have
    `requires_grad` set via the model's autograd graph); `targets`/
    `voxel_mask` are host constants -- build them with `pred_targets_to_tensor`/
    `voxel_mask_to_tensor` from a `packing.PackedBatch`.
    """
    return PerScanValidVoxelMSELoss.apply(preds, targets, voxel_mask)
