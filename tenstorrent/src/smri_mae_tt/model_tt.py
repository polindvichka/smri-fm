"""Top-level MAE model for the TT-NN port: composes `modules_tt.Encoder` +
`modules_tt.Decoder` + `loss_tt.per_scan_valid_voxel_mse_loss` into one
module, matching `src/smri_mae/model_mae.py`'s `MaskedAutoencoderViT.forward()`
orchestration -- encode, then decode, then loss -- as closely as this port's
static-shape contract (`layout.ShapeContract`, TENSTORRENT_PORT.md section 3)
allows.

## What is deliberately *not* reproduced from the reference `forward()`

`MaskedAutoencoderViT.forward` (`model_mae.py:712-795`) additionally does,
on the PyTorch side:

  - `prepare_masks`/`prepare_targets`/`prepare_pred_mask`: derive
    `visible_ids`/`pred_ids`/`targets_patches`/`pred_mask_patches` from a
    dense `img_mask` at runtime, with dynamic `mask_ratio`/`pred_mask_ratio`
    sampling (`masking.pad_patch_mask`).
  - `forward_pred_images`: scatter predictions back into a dense
    `[B, N, patch_dim]` volume for visualization.

Both are host-side, mask/index/visualization bookkeeping with no learned
parameters and no gradient -- exactly the class of thing TENSTORRENT_PORT.md
section 3 moves off the device entirely: "Mask and index computation is pure
control flow with no gradient; it belongs in the dataloader workers." On
this port that role is `packing.pack_batch` (`packing.py`), which produces
`PackedBatch.visible_values/visible_ids/pred_targets/pred_voxel_mask/pred_ids`
*before* this module ever runs -- `MaskedAutoencoderViTTT.forward` below
takes those five arrays directly as its inputs, the same contract
`test_modules_tt.py::test_full_forward_loss_parity_m3_gate` already
exercises against `Encoder`/`Decoder`/`loss_tt` individually. There is no
`mask_ratio`/`pred_mask_ratio`/`pad_to_multiple` argument here: those are
`layout.choose_l_vis`/`choose_q_pred`'s job, resolved once into the
compile-time `ShapeContract.l_vis`/`q_pred` this model is built against, not
a per-call runtime knob (TENSTORRENT_PORT.md section 3's two "semantic
consequences" paragraphs). `forward_pred_images` is pure visualization
scatter-add with no parity requirement of its own (nothing in section 7's
milestone gates mentions it) and is left for `main_pretrain_tt.py`/eval
tooling to add later if needed, not this module.

## Closing the `preds` -> loss rank gap without severing autograd

`Decoder.forward` (`modules_tt.py`) returns `pred` shaped
`[B, 1, Q_pred, patch_dim]` (this port's `(B, 1, S, D)` convention, see
`modules_tt.py`'s module docstring). `loss_tt.PerScanValidVoxelMSELoss`
requires rank-3 `[B, Q_pred, patch_dim]` (`_validate_patch_tensor_shape`) --
`packing.PackedBatch`'s contract, and neither `modules_tt.py` nor
`loss_tt.py` may be edited by this change. Closing that rank-4-to-rank-3 gap
needs a reshape, and squeezing out a middle singleton axis is exactly a
*rank-reducing* reshape.

`ttml.ops.reshape.reshape` exists and is autograd-aware in general (used
throughout `tt-train/sources/examples` for the analogous `[B,1,1,S,D] ->
[B,1,S,D]` case), but tt-train's own test suite documents a load-bearing
limitation, not something inferred here:
`tt-train/tests/python/test_reshape_op.py::test_reshape_multiple_reshapes`
explicitly states "Backward through reshapes that reduce dimensionality
below 4D is currently not supported" and only exercises/asserts backward for
5D->4D reshapes in that file, never for 4D->3D or lower -- the exact
direction this module would need (`[B,1,Q,patch_dim]`, rank 4, down to
`[B,Q,patch_dim]`, rank 3). Depending on that unsupported path here would
mean an untested, likely-broken backward.

`_SqueezeDecoderPredAxis` below is the fix: a small, hand-written Tier-1
`ttml.autograd.Function` (TENSTORRENT_PORT.md section 4) whose forward is a
single raw `ttnn.reshape` dropping the singleton axis, and whose backward is
the *explicit* inverse `ttnn.reshape` of the incoming gradient -- the same
pattern `reshape_op.cpp`'s C++ implementation already uses internally
(`ttnn::reshape(out->get_grad(), original_shape)`), reimplemented here in
Python so this module owns and controls both directions directly rather than
depending on the unverified `ttml.ops.reshape.reshape` path above. Because
both directions are raw `ttnn.reshape` calls (not `ttml.ops.reshape.reshape`),
the documented below-4D-backward limitation does not apply.

This was originally worked around with a host round-trip (`pred` read to
numpy, reshaped, and re-uploaded as a fresh leaf tensor) that was
forward-value-correct but severed the autograd graph between the decoder and
the loss -- gradients computed by `loss_tt`'s backward could not flow back
into `Decoder`'s (or, transitively, `Encoder`'s) parameters through that
seam. `main_pretrain_tt.py` independently hit and fixed this exact gap at
its own call site with `_SqueezeDecoderPredAxis`, and empirically verified on
real Blackhole hardware that after one forward+backward, gradients are
populated, finite, and 100% nonzero on the decoder head weight, a decoder
block's QKV weight, and an encoder block's QKV weight -- i.e. the op
genuinely restores graph connectivity across the decoder/encoder boundary,
not just locally. That fix now lives here (the single source of truth both
this module's own `forward()`/`forward_loss()` and `main_pretrain_tt.py`
import), closing the M4 (backward parity) gap this docstring used to flag as
open -- see `test_model_tt.py`'s M4-gate test for the per-parameter gradient
PCC / grad-norm verification against the PyTorch reference's `.backward()`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ttml
import ttnn

from smri_mae_tt.layout import ShapeContract
from smri_mae_tt.loss_tt import (
    per_scan_valid_voxel_mse_loss,
    pred_targets_to_tensor,
    voxel_mask_to_tensor,
)
from smri_mae_tt.modules_tt import (
    Decoder,
    DecoderParams,
    Encoder,
    EncoderParams,
    init_decoder_params,
    init_encoder_params,
)

from ttml.autograd import Function
from ttml.modules import AbstractModuleBase


# ---------------------------------------------------------------------------
# The autograd fix for the decoder-pred -> loss-input rank gap. See module
# docstring, "Closing the `preds` -> loss rank gap without severing
# autograd", for the full derivation and the empirical hardware verification
# this is based on. `main_pretrain_tt.py` imports `squeeze_decoder_pred` from
# here rather than defining its own copy, so this is the single
# implementation both call sites share.
# ---------------------------------------------------------------------------


class _SqueezeDecoderPredAxis(Function):
    """`[B, 1, Q_pred, patch_dim]` -> `[B, Q_pred, patch_dim]`, autograd-connected.

    Forward and backward are both a single raw `ttnn.reshape` -- forward
    drops the singleton axis, backward puts it back on the incoming
    gradient. This does not call `ttml.ops.reshape.reshape` (see module
    docstring for why); it is a from-scratch Tier-1 op scoped to exactly
    this one shape transition, which is all `Decoder.forward`'s output shape
    (`modules_tt.py`'s `(B, 1, S, D)` convention) vs.
    `loss_tt.PerScanValidVoxelMSELoss`'s required rank-3 input
    (`packing.PackedBatch`'s convention) ever requires.
    """

    @staticmethod
    def forward(ctx, pred: "ttml.autograd.Tensor"):
        shape = tuple(pred.get_value().shape)
        if len(shape) != 4 or shape[1] != 1:
            raise ValueError(
                f"_SqueezeDecoderPredAxis expects a rank-4 [B,1,Q,D] tensor with a "
                f"singleton axis 1 (Decoder.forward's output convention), got shape {shape}"
            )
        ctx.orig_shape = shape
        b, _one, q, d = shape
        return ttnn.reshape(pred.get_value(), ttnn.Shape([b, q, d]))

    @staticmethod
    def backward(ctx, grad_output):
        b, one, q, d = ctx.orig_shape
        return ttnn.reshape(grad_output, ttnn.Shape([b, one, q, d]))


def squeeze_decoder_pred(
    pred: "ttml.autograd.Tensor", contract: ShapeContract, batch_size: int
) -> "ttml.autograd.Tensor":
    """Validate `pred`'s shape against `contract`/`batch_size` and squeeze it
    to the rank-3 shape `loss_tt.per_scan_valid_voxel_mse_loss` requires, via
    `_SqueezeDecoderPredAxis` (autograd-connected -- see module docstring).
    """
    expected = (batch_size, 1, contract.q_pred, contract.patch_dim)
    actual = tuple(pred.shape())
    if actual != expected:
        raise ValueError(f"decoder pred shape {actual} != expected {expected}")
    return _SqueezeDecoderPredAxis.apply(pred)


@dataclass
class ModelParams:
    """Bundles the encoder's and decoder's independently-initialized param
    trees, matching the two-call sequence `test_modules_tt.py`'s parity
    tests already use (`init_encoder_params` then `init_decoder_params` off
    one shared `numpy.random.Generator`)."""

    encoder: EncoderParams
    decoder: DecoderParams


def init_model_params(
    rng: np.random.Generator,
    contract: ShapeContract,
    encoder_depth: int,
    decoder_depth: int,
    mlp_ratio: float = 4.0,
) -> ModelParams:
    """From-scratch numpy init for both submodels off one RNG stream, in the
    same order (`encoder`, then `decoder`) `test_modules_tt.py`'s tests
    already draw them in -- so a model built from `init_model_params(rng,
    ...)` and one built by calling `init_encoder_params`/`init_decoder_params`
    directly off the same seed produce bit-identical parameters.
    """
    return ModelParams(
        encoder=init_encoder_params(rng, contract, depth=encoder_depth, mlp_ratio=mlp_ratio),
        decoder=init_decoder_params(rng, contract, depth=decoder_depth, mlp_ratio=mlp_ratio),
    )


@dataclass
class MaskedAutoencoderOutput:
    """Analogue of `MaskedAutoencoderViT.forward`'s `(loss, state)` return
    (`model_mae.py:712-795`), trimmed to what this port's static-shape
    pipeline actually produces and needs -- no `pred_images`/`targets_stats`
    (visualization/target-normalization, both out of scope, see module
    docstring), and no `mask`/`token_mask` (there is no runtime masking to
    report back, every sample already contributed exactly
    `contract.l_vis`/`q_pred` tokens by construction).
    """

    loss: "ttml.autograd.Tensor"  # [1, 1] scalar, per_scan_valid_voxel_mse_loss's output
    pred: "ttml.autograd.Tensor"  # [B, 1, Q_pred, patch_dim], Decoder.forward's raw output
    cls_embed: "ttml.autograd.Tensor"  # [B, 1, 1, embed_dim]
    patch_embeds: "ttml.autograd.Tensor"  # [B, 1, L_vis, embed_dim]


def _to_numpy_fp32(tensor: "ttml.autograd.Tensor") -> np.ndarray:
    return np.asarray(tensor.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


class MaskedAutoencoderViTTT(AbstractModuleBase):
    """Port of `model_mae.py`'s `MaskedAutoencoderViT`: owns an `Encoder` and
    a `Decoder` (both `modules_tt.py`, untouched by this change) and wires
    `forward` = encode -> decode -> per-scan valid-voxel MSE loss, matching
    the reference `forward`'s call order exactly (`self.encoder(...)` then
    `self.forward_decoder(...)` then `self.forward_loss(...)`,
    `model_mae.py:729-766`) minus the runtime mask/target preparation this
    port resolves host-side before the model ever runs (module docstring).

    Subclasses `AbstractModuleBase` (not a plain composing function) so
    `self.encoder`/`self.decoder` auto-register as child modules
    (`AbstractModuleBase.__setattr__` -> `_bind_module`, see
    `ttml/modules/module_base.py`) -- this is what makes
    `MaskedAutoencoderViTTT`'s parameters visible to a future optimizer via
    `named_parameters()`/`state_dict()` without this class doing anything
    extra, the same reason `modules_tt.py`'s `Encoder`/`Decoder` themselves
    subclass it.
    """

    def __init__(
        self,
        contract: ShapeContract,
        encoder_depth: int,
        decoder_depth: int,
        params: ModelParams,
        mlp_ratio: float = 4.0,
        use_tuned_gelu: bool = False,
    ) -> None:
        super().__init__()
        self.contract = contract
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        self.encoder = Encoder(contract, depth=encoder_depth, params=params.encoder, mlp_ratio=mlp_ratio, use_tuned_gelu=use_tuned_gelu)
        self.decoder = Decoder(contract, depth=decoder_depth, params=params.decoder, mlp_ratio=mlp_ratio, use_tuned_gelu=use_tuned_gelu)

    def forward_encoder(
        self, visible_values: np.ndarray, visible_ids: np.ndarray
    ) -> tuple["ttml.autograd.Tensor", "ttml.autograd.Tensor"]:
        """`Encoder.forward`, forwarded verbatim (`modules_tt.Encoder.forward`
        already does its own shape validation against `self.contract`).
        Returns `(cls_embed, patch_embeds)`.
        """
        return self.encoder.forward(visible_values, visible_ids)

    def forward_decoder(
        self,
        patch_embeds: "ttml.autograd.Tensor",
        visible_ids: np.ndarray,
        pred_ids: np.ndarray,
    ) -> "ttml.autograd.Tensor":
        """`Decoder.forward`, forwarded verbatim. Returns `pred`, `[B, 1,
        Q_pred, patch_dim]`."""
        return self.decoder.forward(patch_embeds, visible_ids, pred_ids)

    def _decoder_pred_to_loss_input(self, pred: "ttml.autograd.Tensor", batch: int) -> "ttml.autograd.Tensor":
        """`[B, 1, Q_pred, patch_dim]` -> a rank-3 `[B, Q_pred, patch_dim]`
        tensor, autograd-connected back to `pred` via `squeeze_decoder_pred`
        (module docstring, "Closing the `preds` -> loss rank gap without
        severing autograd"). Unlike the earlier host-round-trip
        implementation, the tensor this returns is not a fresh leaf: a
        `loss.backward()` on the result of `forward_loss`/`forward` below
        propagates real gradients back through this squeeze into
        `Decoder`'s (and, transitively, `Encoder`'s) parameters.
        """
        return squeeze_decoder_pred(pred, self.contract, batch)

    def forward_loss(
        self,
        pred: "ttml.autograd.Tensor",
        pred_targets: np.ndarray,
        pred_voxel_mask: np.ndarray,
    ) -> "ttml.autograd.Tensor":
        """Port of `MaskedAutoencoderViT.forward_loss` (`model_mae.py:660-683`),
        specialized to this port's static shapes (`loss_tt.py`'s module
        docstring: the `scatter_add_`-by-`batch_ids` reference formula
        degenerates to a plain `[B, Q_pred, patch_dim] -> [B]` reduction
        here, since every sample contributes exactly `contract.q_pred`
        prediction slots by construction -- no `pred_token_mask` exists on
        this port). `pred_targets`/`pred_voxel_mask`: host numpy, `[B,
        Q_pred, patch_dim]` (`packing.PackedBatch.pred_targets`/
        `pred_voxel_mask`).
        """
        contract = self.contract
        batch = pred_targets.shape[0]
        expected_targets_shape = (batch, contract.q_pred, contract.patch_dim)
        if pred_targets.shape != expected_targets_shape:
            raise ValueError(f"pred_targets shape {pred_targets.shape} != expected {expected_targets_shape}")
        if pred_voxel_mask.shape != expected_targets_shape:
            raise ValueError(f"pred_voxel_mask shape {pred_voxel_mask.shape} != expected {expected_targets_shape}")

        pred_shape = tuple(pred.shape())
        expected_pred_shape = (batch, 1, contract.q_pred, contract.patch_dim)
        if pred_shape != expected_pred_shape:
            raise ValueError(
                f"decoder pred shape {pred_shape} != expected {expected_pred_shape} "
                "(batch inferred from pred_targets); pred must be Decoder.forward's "
                "direct output for this batch"
            )

        preds_t = self._decoder_pred_to_loss_input(pred, batch)
        targets_t = pred_targets_to_tensor(pred_targets)
        mask_t = voxel_mask_to_tensor(pred_voxel_mask)
        return per_scan_valid_voxel_mse_loss(preds_t, targets_t, mask_t)

    def forward(
        self,
        visible_values: np.ndarray,
        visible_ids: np.ndarray,
        pred_ids: np.ndarray,
        pred_targets: np.ndarray,
        pred_voxel_mask: np.ndarray,
    ) -> MaskedAutoencoderOutput:
        """Full encode -> decode -> loss pipeline, matching
        `MaskedAutoencoderViT.forward`'s call order. All five arguments are
        host numpy, exactly `packing.PackedBatch`'s fields
        (`visible_values`, `visible_ids`, `pred_ids`, `pred_targets`,
        `pred_voxel_mask`) -- this is deliberately the same input contract
        `test_modules_tt.py::test_full_forward_loss_parity_m3_gate` already
        exercises against the three pieces individually, just formalized
        into one reusable call.

        Fails fast on a batch-size mismatch across the five host arrays
        before any device work runs (a mismatch here means the caller
        assembled a `PackedBatch` inconsistently -- e.g. mixed two
        different batches' arrays -- which is a bug to surface immediately,
        not something `Encoder`/`Decoder`/`loss_tt`'s own per-piece
        validation would necessarily catch as *this* specific problem,
        since none of them see all five arrays at once).
        """
        batches = {
            "visible_values": visible_values.shape[0],
            "visible_ids": visible_ids.shape[0],
            "pred_ids": pred_ids.shape[0],
            "pred_targets": pred_targets.shape[0],
            "pred_voxel_mask": pred_voxel_mask.shape[0],
        }
        if len(set(batches.values())) != 1:
            raise ValueError(f"inconsistent batch size across forward() inputs: {batches}")

        cls_embed, patch_embeds = self.forward_encoder(visible_values, visible_ids)
        pred = self.forward_decoder(patch_embeds, visible_ids, pred_ids)
        loss = self.forward_loss(pred, pred_targets, pred_voxel_mask)
        return MaskedAutoencoderOutput(loss=loss, pred=pred, cls_embed=cls_embed, patch_embeds=patch_embeds)
