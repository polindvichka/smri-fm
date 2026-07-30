"""Tests for `params.py`: torch `state_dict` <-> TT parameter conversion.

Two things are checked, using `parity.py`'s shared frozen fixture / harness
(TENSTORRENT_PORT.md section 7) rather than this file's own copy of
`test_modules_tt.py`'s dense-SDPA-reference/`compare()` pattern:

1. **Round trip.** A small PyTorch reference `MaskedEncoder`/`MaskedDecoder`
   pair (`src/smri_mae/model_mae.py`) is built with the library's own default
   init, its `state_dict()` is loaded into a matching `modules_tt.Encoder`/
   `Decoder` via `params.state_dict_to_tt`, and both sides are run on
   identical inputs -- gated at PCC >= 0.999, the M2/M3 gate. Then
   `params.tt_to_state_dict` reads the TT parameters back out, is loaded
   (via plain `nn.Module.load_state_dict(strict=True)`, which also proves
   the emitted state_dict is *complete* -- no missing/extra keys) into a
   *fresh* PyTorch reference pair, and that pair's forward pass on the same
   inputs is compared against the *original* reference forward pass with a
   bounded relative-error gate (not exact equality -- a bf16/bf8b round
   trip is expected to lose precision, see `params.py`'s module docstring).
2. **Fail-fast.** A missing key, a wrong-shaped tensor, and an unexpected
   extra key in the `state_dict` each must raise immediately -- never be
   silently skipped, clamped, or ignored.

No trained checkpoint exists anywhere in this repo (see `params.py`'s module
docstring), so, exactly as in `test_modules_tt.py`, "the PyTorch reference"
here means a freshly-constructed small-scale `MaskedEncoder`/`MaskedDecoder`,
not a real trained model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae.model_mae import MaskedDecoder, MaskedEncoder  # noqa: E402
from smri_mae.modules import patchify3d  # noqa: E402

from smri_mae_tt import modules_tt as M  # noqa: E402
from smri_mae_tt import params as P  # noqa: E402
from smri_mae_tt.layout import ShapeContract  # noqa: E402
from smri_mae_tt.parity import (  # noqa: E402
    FROZEN_FIXTURE,
    build_reference_decoder as _build_ref_decoder,
    build_reference_encoder as _build_ref_encoder,
    compare,
    dense_decoder_forward as _dense_decoder_forward,
    dense_encoder_forward_visible_ids as _dense_encoder_forward_visible_ids,
    to_numpy_fp32 as _to_numpy_fp32,
)

SEED = FROZEN_FIXTURE.param_seed
PCC_GATE = 0.999

# Bound on relative Frobenius error after a full fp32 -> TT (bf16/bf8b) ->
# fp32 round trip, reproduced through *two full forward passes* (not just
# reading the weights back). Not 0: the whole point of this test is that the
# round trip is lossy by design (params.py's module docstring). Calibrated
# generously above what was actually measured on real hardware for this
# small config (see the printed `[roundtrip]` lines when running with -s) --
# tight enough to catch a real correctness bug (e.g. a transposed weight,
# which would blow this bound wide open), loose enough not to flake on
# ordinary bf16/bf8b rounding.
ENCODER_ROUNDTRIP_REL_ERR_BOUND = 0.05
DECODER_ROUNDTRIP_REL_ERR_BOUND = 0.15


def _contract() -> ShapeContract:
    # Same dims as parity.FROZEN_FIXTURE.contract: head_dim=32 for both
    # encoder (dim=64/heads=2) and decoder (dim=32/heads=1) -- required, not
    # a style choice, see modules_tt.py's Attention.__init__ docstring on
    # ttml.ops.multi_head_utils.heads_creation silently corrupting data for
    # non-tile-aligned head_dim.
    return FROZEN_FIXTURE.contract


def _combined_state_dict(ref_encoder: MaskedEncoder, ref_decoder: MaskedDecoder) -> dict[str, torch.Tensor]:
    sd: dict[str, torch.Tensor] = {}
    for k, v in ref_encoder.state_dict().items():
        sd[f"encoder.{k}"] = v
    for k, v in ref_decoder.state_dict().items():
        sd[f"decoder.{k}"] = v
    return sd


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_state_dict_round_trip_matches_pytorch_forward():
    contract = _contract()
    encoder_depth = 2
    decoder_depth = 1

    # --- PyTorch reference, default (library) init ---
    ref_encoder = _build_ref_encoder(contract, encoder_depth)
    ref_decoder = _build_ref_decoder(contract, decoder_depth)
    ref_encoder.eval()
    ref_decoder.eval()
    sd = _combined_state_dict(ref_encoder, ref_decoder)

    # --- Shared input data ---
    data_rng = np.random.default_rng(SEED)
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

    visible_ids = np.zeros((batch, contract.l_vis), dtype=np.int64)
    pred_ids = np.zeros((batch, contract.q_pred), dtype=np.int64)
    for b in range(batch):
        perm = data_rng.permutation(num_patches)
        visible_ids[b] = perm[: contract.l_vis]
        pred_ids[b] = perm[contract.l_vis : contract.l_vis + contract.q_pred]

    b_idx = np.arange(batch)[:, None]
    visible_values = all_patches[b_idx, visible_ids]  # [B, L_vis, patch_dim]

    visible_ids_t = torch.from_numpy(visible_ids)
    pred_ids_t = torch.from_numpy(pred_ids)

    # --- Original PyTorch forward (pre-round-trip reference) ---
    with torch.no_grad():
        cls_ref, reg_ref, patch_ref = _dense_encoder_forward_visible_ids(ref_encoder, vol_t, visible_ids_t)
        assert reg_ref is None  # reg_tokens=0
        pred_ref = _dense_decoder_forward(ref_decoder, patch_ref, visible_ids_t, pred_ids_t)

    # --- Build TT modules (dummy init -- overwritten below), load via params.py, run ---
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        device = ctx.get_device()

        param_rng = np.random.default_rng(SEED + 1)
        enc_params, dec_params = P.init_tt_params(param_rng, contract, encoder_depth, decoder_depth)
        enc = M.Encoder(contract, depth=encoder_depth, params=enc_params)
        dec = M.Decoder(contract, depth=decoder_depth, params=dec_params)

        P.state_dict_to_tt(sd, enc, dec)

        cls_tt, patch_tt = enc.forward(visible_values, visible_ids.astype(np.uint32))
        ttnn.synchronize_device(device)
        pred_tt = dec.forward(patch_tt, visible_ids.astype(np.uint32), pred_ids.astype(np.uint32))
        ttnn.synchronize_device(device)

        cls_tt_np = _to_numpy_fp32(cls_tt)
        patch_tt_np = _to_numpy_fp32(patch_tt)
        pred_tt_np = _to_numpy_fp32(pred_tt)

        # --- Read TT parameters back out, into a fresh PyTorch pair ---
        sd2 = P.tt_to_state_dict(enc, dec)
    finally:
        ctx.close_device()

    cls_ref_np = cls_ref.numpy()
    patch_ref_np = patch_ref.numpy()
    pred_ref_np = pred_ref.numpy()

    assert cls_tt_np.reshape(cls_ref_np.shape).shape == cls_ref_np.shape
    assert patch_tt_np.reshape(patch_ref_np.shape).shape == patch_ref_np.shape
    assert pred_tt_np.reshape(pred_ref_np.shape).shape == pred_ref_np.shape

    # --- Stage 1: state_dict_to_tt loaded the weights correctly (M2/M3 gate) ---
    compare("cls_embed (state_dict_to_tt, M2 gate)", cls_ref_np, cls_tt_np)
    compare("patch_embeds (state_dict_to_tt, M2 gate)", patch_ref_np, patch_tt_np)
    compare("decoder preds (state_dict_to_tt, M3 gate)", pred_ref_np, pred_tt_np)

    # --- Stage 2: tt_to_state_dict's output is complete and strict-loadable ---
    enc_sd2 = {k[len("encoder.") :]: v for k, v in sd2.items() if k.startswith("encoder.")}
    dec_sd2 = {k[len("decoder.") :]: v for k, v in sd2.items() if k.startswith("decoder.")}
    assert set(enc_sd2) == set(ref_encoder.state_dict().keys())
    assert set(dec_sd2) == set(ref_decoder.state_dict().keys())

    ref_encoder2 = _build_ref_encoder(contract, encoder_depth)
    ref_decoder2 = _build_ref_decoder(contract, decoder_depth)
    ref_encoder2.load_state_dict(enc_sd2, strict=True)
    ref_decoder2.load_state_dict(dec_sd2, strict=True)
    ref_encoder2.eval()
    ref_decoder2.eval()

    with torch.no_grad():
        cls_rt, reg_rt, patch_rt = _dense_encoder_forward_visible_ids(ref_encoder2, vol_t, visible_ids_t)
        assert reg_rt is None
        pred_rt = _dense_decoder_forward(ref_decoder2, patch_rt, visible_ids_t, pred_ids_t)

    # --- Stage 3: round-tripped forward reproduces the original forward up to
    # bounded bf16/bf8b precision loss (not exact equality -- see params.py's
    # module docstring on what precision each direction preserves/loses). ---
    cls_stats = compare("cls_embed (round trip)", cls_ref_np, cls_rt.numpy(), pcc_gate=0.0)
    patch_stats = compare("patch_embeds (round trip)", patch_ref_np, patch_rt.numpy(), pcc_gate=0.0)
    pred_stats = compare("decoder preds (round trip)", pred_ref_np, pred_rt.numpy(), pcc_gate=0.0)

    assert cls_stats["rel_frob_err"] < ENCODER_ROUNDTRIP_REL_ERR_BOUND, (
        f"cls_embed round-trip rel_frob_err={cls_stats['rel_frob_err']:.4%} exceeds bound "
        f"{ENCODER_ROUNDTRIP_REL_ERR_BOUND:.4%}"
    )
    assert patch_stats["rel_frob_err"] < ENCODER_ROUNDTRIP_REL_ERR_BOUND, (
        f"patch_embeds round-trip rel_frob_err={patch_stats['rel_frob_err']:.4%} exceeds bound "
        f"{ENCODER_ROUNDTRIP_REL_ERR_BOUND:.4%}"
    )
    assert pred_stats["rel_frob_err"] < DECODER_ROUNDTRIP_REL_ERR_BOUND, (
        f"decoder preds round-trip rel_frob_err={pred_stats['rel_frob_err']:.4%} exceeds bound "
        f"{DECODER_ROUNDTRIP_REL_ERR_BOUND:.4%} (decoder qkv/mlp are bfloat8_b, so a looser "
        "bound than the encoder's is expected, not just tolerated)"
    )


def test_state_dict_to_tt_missing_key_raises():
    contract = _contract()
    ref_encoder = _build_ref_encoder(contract, depth=1)
    ref_decoder = _build_ref_decoder(contract, depth=1)
    sd = _combined_state_dict(ref_encoder, ref_decoder)

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        param_rng = np.random.default_rng(SEED + 2)
        enc_params, dec_params = P.init_tt_params(param_rng, contract, 1, 1)
        enc = M.Encoder(contract, depth=1, params=enc_params)
        dec = M.Decoder(contract, depth=1, params=dec_params)

        missing = dict(sd)
        del missing["encoder.patch_embed.weight"]
        with pytest.raises(KeyError, match="encoder.patch_embed.weight"):
            P.state_dict_to_tt(missing, enc, dec)

        wrong_shape = dict(sd)
        wrong_shape["encoder.patch_embed.weight"] = torch.zeros(3, 3)
        with pytest.raises(ValueError, match="encoder.patch_embed.weight"):
            P.state_dict_to_tt(wrong_shape, enc, dec)

        extra_key = dict(sd)
        extra_key["encoder.bogus_extra_param"] = torch.zeros(4)
        with pytest.raises(KeyError, match="bogus_extra_param"):
            P.state_dict_to_tt(extra_key, enc, dec)

        stray_prefix = dict(sd)
        stray_prefix["module.encoder.patch_embed.weight"] = torch.zeros(4)
        with pytest.raises(KeyError, match="module.encoder.patch_embed.weight"):
            P.state_dict_to_tt(stray_prefix, enc, dec)

        # And a valid state_dict still loads cleanly through the same encoder/decoder
        # instances after all of the above raised (proving no partial mutation left
        # them in a bad state -- see params.py's two-pass validate-then-write design).
        P.state_dict_to_tt(sd, enc, dec)
    finally:
        ctx.close_device()
