"""Tests for `checkpoint_tt.py`: save/load round trip and fail-fast checks.

Round-trip fidelity is checked at the *forward output* level (not just
"the state_dict keys match"), the same principle `test_params.py`'s own
round-trip test uses: build a small TT `Encoder`/`Decoder` pair, save a
checkpoint, load it into a *second*, differently-initialized pair, and
confirm both pairs now produce the same forward output (PCC >= 0.999 --
`checkpoint_tt`'s round trip is TT-bf16 -> fp32 -> TT-bf16, which per
`params.py`'s module docstring should lose *far* less precision than a
fresh fp32-PyTorch -> TT round trip, so the tight `test_params.py` PCC gate
applies here too, not a looser one).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae_tt import checkpoint_tt  # noqa: E402
from smri_mae_tt import modules_tt as M  # noqa: E402
from smri_mae_tt import params as P  # noqa: E402
from smri_mae_tt.parity import FROZEN_FIXTURE, compare, to_numpy_fp32  # noqa: E402

PCC_GATE = 0.999


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _contract():
    return FROZEN_FIXTURE.contract


def test_checkpoint_save_load_round_trip(tmp_path):
    contract = _contract()
    encoder_depth, decoder_depth = 2, 1

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        # --- Model A: the one we save. ---
        rng_a = np.random.default_rng(FROZEN_FIXTURE.param_seed)
        enc_a_params, dec_a_params = P.init_tt_params(rng_a, contract, encoder_depth, decoder_depth)
        enc_a = M.Encoder(contract, depth=encoder_depth, params=enc_a_params)
        dec_a = M.Decoder(contract, depth=decoder_depth, params=dec_a_params)

        # --- Model B: differently initialized, will be overwritten by the checkpoint. ---
        rng_b = np.random.default_rng(FROZEN_FIXTURE.param_seed + 999)
        enc_b_params, dec_b_params = P.init_tt_params(rng_b, contract, encoder_depth, decoder_depth)
        enc_b = M.Encoder(contract, depth=encoder_depth, params=enc_b_params)
        dec_b = M.Decoder(contract, depth=decoder_depth, params=dec_b_params)

        # --- Shared input data. ---
        data_rng = np.random.default_rng(FROZEN_FIXTURE.param_seed + 1)
        batch, num_patches = contract.batch, contract.num_patches
        visible_ids = np.zeros((batch, contract.l_vis), dtype=np.uint32)
        pred_ids = np.zeros((batch, contract.q_pred), dtype=np.uint32)
        for b in range(batch):
            perm = data_rng.permutation(num_patches)
            visible_ids[b] = perm[: contract.l_vis]
            pred_ids[b] = perm[contract.l_vis : contract.l_vis + contract.q_pred]
        visible_values = data_rng.standard_normal((batch, contract.l_vis, contract.patch_dim)).astype(np.float32)

        # --- Sanity: A and B disagree before the checkpoint is loaded. ---
        cls_a0, patch_a0 = enc_a.forward(visible_values, visible_ids)
        ttnn.synchronize_device(ctx.get_device())
        cls_b0, patch_b0 = enc_b.forward(visible_values, visible_ids)
        ttnn.synchronize_device(ctx.get_device())
        assert not np.allclose(to_numpy_fp32(cls_a0), to_numpy_fp32(cls_b0), atol=1e-3)

        # --- Save A, load into B. ---
        checkpoint_dir = str(tmp_path / "ckpt")
        path = checkpoint_tt.save_checkpoint(enc_a, dec_a, step=42, checkpoint_dir=checkpoint_dir)
        assert path == checkpoint_tt.checkpoint_path(checkpoint_dir, 42)
        import os

        assert os.path.isfile(path)

        payload = checkpoint_tt.load_checkpoint(path, enc_b, dec_b)
        assert payload["step"] == 42

        # --- A and B now agree (round trip through the checkpoint). ---
        cls_a, patch_a = enc_a.forward(visible_values, visible_ids)
        ttnn.synchronize_device(ctx.get_device())
        pred_a = dec_a.forward(patch_a, visible_ids, pred_ids)
        ttnn.synchronize_device(ctx.get_device())

        cls_b, patch_b = enc_b.forward(visible_values, visible_ids)
        ttnn.synchronize_device(ctx.get_device())
        pred_b = dec_b.forward(patch_b, visible_ids, pred_ids)
        ttnn.synchronize_device(ctx.get_device())

        cls_a_np, patch_a_np, pred_a_np = to_numpy_fp32(cls_a), to_numpy_fp32(patch_a), to_numpy_fp32(pred_a)
        cls_b_np, patch_b_np, pred_b_np = to_numpy_fp32(cls_b), to_numpy_fp32(patch_b), to_numpy_fp32(pred_b)

        compare("checkpoint round trip: cls_embed", cls_a_np, cls_b_np, pcc_gate=PCC_GATE)
        compare("checkpoint round trip: patch_embeds", patch_a_np, patch_b_np, pcc_gate=PCC_GATE)
        compare("checkpoint round trip: decoder preds", pred_a_np, pred_b_np, pcc_gate=PCC_GATE)
    finally:
        ctx.close_device()


def test_load_checkpoint_missing_file_raises(tmp_path):
    contract = _contract()
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        rng = np.random.default_rng(FROZEN_FIXTURE.param_seed)
        enc_params, dec_params = P.init_tt_params(rng, contract, 1, 1)
        enc = M.Encoder(contract, depth=1, params=enc_params)
        dec = M.Decoder(contract, depth=1, params=dec_params)

        with pytest.raises(FileNotFoundError):
            checkpoint_tt.load_checkpoint(str(tmp_path / "does_not_exist.pt"), enc, dec)
    finally:
        ctx.close_device()


def test_load_checkpoint_malformed_payload_raises(tmp_path):
    import torch

    contract = _contract()
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        rng = np.random.default_rng(FROZEN_FIXTURE.param_seed)
        enc_params, dec_params = P.init_tt_params(rng, contract, 1, 1)
        enc = M.Encoder(contract, depth=1, params=enc_params)
        dec = M.Decoder(contract, depth=1, params=dec_params)

        bad_path = tmp_path / "bad.pt"
        torch.save({"not_a_state_dict": 1}, bad_path)
        with pytest.raises(KeyError, match="state_dict"):
            checkpoint_tt.load_checkpoint(str(bad_path), enc, dec)
    finally:
        ctx.close_device()
