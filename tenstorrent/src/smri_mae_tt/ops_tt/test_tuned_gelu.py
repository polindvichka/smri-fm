"""Tests for `ops_tt/tuned_gelu.py`, the M6 tuned-GELU Tier-1 op.

Op-level correctness is verified with `perf_parity.run_op_parity` against
`ttml.ops.unary.gelu` (the existing, already-verified-correct op this is
meant to be a drop-in replacement for) at real ViT-L-relevant MLP-hidden
shapes (encoder `hidden=4096`, decoder `hidden=2048` -- `Mlp.__init__`'s
`hidden = int(dim * mlp_ratio)`), gated at `perf_parity.OP_FWD_PCC_GATE`/
`OP_BWD_PCC_GATE` (no gate loosening, per `perf_parity.py`'s module
docstring and this project's standing fail-fast convention). See
`tuned_gelu.py`'s module docstring for why forward (`FastLut`) and backward
(exact analytic `"none"`) using different approximation modes is the
intended pairing, not a bug -- these tests are the standing, gated
empirical check that backs that claim, not just the module's own ad hoc
isolated numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402

from smri_mae_tt import perf_parity as PP  # noqa: E402
from smri_mae_tt.ops_tt import tuned_gelu as TG  # noqa: E402

SEED = 2026 + 730


@pytest.fixture(autouse=False)
def _device():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        yield ctx
    finally:
        ctx.close_device()


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _reference_gelu_op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
    return ttml.ops.unary.gelu(leaves["x"])


def _candidate_gelu_op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
    return TG.tuned_gelu(leaves["x"])


@pytest.mark.parametrize(
    "label,batch,seq,hidden",
    [
        ("encoder_mlp_scale", 2, 64, 4096),
        ("decoder_mlp_scale", 2, 64, 2048),
    ],
)
def test_tuned_gelu_matches_default_gelu(_device, label, batch, seq, hidden):
    rng = np.random.default_rng(SEED)
    # realistic MLP-fc1-output activation scale (unit-normal-ish, not tiny) --
    # FastLut's ~1% relative error is a fixed fraction of the *input* scale,
    # so a too-small synthetic scale would under-exercise it relative to the
    # real model's actual fc1-output magnitudes.
    x = (rng.standard_normal((batch, 1, seq, hidden)) * 2.0).astype(np.float32)

    result = PP.run_op_parity(f"tuned_gelu-{label}", _reference_gelu_op, _candidate_gelu_op, {"x": x})

    assert result.forward.pcc >= PP.OP_FWD_PCC_GATE
    assert set(result.backward) == {"x"}
    assert result.min_backward_pcc() >= PP.OP_BWD_PCC_GATE
