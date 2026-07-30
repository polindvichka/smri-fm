"""Tests for `perf_parity.py`, the M6 custom-op verification harness.

These tests exercise the harness itself, not any tuned/optimized op --
M6's actual op rewrite is explicitly out of scope for this task (see
`perf_parity.py`'s module docstring). Both sides of every `run_op_parity`
call below are the *same*, already-verified-correct existing op
(`ttml.ops.linear.linear` / `ops_tt.attention.bidirectional_scaled_dot_
product_attention`) -- proving the harness reports near-perfect parity when
the two sides genuinely compute the same thing, and reports a loud gate
failure when they don't (a deliberately perturbed candidate).

Shapes: `embed_dim=256` (`in_f=out_f=256` for the linear test, `H=8/head_dim
=32` i.e. `dim=256` for the attention test) -- not `parity.FROZEN_FIXTURE`'s
tiny `embed_dim=64`. See `perf_parity.py`'s "Shape" section: per-parameter
gradient PCC at `embed_dim=64` is empirically known (`test_model_tt.py`'s
`_grad_parity_contract` docstring) to land as low as ~0.85 for some
parameters from bf16 rounding noise alone, even with a fully correct
backward -- a false-negative trap this file deliberately avoids by using the
same wider width `test_model_tt.py`'s own M4 gate test already switched to.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae_tt import perf_parity as PP  # noqa: E402
from smri_mae_tt.ops_tt.attention import bidirectional_scaled_dot_product_attention  # noqa: E402

SEED = 2026 + 900


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


# ---------------------------------------------------------------------------
# Gate constants
# ---------------------------------------------------------------------------


def test_gate_constants_match_project_convention():
    """`OP_FWD_PCC_GATE` mirrors the M2/M3 whole-model forward gate
    (`parity.DEFAULT_PCC_GATE`); `OP_BWD_PCC_GATE` mirrors the M4
    whole-model backward gate (`test_model_tt.py`'s `pcc_gate = 0.99`).
    See `perf_parity.py`'s module docstring, "Gate thresholds and why"."""
    assert PP.OP_FWD_PCC_GATE == 0.999
    assert PP.OP_BWD_PCC_GATE == 0.99


# ---------------------------------------------------------------------------
# Linear/matmul op family (QKV/MLP projections)
# ---------------------------------------------------------------------------


def _linear_inputs(rng: np.random.Generator, batch: int, seq: int, dim: int) -> dict[str, np.ndarray]:
    return {
        "x": rng.standard_normal((batch, seq, dim)).astype(np.float32) * 0.1,
        "weight": rng.standard_normal((dim, dim)).astype(np.float32) * 0.02,
        "bias": rng.standard_normal((dim,)).astype(np.float32) * 0.01,
    }


def _linear_op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
    return ttml.ops.linear.linear(leaves["x"], leaves["weight"], leaves["bias"])


def test_run_op_parity_passes_for_identical_linear_op(_device):
    rng = np.random.default_rng(SEED)
    inputs = _linear_inputs(rng, batch=2, seq=64, dim=256)

    result = PP.run_op_parity("linear", _linear_op, _linear_op, inputs)

    assert result.forward.pcc == pytest.approx(1.0, abs=1e-6)
    assert set(result.backward) == {"x", "weight", "bias"}
    assert result.min_backward_pcc() == pytest.approx(1.0, abs=1e-6)


def test_run_op_parity_fails_for_perturbed_linear_candidate(_device):
    rng = np.random.default_rng(SEED + 1)
    inputs = _linear_inputs(rng, batch=2, seq=64, dim=256)
    # A deliberately wrong "candidate" weight: scaled+shifted host-side,
    # well past the 0.999 forward gate (empirically ~0.91 PCC at this
    # perturbation, measured while building this harness) -- this must fail
    # loud, not silently pass.
    bad_weight_np = (inputs["weight"] * 1.05 + 0.01).astype(np.float32)

    def perturbed_linear_op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
        bad_weight = ttml.autograd.Tensor.from_numpy(bad_weight_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        bad_weight.set_requires_grad(True)
        return ttml.ops.linear.linear(leaves["x"], bad_weight, leaves["bias"])

    with pytest.raises(AssertionError, match="below gate"):
        PP.run_op_parity("linear-perturbed", _linear_op, perturbed_linear_op, inputs)


# ---------------------------------------------------------------------------
# Attention/SDPA op family
# ---------------------------------------------------------------------------


def _attention_inputs(rng: np.random.Generator, batch: int, heads: int, seq: int, head_dim: int) -> dict[str, np.ndarray]:
    shape = (batch, heads, seq, head_dim)
    return {
        "q": rng.standard_normal(shape).astype(np.float32) * 0.1,
        "k": rng.standard_normal(shape).astype(np.float32) * 0.1,
        "v": rng.standard_normal(shape).astype(np.float32) * 0.1,
    }


def _make_attention_op(seq_len: int) -> "PP.OpFn":
    def op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
        return bidirectional_scaled_dot_product_attention(leaves["q"], leaves["k"], leaves["v"], seq_len=seq_len)

    return op


def test_run_op_parity_passes_for_identical_attention_op(_device):
    # heads=8, head_dim=32 -> dim=256, matching this file's module-docstring
    # "avoid the tiny-embed_dim trap" shape, and head_dim=32 is tile-aligned
    # (modules_tt.Attention.__init__'s hard constraint on nlp_create_qkv_heads).
    seq_len = 64
    rng = np.random.default_rng(SEED + 2)
    inputs = _attention_inputs(rng, batch=2, heads=8, seq=seq_len, head_dim=32)
    op = _make_attention_op(seq_len)

    result = PP.run_op_parity("attention", op, op, inputs)

    assert result.forward.pcc == pytest.approx(1.0, abs=1e-6)
    assert set(result.backward) == {"q", "k", "v"}
    assert result.min_backward_pcc() == pytest.approx(1.0, abs=1e-6)


def test_run_op_parity_passes_for_attention_op_real_decoder_scale(_device):
    """Same self-comparison pattern as
    `test_run_op_parity_passes_for_identical_attention_op` above, but at
    `main_pretrain_tt.full_vitl_real_contract()`'s real decoder scale
    (B=24, H=16, S=2816, D=32) instead of a tiny smoke shape, gated at this
    project's standard `OP_FWD_PCC_GATE`/`OP_BWD_PCC_GATE` (not the
    near-1.0 `pytest.approx` used above) rather than loosened for size.

    Closes a real coverage gap found while investigating the SDPA-backward
    KV-kernel redundant-Q/`dO`-reread theory (`tenstorrent/docs/
    m6-optimization-results.md`, "Redundant Q/dO/O re-read theory" section):
    no committed test anywhere previously exercised `sdpa_bw`'s
    forward+backward at real production scale with a PCC gate -- the only
    real-scale numbers on record (`ops_tt/attention.py`'s `78.17ms ->
    75.50ms` / `402.68ms -> 359.09ms`) came from a one-off ad hoc
    measurement during the `AttentionMaskType::None` forward-kernel fix
    session, not a standing regression test.

    What this test can and cannot catch: `reference_fn`/`candidate_fn` are
    the *same* function (`bidirectional_scaled_dot_product_attention`),
    matching this file's established self-comparison convention (see the
    module docstring) -- so on its own it cannot catch a systematic
    correctness bug reproduced identically by both sides (that is what the
    new C++ `sdpa_bw_op_test.cpp::NIGHTLY_NoMask_ProductionEncoderShape`
    test, gated against a from-scratch CPU float reference at B=24,H=16,
    S=544,D=64, is for -- decoder scale S=2816 is infeasible for that CPU
    reference, which materializes a full `(B,H,S,S)` array). What it *does*
    catch at real batch/core-grid scale: shape/OOM failures and a silently
    disconnected backward graph for q/k/v -- and, per this exact op
    family's own history, it has a real chance of catching data-dependent
    bugs like the `AttentionMaskType::None` forward-kernel bug fixed
    earlier this project (reading uninitialized L1 as mask data), which was
    invisible at tiny scale and only manifested at real batch/core-grid
    scale: two independently-built device tensor sets computing the same
    (possibly buggy) op can diverge from each other if a bug's symptom
    depends on incidental L1/DRAM allocation history, not purely on input
    data.
    """
    seq_len = 2816
    rng = np.random.default_rng(SEED + 4)
    inputs = _attention_inputs(rng, batch=24, heads=16, seq=seq_len, head_dim=32)
    op = _make_attention_op(seq_len)

    result = PP.run_op_parity("attention-decoder-scale", op, op, inputs)

    assert result.forward.pcc >= PP.OP_FWD_PCC_GATE
    assert set(result.backward) == {"q", "k", "v"}
    assert result.min_backward_pcc() >= PP.OP_BWD_PCC_GATE


# ---------------------------------------------------------------------------
# run_op_parity's own input validation / gradient-connectivity checks
# ---------------------------------------------------------------------------


def test_run_op_parity_rejects_empty_input_arrays(_device):
    with pytest.raises(ValueError, match="empty"):
        PP.run_op_parity("empty", _linear_op, _linear_op, {})


def test_run_op_parity_reports_missing_gradient_loudly(_device):
    rng = np.random.default_rng(SEED + 3)
    inputs = _linear_inputs(rng, batch=2, seq=64, dim=256)

    def op_dropping_bias_grad(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
        # Uses bias as a *value* but not through an autograd-connected op
        # (ttnn.add is a raw op, not ttml.ops), so the graph never reaches
        # this leaf -- exactly the "candidate silently drops a leaf from its
        # backward graph" bug run_op_parity must catch, not skip past.
        y = ttml.ops.linear.linear(leaves["x"], leaves["weight"], None)
        return y

    with pytest.raises(AssertionError, match="no gradient"):
        PP.run_op_parity("drops-bias-grad", op_dropping_bias_grad, op_dropping_bias_grad, inputs)
