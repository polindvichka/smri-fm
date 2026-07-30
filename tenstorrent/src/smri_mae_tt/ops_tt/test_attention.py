"""M0-gate regression test: `ttml`'s fused SDPA kernel, driven through
`smri_mae_tt.ops_tt.attention`, produces true bidirectional attention (not
the library's causal default) to PCC >= 0.999 against a from-scratch numpy
reference.

`ttml` is only importable once tt-metal has been built with
`--build-tt-train` (TENSTORRENT_PORT.md section 0: "Python `ttml` ... absent
-- must be built"). `pytest.importorskip("ttml")` is the one legitimate skip
here -- an environment-availability skip, not a fallback masking a bug (see
`feedback_fail_fast` memory). No other skip/try-except is used: once `ttml`
is importable, opening the device and running the assertions below is not
optional.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae_tt.ops_tt.attention import bidirectional_scaled_dot_product_attention  # noqa: E402

SEED = 2026
PCC_GATE = 0.999


def _numpy_bidirectional_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """From-scratch reference: softmax(Q @ K^T / sqrt(d)) @ V, every query
    attending to every key (no mask, no causality). q/k/v: (B, H, S, D).
    """
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q/k/v shapes must match for this self-attention reference, got {q.shape}/{k.shape}/{v.shape}")
    d = q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    scores = np.einsum("bhqd,bhkd->bhqk", q, k) * scale
    # subtract per-row max for numerical stability; does not change the result
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", weights, v)


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation coefficient over the flattened tensors."""
    af = a.reshape(-1).astype(np.float64)
    bf = b.reshape(-1).astype(np.float64)
    af = af - af.mean()
    bf = bf - bf.mean()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0.0:
        raise ValueError("PCC is undefined here: one of the compared tensors is constant (zero variance)")
    return float(np.dot(af, bf) / denom)


def _to_numpy_fp32(tensor: "ttml.autograd.Tensor") -> np.ndarray:
    return np.asarray(tensor.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_bidirectional_attention_matches_numpy_reference():
    """The M0 gate from TENSTORRENT_PORT.md section 7: ttml bidirectional
    SDPA (driven through the mask-based Arbitrary path, not a causal one)
    matches an independent bidirectional reference to PCC >= 0.999.
    """
    batch, heads, seq_len, head_dim = 2, 4, 64, 32

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        device = ctx.get_device()

        rng = np.random.default_rng(SEED)
        q_np = rng.standard_normal((batch, heads, seq_len, head_dim), dtype=np.float32)
        k_np = rng.standard_normal((batch, heads, seq_len, head_dim), dtype=np.float32)
        v_np = rng.standard_normal((batch, heads, seq_len, head_dim), dtype=np.float32)

        query = ttml.autograd.Tensor.from_numpy(q_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        key = ttml.autograd.Tensor.from_numpy(k_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        value = ttml.autograd.Tensor.from_numpy(v_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

        # Read the BFLOAT16-quantized Q/K/V back so the numpy reference is
        # computed from the exact values the device kernel sees -- otherwise
        # bf16 rounding shows up as spurious PCC error unrelated to the
        # thing under test (masking behavior).
        q_bf16 = _to_numpy_fp32(query)
        k_bf16 = _to_numpy_fp32(key)
        v_bf16 = _to_numpy_fp32(value)

        out = bidirectional_scaled_dot_product_attention(query, key, value, seq_len=seq_len)
        ttnn.synchronize_device(device)
        out_np = _to_numpy_fp32(out)

        reference = _numpy_bidirectional_attention(q_bf16, k_bf16, v_bf16)

        assert out_np.shape == reference.shape, f"shape mismatch: device={out_np.shape} reference={reference.shape}"

        pcc = _pearson_corr(out_np, reference)
        assert pcc >= PCC_GATE, f"bidirectional SDPA PCC={pcc:.6f} below M0 gate {PCC_GATE}"
    finally:
        ctx.close_device()


def test_bidirectional_attention_differs_from_causal_reference():
    """Sanity check on the test itself: if the kernel were silently applying
    its causal default (the exact bug this module exists to avoid), output
    would match a *causal* reference far better than the bidirectional one,
    and would NOT match the bidirectional reference to PCC >= 0.999 in
    general. This guards against a vacuously-passing first test (e.g. a
    seq_len=1 edge case where causal and bidirectional coincide).
    """
    batch, heads, seq_len, head_dim = 1, 2, 64, 32

    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        device = ctx.get_device()

        rng = np.random.default_rng(SEED + 1)
        q_np = rng.standard_normal((batch, heads, seq_len, head_dim), dtype=np.float32)
        k_np = rng.standard_normal((batch, heads, seq_len, head_dim), dtype=np.float32)
        v_np = rng.standard_normal((batch, heads, seq_len, head_dim), dtype=np.float32)

        query = ttml.autograd.Tensor.from_numpy(q_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        key = ttml.autograd.Tensor.from_numpy(k_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        value = ttml.autograd.Tensor.from_numpy(v_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

        q_bf16 = _to_numpy_fp32(query)
        k_bf16 = _to_numpy_fp32(key)
        v_bf16 = _to_numpy_fp32(value)

        out = bidirectional_scaled_dot_product_attention(query, key, value, seq_len=seq_len)
        ttnn.synchronize_device(device)
        out_np = _to_numpy_fp32(out)

        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.einsum("bhqd,bhkd->bhqk", q_bf16, k_bf16) * scale
        scores = np.where(causal_mask[None, None, :, :], scores, -np.inf)
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        causal_reference = np.einsum("bhqk,bhkd->bhqd", weights, v_bf16)

        causal_pcc = _pearson_corr(out_np, causal_reference)
        bidirectional_reference = _numpy_bidirectional_attention(q_bf16, k_bf16, v_bf16)
        bidirectional_pcc = _pearson_corr(out_np, bidirectional_reference)

        assert bidirectional_pcc >= PCC_GATE, f"bidirectional SDPA PCC={bidirectional_pcc:.6f} below M0 gate {PCC_GATE}"
        assert bidirectional_pcc > causal_pcc, (
            f"device output correlates at least as well with a causal reference "
            f"(PCC={causal_pcc:.6f}) as with the bidirectional one "
            f"(PCC={bidirectional_pcc:.6f}) -- this would indicate the causal "
            f"default from ttml::ops::scaled_dot_product_attention is still "
            f"being applied despite the mask."
        )
    finally:
        ctx.close_device()
