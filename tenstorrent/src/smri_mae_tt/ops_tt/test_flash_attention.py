"""Test-first harness for the not-yet-written M6 Tier-1 chunked flash-attention
op, `ops_tt/flash_attention.py::chunked_bidirectional_attention`.

## Why this file exists before the implementation

Per project convention ("Test before rewrite" -- risky rewrites get a test
harness + golden-baseline comparison written first, by someone who is not
writing the implementation, so the test cannot be shaped to fit shortcuts in
the implementation). This file is that harness. It is written and committed
*before* `flash_attention.py` exists, so every test below currently fails at
collection with `ModuleNotFoundError` -- that is expected and correct, not a
bug in this file. It stays red until a later step implements the op.

## What is being replaced, and why

`bidirectional_scaled_dot_product_attention`
(`smri_mae_tt.ops_tt.attention`) already gives correct bidirectional
self-attention via `ttml`'s fused `sdpa_fw`/`sdpa_bw` kernels, and is already
used for training. But at the real decoder shape
(`main_pretrain_tt.full_vitl_real_contract()`: batch=24, heads=16,
seq_len=2816, head_dim=32) it is measured to cost ~224ms forward+backward per
call, x4 decoder layers = ~896ms of a ~5.8s/step total -- the single largest
line item in a real training step. A separate, much faster tunable op
(`ttnn.transformer.scaled_dot_product_attention` +
`ttnn.SDPAProgramConfig`) exists but has no backward anywhere in this
tt-metal build, so it cannot be used for training as-is.

`chunked_bidirectional_attention` is meant to be a hand-derived,
flash-attention-2-style forward AND backward (chunked, online-softmax),
built only from existing `ttnn`/`ttml` primitives in a Python loop -- no new
C++/Metal kernel, no vendored `tenstorrent/tt-metal/` edits -- that should be
much faster than the current fused-but-uncontrollable kernel at this long
sequence length, while remaining numerically correct. It is a drop-in
replacement for `bidirectional_scaled_dot_product_attention` specifically
(same shape/dtype/semantics contract: BFLOAT16, TILE layout, plain
bidirectional self-attention, every query attends every key, no masking, no
causal behavior) -- not a general-purpose SDPA (no GQA/causal support is
tested or required).

## How correctness is checked

Via `perf_parity.run_op_parity`, exactly like every other M6 op-parity test
in this codebase (`test_tuned_linear.py`): forward output and every leaf's
backward gradient are compared, via PCC, against the *existing,
already-verified* `bidirectional_scaled_dot_product_attention` as the
reference -- not against a fresh numpy re-derivation, since the point here is
"does the new chunked op match the op it's replacing," which
`bidirectional_scaled_dot_product_attention` itself already established
against a from-scratch numpy reference in `test_attention.py`. Gates are
`perf_parity.OP_FWD_PCC_GATE` (0.999) / `OP_BWD_PCC_GATE` (0.99), unloosened
-- see `perf_parity.py`'s module docstring for why loosening a gate is not
the correct response to a low measured PCC (re-measure at a wider shape
instead).

`_reset_state` clears the autograd graph between tests. (Older revisions of
this file also called `attention.clear_mask_cache()` here --
`bidirectional_scaled_dot_product_attention` used to build/cache an
all-ones mask tensor keyed by `seq_len`, and a stale cached mask from a
previous test's closed device context was not valid on a freshly reopened
one. It now goes through the fused kernel's `no_mask=True`/
`AttentionMaskType::None` path instead of an explicit mask tensor, so there
is no cache to invalidate -- see `ops_tt/attention.py`'s module docstring.)
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae_tt import perf_parity as PP  # noqa: E402
from smri_mae_tt.ops_tt.attention import bidirectional_scaled_dot_product_attention  # noqa: E402
from smri_mae_tt.ops_tt.flash_attention import chunked_bidirectional_attention  # noqa: E402

SEED = 2026 + 623


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
# Op-level parity vs. bidirectional_scaled_dot_product_attention.
# ---------------------------------------------------------------------------


def _attention_inputs(rng: np.random.Generator, batch: int, heads: int, seq_len: int, head_dim: int) -> dict[str, np.ndarray]:
    shape = (batch, heads, seq_len, head_dim)
    return {
        "q": rng.standard_normal(shape, dtype=np.float32),
        "k": rng.standard_normal(shape, dtype=np.float32),
        "v": rng.standard_normal(shape, dtype=np.float32),
    }


def _make_reference_op(seq_len: int) -> "PP.OpFn":
    def op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
        return bidirectional_scaled_dot_product_attention(leaves["q"], leaves["k"], leaves["v"], seq_len=seq_len)

    return op


def _make_candidate_op(seq_len: int, q_chunk_size: int, k_chunk_size: int) -> "PP.OpFn":
    def op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
        return chunked_bidirectional_attention(
            leaves["q"],
            leaves["k"],
            leaves["v"],
            seq_len=seq_len,
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
        )

    return op


def test_chunked_attention_matches_reference_small(_device):
    """Smallest shape that still forces the multi-chunk online-softmax
    accumulation path on both axes: seq_len=128 with q_chunk_size=
    k_chunk_size=64 gives 2 query chunks x 2 key chunks (4 iterations of the
    online-softmax update), not one giant chunk covering all of S -- a
    q/k_chunk_size >= seq_len candidate would vacuously pass a parity check
    without ever exercising accumulation across chunks.
    """
    batch, heads, seq_len, head_dim = 1, 2, 128, 32
    q_chunk_size, k_chunk_size = 64, 64
    rng = np.random.default_rng(SEED)
    inputs = _attention_inputs(rng, batch, heads, seq_len, head_dim)

    result = PP.run_op_parity(
        "flash_attn-small",
        _make_reference_op(seq_len),
        _make_candidate_op(seq_len, q_chunk_size, k_chunk_size),
        inputs,
    )

    assert result.forward.pcc >= PP.OP_FWD_PCC_GATE
    assert set(result.backward) == {"q", "k", "v"}
    assert result.min_backward_pcc() >= PP.OP_BWD_PCC_GATE


def test_chunked_attention_matches_reference_nonsquare_chunks(_device):
    """Q and K use *different* chunk sizes (128 vs 64) -- a real
    implementation may legitimately pick different values for each axis
    (e.g. to fit different SRAM budgets for the running Q chunk vs the
    streamed K/V chunks), so this must be exercised independently of the
    square-chunk case above. seq_len=256 gives 2 query chunks x 4 key
    chunks.
    """
    batch, heads, seq_len, head_dim = 1, 2, 256, 32
    q_chunk_size, k_chunk_size = 128, 64
    rng = np.random.default_rng(SEED + 1)
    inputs = _attention_inputs(rng, batch, heads, seq_len, head_dim)

    result = PP.run_op_parity(
        "flash_attn-nonsquare_chunks",
        _make_reference_op(seq_len),
        _make_candidate_op(seq_len, q_chunk_size, k_chunk_size),
        inputs,
    )

    assert result.forward.pcc >= PP.OP_FWD_PCC_GATE
    assert set(result.backward) == {"q", "k", "v"}
    assert result.min_backward_pcc() >= PP.OP_BWD_PCC_GATE


def test_chunked_attention_matches_reference_real_decoder_head_shape(_device):
    """Real decoder per-head dims (heads=16, head_dim=32 --
    `main_pretrain_tt.full_vitl_real_contract()`), batch/seq_len reduced from
    the real B=24/S=2816 to keep pytest fast while staying tile-aligned.
    seq_len=512 with q_chunk_size=k_chunk_size=256 is a real, documented
    choice (not arbitrary): it is the smallest split of this seq_len that
    still yields >1 chunk per axis (2x2), matching the "must actually
    chunk" requirement of the two tests above, at the shape that actually
    matters for the ~224ms/call cost this op exists to fix. The 512 default
    `q_chunk_size`/`k_chunk_size` in `chunked_bidirectional_attention`'s own
    signature would collapse this seq_len to a single chunk, so it is
    deliberately not used here.
    """
    batch, heads, seq_len, head_dim = 2, 16, 512, 32
    q_chunk_size, k_chunk_size = 256, 256
    rng = np.random.default_rng(SEED + 2)
    inputs = _attention_inputs(rng, batch, heads, seq_len, head_dim)

    result = PP.run_op_parity(
        "flash_attn-real_decoder_head_shape",
        _make_reference_op(seq_len),
        _make_candidate_op(seq_len, q_chunk_size, k_chunk_size),
        inputs,
    )

    assert result.forward.pcc >= PP.OP_FWD_PCC_GATE
    assert set(result.backward) == {"q", "k", "v"}
    assert result.min_backward_pcc() >= PP.OP_BWD_PCC_GATE


# ---------------------------------------------------------------------------
# Fail-fast: non-tile-divisible chunk sizes must raise, never silently
# round/pad/clamp (project convention -- see "Fail-fast, no fallbacks").
# ---------------------------------------------------------------------------


def _dummy_qkv(rng: np.random.Generator, batch: int, heads: int, seq_len: int, head_dim: int) -> tuple[
    "ttml.autograd.Tensor", "ttml.autograd.Tensor", "ttml.autograd.Tensor"
]:
    shape = (batch, heads, seq_len, head_dim)
    tensors = []
    for _ in range(3):
        arr = rng.standard_normal(shape, dtype=np.float32)
        tensors.append(ttml.autograd.Tensor.from_numpy(arr, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16))
    return tuple(tensors)  # type: ignore[return-value]


def test_chunked_attention_rejects_seq_len_not_divisible_by_q_chunk_size(_device):
    """seq_len=128 is not evenly divisible by q_chunk_size=96 (128 % 96 ==
    32 != 0) even though 96 itself is tile-aligned -- this isolates the
    "does not evenly divide seq_len" failure from a separate
    tile-alignment failure, and must raise rather than silently rounding or
    padding the last chunk.
    """
    batch, heads, seq_len, head_dim = 1, 1, 128, 32
    rng = np.random.default_rng(SEED + 3)
    q, k, v = _dummy_qkv(rng, batch, heads, seq_len, head_dim)

    with pytest.raises(ValueError):
        chunked_bidirectional_attention(q, k, v, seq_len=seq_len, q_chunk_size=96, k_chunk_size=128)


def test_chunked_attention_rejects_seq_len_not_divisible_by_k_chunk_size(_device):
    """Mirror of the q_chunk_size case above, for k_chunk_size: seq_len=128
    is not evenly divisible by k_chunk_size=96."""
    batch, heads, seq_len, head_dim = 1, 1, 128, 32
    rng = np.random.default_rng(SEED + 4)
    q, k, v = _dummy_qkv(rng, batch, heads, seq_len, head_dim)

    with pytest.raises(ValueError):
        chunked_bidirectional_attention(q, k, v, seq_len=seq_len, q_chunk_size=128, k_chunk_size=96)
