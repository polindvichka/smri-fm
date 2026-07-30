"""Custom-op verification harness for M6 (performance) work.

TENSTORRENT_PORT.md section 7 lists M6 ("tune SDPAProgramConfig / consider a
custom kernel") as future work. A first M6 pass (this task) found concrete,
measured speedup opportunities for both the matmul/linear ops (QKV/MLP
projections) and attention/SDPA, but explicitly did **not** implement either
-- the user wants a correctness-verification harness and golden baselines
*first*, so a later rewrite agent has something to check its work against
instead of "the model still trains, looks fine to me."

This module is that harness. It does not implement, tune, or benchmark any
op itself -- see `test_perf_parity.py` for the only op-level comparisons in
this file, and note both sides of every comparison there are *existing*,
already-verified ops (exercising the harness itself), never a new tuned
kernel.

## How a future M6-rewrite agent plugs in a candidate op

Write two small closures, each taking a `dict[str, ttml.autograd.Tensor]` of
*leaf* inputs (the tensors you want forward/backward parity numbers for) and
returning one `ttml.autograd.Tensor` output. Anything that is not itself a
gradient-bearing tensor -- `seq_len`, a `program_config`, a dtype policy --
is closed over by the lambda, not put in the leaves dict:

    def reference_fn(leaves):
        return bidirectional_scaled_dot_product_attention(
            leaves["q"], leaves["k"], leaves["v"], seq_len=S
        )

    def candidate_fn(leaves):
        return my_new_tuned_sdpa(leaves["q"], leaves["k"], leaves["v"], seq_len=S)

    result = run_op_parity(
        "attention",
        reference_fn,
        candidate_fn,
        {"q": q_np, "k": k_np, "v": v_np},  # numpy arrays, same shape
    )

`run_op_parity` (below) builds two independent sets of device leaf tensors
from the *same* numpy arrays (both sides see bit-identical input data),
runs both forwards, compares outputs via `parity.compare()`, then reduces
each output to a scalar with a small fixed sum-of-squares-mean reduction
(`_scalar_mean_square` below, needed because `ttml.ops.unary.mean`'s
built-in backward only supports rank-4 inputs -- see that function's
docstring) and calls `.backward(False)` on each side, comparing every
leaf's gradient the same way via `parity.compare()`.

`parity.compare()` already asserts its own PCC gate and raises
`AssertionError` with the full PCC/max-abs-err/rel-Frobenius-err triple on
failure, so `run_op_parity` gates automatically at the thresholds below
(pass `fwd_pcc_gate=0.0`/`bwd_pcc_gate=0.0` to record without gating, the
same escape hatch `compare()` itself documents). This module reuses
`parity.ParityReport`/`parity.compare()` for every number it reports --
nothing here reimplements PCC/max-abs-err/rel-Frobenius-err.

## Gate thresholds and why

- `OP_FWD_PCC_GATE = 0.999` -- identical to the M2/M3 whole-encoder/decoder
  forward gate (`parity.DEFAULT_PCC_GATE`). An isolated op's forward output
  is *closer* to its reference than a whole-model forward pass (no
  compounding across 24+ layers), so there is no reason to loosen this below
  what the whole model already has to clear: if a single op cannot hit 0.999
  forward PCC against its own reference in isolation, wiring it into the
  model only makes things worse, not better.

- `OP_BWD_PCC_GATE = 0.99` -- identical to the M4 whole-model gate
  (`test_model_tt.py::test_model_backward_parity_m4_gate`'s `pcc_gate =
  0.99`). Backward compounds bf16 rounding noise through more ops than
  forward alone (that test's own docstring); M4 already had to loosen from
  0.999 to 0.99 for exactly that reason at the *whole-model* level, where
  compounding is worst. An isolated op's own backward compounds less than a
  full 24-layer model backward does, so asking it to clear the same 0.99 bar
  is, if anything, conservative rather than lenient. Kept identical (not
  loosened further) so a candidate that clears this op-level gate is not
  then surprised by a stricter one once it is wired into the real model
  gate.

- Neither gate is loosened for M6 candidates just because they are "only" an
  isolated op swap. This project's own M4 finding (see "Shape" below) is
  that a low measured PCC usually means "measured at too small a shape for
  bf16 noise to average out," not "the op is actually less correct" -- so
  the fix for a real candidate that fails these gates is to re-measure at a
  wider shape, never to widen the gate.

## Shape: avoid the tiny-embed_dim false-negative trap

`test_model_tt.py`'s `_grad_parity_contract()` docstring documents an
empirically-confirmed pitfall: per-parameter gradient PCC at
`parity.FROZEN_FIXTURE`'s tiny `embed_dim=64` lands as low as ~0.85 for some
LayerNorm parameters *even when the backward is correct* -- bf16 rounding
noise does not average out across a narrow reduction dimension. Widening to
`embed_dim=256`/`decoder_embed_dim=128` (`head_dim=32` preserved) made every
parameter clear PCC>=0.99 with the same backward code. `test_perf_parity.py`
exercises `run_op_parity` at that same `embed_dim=256` scale (not
`FROZEN_FIXTURE`) for exactly that reason; a future M6 op's own parity test
should do the same, or go straight to `embed_dim=1024` (real ViT-L) if the
candidate op supports it directly.

## Golden-baseline regression gate

See `test_golden_baseline.py`: a handful of real `main_pretrain_tt.train()`
steps at `main_pretrain_tt.smoke_contract()` scale, run against the current
(slow but already-verified-correct) implementation, with per-step loss and
grad-norm hardcoded and checked with a tight, documented tolerance. This
catches "the fast version is subtly wrong end-to-end" even when an isolated
op's own `run_op_parity` numbers above look fine -- it is a different kind
of check (whole-pipeline regression, not isolated-op parity), deliberately
kept separate rather than folded into `run_op_parity`. That file's docstring
documents exactly where its hardcoded numbers came from and how to
regenerate them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import ttml
import ttnn
from ttml.autograd import Function

from smri_mae_tt.parity import ParityReport, compare, to_numpy_fp32

# ---------------------------------------------------------------------------
# Gate thresholds -- see module docstring, "Gate thresholds and why".
# ---------------------------------------------------------------------------

OP_FWD_PCC_GATE = 0.999
OP_BWD_PCC_GATE = 0.99

OpFn = Callable[[dict[str, "ttml.autograd.Tensor"]], "ttml.autograd.Tensor"]


# ---------------------------------------------------------------------------
# Scalar reduction: turns an arbitrary-rank op output into something
# `.backward(False)` can be called on directly.
# ---------------------------------------------------------------------------


class _ScalarMeanSquare(Function):
    """`out -> mean(out**2)`, a `[1, 1]` scalar, autograd-connected, any rank.

    Not `ttml.ops.unary.mean(ttml.ops.binary.mul(out, out))`: empirically
    confirmed (while building this harness) that `ttml.ops.unary.mean`'s
    backward (`ttnn.moreh_mean_backward`) only works for rank-4 input
    tensors -- it raises `TT_FATAL ... dim < rank` (`dim` set to the input's
    own rank, "reduce over everything") for rank-3 inputs, which is exactly
    the shape a raw `[B, S, D]` op output (e.g. `bidirectional_scaled_dot_
    product_attention`'s fused-heads form, or a bare `ttml.ops.linear.linear`
    output) has. This is therefore a small, hand-written Tier-1
    `ttml.autograd.Function` (the same pattern as `model_tt._SqueezeDecoder
    PredAxis` and `loss_tt.PerScanValidVoxelMSELoss`): forward built from raw
    `ttnn` reduction primitives that work at any rank, explicit closed-form
    backward. This is test-harness plumbing, not a modeling op -- it exists
    only so `run_op_parity` can call `.backward(False)` on any op's output
    regardless of its rank.

    Forward:  loss = sum(x**2) / N,  N = x.numel()
    Backward: d(loss)/dx = grad_output * 2 * x / N
    (`grad_output` is the incoming `[1, 1]` upstream scalar gradient -- 1.0
    for a direct top-level `.backward()`, same convention as every other
    scalar loss in this port, e.g. `loss_tt.py`'s module docstring.)
    """

    @staticmethod
    def forward(ctx, x: "ttml.autograd.Tensor"):
        val = x.get_value()
        val32 = ttnn.typecast(val, ttnn.DataType.FLOAT32)
        total = ttnn.multiply(val32, val32)
        for _ in range(len(val.shape)):
            total = ttnn.sum(total, dim=-1, keepdim=False)
        n = 1
        for s in tuple(val.shape):
            n *= int(s)
        if n <= 0:
            raise ValueError(f"_ScalarMeanSquare: input has non-positive element count {n} (shape {tuple(val.shape)})")
        total = ttnn.reshape(total, ttnn.Shape([1, 1]))
        ctx.save_for_backward(x)
        ctx.n = n
        ctx.orig_dtype = val.dtype
        return ttnn.multiply(total, 1.0 / n)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        val32 = ttnn.typecast(x.get_value(), ttnn.DataType.FLOAT32)
        go32 = ttnn.typecast(grad_output, ttnn.DataType.FLOAT32)
        go_scalar = float(np.asarray(go32.to_numpy()).reshape(-1)[0])
        grad = ttnn.multiply(val32, (2.0 / ctx.n) * go_scalar)
        return ttnn.typecast(grad, ctx.orig_dtype)


def _scalar_mean_square(x: "ttml.autograd.Tensor") -> "ttml.autograd.Tensor":
    return _ScalarMeanSquare.apply(x)


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpParityResult:
    """Forward + per-leaf-backward `ParityReport`s for one `run_op_parity`
    call. `backward` is keyed by the same names as the `input_arrays` dict
    passed to `run_op_parity`."""

    forward: ParityReport
    backward: dict[str, ParityReport] = field(default_factory=dict)

    def min_backward_pcc(self) -> float:
        if not self.backward:
            raise ValueError("min_backward_pcc: no backward comparisons were recorded")
        return min(r.pcc for r in self.backward.values())


def _leaf_tensors(
    input_arrays: dict[str, np.ndarray], dtype: "ttnn.DataType"
) -> dict[str, "ttml.autograd.Tensor"]:
    leaves = {}
    for name, arr in input_arrays.items():
        arr32 = np.asarray(arr, dtype=np.float32)
        t = ttml.autograd.Tensor.from_numpy(arr32, layout=ttnn.Layout.TILE, new_type=dtype)
        t.set_requires_grad(True)
        leaves[name] = t
    return leaves


def run_op_parity(
    name: str,
    reference_fn: OpFn,
    candidate_fn: OpFn,
    input_arrays: dict[str, np.ndarray],
    *,
    dtype: "ttnn.DataType" = ttnn.DataType.BFLOAT16,
    fwd_pcc_gate: float = OP_FWD_PCC_GATE,
    bwd_pcc_gate: float = OP_BWD_PCC_GATE,
) -> OpParityResult:
    """Run `reference_fn` and `candidate_fn` on the same random input(s) and
    report/assert forward-output and per-input-gradient parity.

    `input_arrays` maps a leaf-input name to a numpy array. Two independent
    sets of device leaf tensors are built from those same arrays (so both
    sides see bit-identical data, uploaded separately -- this deliberately
    does not share tensor objects between the two graphs, matching how a
    real reference-vs-candidate comparison in a future rewrite would build
    two independent ops), each with `requires_grad=True`.

    Forward: both `*_fn(leaves)` are called, and their outputs compared via
    `parity.compare()` (PCC gated at `fwd_pcc_gate`).

    Backward: each output is reduced to a scalar via `_scalar_mean_square`
    (mean of squares -- any rank) and `.backward(False)` is called on each
    side independently. Every key in `input_arrays` is then compared via
    `parity.compare()` (PCC gated at `bwd_pcc_gate`) -- raises loudly if
    either side never populated a gradient for one of those leaves (a
    candidate that silently drops a leaf from its backward graph is exactly
    the kind of bug this harness exists to catch, not something to skip
    past).

    Every `compare()` call already asserts its own gate (raising
    `AssertionError` with the full PCC/max-abs-err/rel-Frobenius-err triple
    on failure); pass `fwd_pcc_gate=0.0`/`bwd_pcc_gate=0.0` to record without
    gating, same escape hatch `compare()` itself documents.
    """
    if not input_arrays:
        raise ValueError("run_op_parity: input_arrays is empty -- nothing to compare")

    ref_leaves = _leaf_tensors(input_arrays, dtype)
    cand_leaves = _leaf_tensors(input_arrays, dtype)

    ref_out = reference_fn(ref_leaves)
    cand_out = candidate_fn(cand_leaves)

    fwd_report = compare(
        f"{name}.forward", to_numpy_fp32(ref_out), to_numpy_fp32(cand_out), pcc_gate=fwd_pcc_gate
    )

    ref_loss = _scalar_mean_square(ref_out)
    cand_loss = _scalar_mean_square(cand_out)
    ref_loss.backward(False)
    cand_loss.backward(False)

    backward_reports: dict[str, ParityReport] = {}
    for key in input_arrays:
        ref_t, cand_t = ref_leaves[key], cand_leaves[key]
        if not ref_t.is_grad_initialized():
            raise AssertionError(
                f"{name}: reference_fn's input {key!r} has no gradient after backward() -- "
                "reference_fn does not connect this leaf to its output"
            )
        if not cand_t.is_grad_initialized():
            raise AssertionError(
                f"{name}: candidate_fn's input {key!r} has no gradient after backward() -- "
                "candidate_fn does not connect this leaf to its output"
            )
        ref_grad = to_numpy_fp32(ref_t.get_grad_tensor())
        cand_grad = to_numpy_fp32(cand_t.get_grad_tensor())
        backward_reports[key] = compare(f"{name}.backward[{key}]", ref_grad, cand_grad, pcc_gate=bwd_pcc_gate)

    return OpParityResult(forward=fwd_report, backward=backward_reports)
