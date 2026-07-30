"""Synthetic device-profiler probe for isolating per-Q-row-iteration FIXED
overhead in `sdpa_bw_kv`/`sdpa_bw_q`'s compute kernels from real FLOP-bearing
`matmul_tiles` work, at controlled (non-production) shapes.

## Why this exists, and why production shapes can't answer this question

`tenstorrent/docs/m6-optimization-results.md`'s "Real device-level profiling"
section confirmed `SDPABackwardKVDeviceOperation` is 26.9% of one production
step's device time, and that at BOTH real shapes (encoder B=24,H=16,S=544,
D=64 and decoder B=24,H=16,S=2816,D=32) all 5 RISC cores finish within
~0.006-0.11% of each other -- evidence against a DRAM-stall-on-one-RISC
bottleneck, but not proof of "compute-bound" in the strict sense of "the
matmul FLOPs, not fixed per-iteration setup instructions (`matmul_init`,
`reconfig_data_format`, `tile_regs_acquire/commit/release`, CB bookkeeping),
dominate the cost."

At production NC=batch*heads scale, `process_single_row`'s Q-row-iteration
COUNT per core is `(NC*Ht/num_cores) * Ht`, i.e. O(Ht^2) -- so total op
duration scales ~O(Ht^2) *regardless* of whether each iteration's cost is
dominated by fixed overhead or by real per-tile compute (both terms get
multiplied by the same O(Ht^2) iteration count). Comparing encoder vs decoder
duration alone therefore cannot separate "many cheap iterations" from "fewer
expensive iterations" -- a controlled experiment is needed.

## The controlled experiment

Use `batch=1, heads=1` (NC=1) so each of the (up to 110) cores gets AT MOST
ONE K/V row (`num_rows_per_core <= 1`) -- confirmed via
`sdpa_bw_kv_program_factory.cpp`'s `split_work_to_cores(grid, NC*Ht)` splitting
`NC*Ht` total rows round-robin/contiguously across the grid. With NC=1, a
core's total Q-row-iteration count collapses to exactly `Ht` (one row's
`process_single_row` inner loop), not `Ht^2` -- so sweeping `Ht` here gives a
genuinely LINEAR relationship if per-iteration cost is constant:

    op_duration(Ht) = a + b * Ht

where `a` is one-time per-row overhead (the row-level `cb_wait_front(cb_key,
...)`/`cb_wait_front(cb_value, ...)` at the top of `process_single_row`, plus
launch/teardown) and `b` is the per-Q-row-ITERATION cost -- exactly the
quantity item 1 of this session's task is asking about.

Comparing the fitted slope `b` at `head_dim=32` (qWt=vWt=1, decoder's real
head_dim) vs `head_dim=64` (qWt=vWt=2, encoder's real head_dim) isolates
FLOP-bearing cost from fixed overhead: doubling `qWt`/`vWt` roughly doubles
every `matmul_tiles`-loop's real tile-compute work (QK matmul, `update_grad_
value`'s inner block loop, `compute_grad_attn_weights`, `update_grad_key`'s
inner block loop -- 4 sites, `process_single_row` in `sdpa_bw_kv_compute_
kernel.cpp`) while leaving the FIXED-cost operations (`matmul_init`/
`reconfig_data_format` call COUNT, `tile_regs_acquire/commit/release` PAIR
COUNT, CB `cb_wait_front`/`cb_pop_front` call COUNT) exactly unchanged (same
number of calls, just each touching 2 tiles instead of 1). If `b(D=64) ~=
2*b(D=32)`, per-iteration cost is FLOP-dominated (batching Q-rows would only
amortize a small fixed remainder). If `b(D=64) ~= b(D=32)` (barely moves),
per-iteration cost is dominated by the fixed setup/instrumentation calls
(batching Q-rows into wider per-iteration compute passes has real headroom).

## Usage

    python -m smri_mae_tt.scripts.sdpa_bw_overhead_probe --seq-len 2816 --head-dim 32

Meant to be launched under `python -m tracy` (same pattern as
`device_profile_step.py` -- see that file's docstring for the full
`TT_METAL_DEVICE_PROFILER`/signpost/mid-run-flush mechanics, reused verbatim
here since this workload is far smaller than a full training step, no
mid-run flush needed). A driver loop (not committed -- ad hoc, see the
session's report) invokes this once per `(seq_len, head_dim)` combo and
parses each run's `ops_perf_results_*.csv` the same way `device_profile_
report.py` does, filtering to `SDPABackwardKVDeviceOperation`/
`SDPABackwardQDeviceOperation` OP CODE rows.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import ttml
import ttnn
from tracy import signpost

from smri_mae_tt import perf_parity as PP
from smri_mae_tt.ops_tt.attention import bidirectional_scaled_dot_product_attention

SEED = 424242


def run(seq_len: int, head_dim: int, batch: int, heads: int, warmup: int) -> None:
    rng = np.random.default_rng(SEED)
    shape = (batch, heads, seq_len, head_dim)

    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()
    device = auto_ctx.get_device()
    try:
        q_np = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        k_np = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        v_np = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)

        q = ttml.autograd.Tensor.from_numpy(q_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        k = ttml.autograd.Tensor.from_numpy(k_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        v = ttml.autograd.Tensor.from_numpy(v_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        q.set_requires_grad(True)
        k.set_requires_grad(True)
        v.set_requires_grad(True)

        for _ in range(warmup):
            out = bidirectional_scaled_dot_product_attention(q, k, v, seq_len=seq_len)
            ttnn.synchronize_device(device)
            loss = PP._scalar_mean_square(out)
            loss.backward(False)
            ttnn.synchronize_device(device)
            auto_ctx.reset_graph()
            ttnn.synchronize_device(device)

        signpost("start")
        out = bidirectional_scaled_dot_product_attention(q, k, v, seq_len=seq_len)
        ttnn.synchronize_device(device)
        loss = PP._scalar_mean_square(out)
        loss.backward(False)
        ttnn.synchronize_device(device)
        signpost("stop")

        auto_ctx.reset_graph()
        ttnn.synchronize_device(device)
        print(f"sdpa_bw_overhead_probe: done batch={batch} heads={heads} seq_len={seq_len} head_dim={head_dim}", flush=True)
    finally:
        auto_ctx.close_device()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)
    run(args.seq_len, args.head_dim, args.batch, args.heads, args.warmup)


if __name__ == "__main__":
    main(sys.argv[1:])
