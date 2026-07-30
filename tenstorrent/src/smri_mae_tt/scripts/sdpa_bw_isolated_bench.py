"""Isolated device-time microbenchmark for the fused `sdpa_fw`/`sdpa_bw`
kernels (`ttml.ops.attention.scaled_dot_product_attention`), decoupled from
a full training step.

## Why this script exists

`scripts/profile_step_stages.py` established that backward is 65% of the
full step and that isolated encoder/decoder-scale SDPA timing puts decoder
SDPA backward alone at ~49% of *all* backward time in the model (see
`tenstorrent/docs/m6-optimization-results.md`, "Stage-level step profiling"
and the "Redundant Q/dO re-read" follow-up section). Those numbers were
produced by an ad hoc, uncommitted script during that session; this is the
same measurement made reusable and committed, so any future kernel change
in this area (e.g. the KV-kernel redundant-re-read restructuring discussed
in the doc) has a one-command, apples-to-apples before/after.

Q/K/V leaf tensors are built ONCE via `from_numpy` and reused across every
timed iteration -- rebuilding them from numpy on every call inflates
timings 5-10x (an earlier mistake this session's doc explicitly calls out).
Forward and backward are each bounded by an explicit `ttnn.
synchronize_device` so timings reflect real device completion, not host
dispatch racing ahead -- same idiom as `main_pretrain_tt.train()`'s
`on_stage_timing` hook and `profile_step_stages.py`.

Run with:
    python -m smri_mae_tt.scripts.sdpa_bw_isolated_bench
    python -m smri_mae_tt.scripts.sdpa_bw_isolated_bench --shape decoder --iters 10
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import ttml
import ttnn

from smri_mae_tt import perf_parity as PP
from smri_mae_tt.ops_tt.attention import bidirectional_scaled_dot_product_attention

SEED = 2026 + 729

# (batch, heads, seq_len, head_dim) per full_vitl_real_contract() (main_pretrain_tt.py):
# encoder D=1024/heads=16 -> head_dim=64, seq_len=544; decoder D=512/heads=16 -> head_dim=32, seq_len=2816.
SHAPES = {
    "decoder": dict(batch=24, heads=16, seq_len=2816, head_dim=32),
    "encoder": dict(batch=24, heads=16, seq_len=544, head_dim=64),
}


def bench(batch: int, heads: int, seq_len: int, head_dim: int, iters: int, warmup: int, label: str) -> None:
    rng = np.random.default_rng(SEED)
    shape = (batch, heads, seq_len, head_dim)

    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()
    device = auto_ctx.get_device()
    try:
        q_np = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        k_np = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        v_np = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)

        # Leaves built ONCE, reused across every iteration below.
        q = ttml.autograd.Tensor.from_numpy(q_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        k = ttml.autograd.Tensor.from_numpy(k_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        v = ttml.autograd.Tensor.from_numpy(v_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
        q.set_requires_grad(True)
        k.set_requires_grad(True)
        v.set_requires_grad(True)

        fwd_times: list[float] = []
        bwd_times: list[float] = []

        for i in range(warmup + iters):
            t0 = time.monotonic()
            out = bidirectional_scaled_dot_product_attention(q, k, v, seq_len=seq_len)
            ttnn.synchronize_device(device)
            t1 = time.monotonic()

            loss = PP._scalar_mean_square(out)
            loss.backward(False)
            ttnn.synchronize_device(device)
            t2 = time.monotonic()

            auto_ctx.reset_graph()
            ttnn.synchronize_device(device)

            if i >= warmup:
                fwd_times.append((t1 - t0) * 1000.0)
                bwd_times.append((t2 - t1) * 1000.0)

        fwd_avg = statistics.mean(fwd_times)
        bwd_avg = statistics.mean(bwd_times)
        print(
            f"{label}: B={batch} H={heads} S={seq_len} D={head_dim}  "
            f"fwd={fwd_avg:.2f}ms  bwd={bwd_avg:.2f}ms  (bwd/fwd={bwd_avg / fwd_avg:.2f}x)  "
            f"[n={iters}, fwd_stdev={statistics.stdev(fwd_times):.2f}, bwd_stdev={statistics.stdev(bwd_times):.2f}]"
        )
    finally:
        auto_ctx.close_device()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", choices=[*SHAPES, "both"], default="both")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)

    shapes = SHAPES if args.shape == "both" else {args.shape: SHAPES[args.shape]}
    for label, shape_kwargs in shapes.items():
        bench(**shape_kwargs, iters=args.iters, warmup=args.warmup, label=f"{label}-scale (per block)")


if __name__ == "__main__":
    main()
