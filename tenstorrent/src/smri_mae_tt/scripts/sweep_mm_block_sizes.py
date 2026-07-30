"""Offline matmul block-size sweep -- TENSTORRENT_PORT.md section 5, item 1.

## Why this script exists

`ops_tt/tuned_linear.py`'s `build_program_config` always builds the single
*maximal-grid* `MatmulMultiCoreReuseMultiCastProgramConfig` (largest grid
divisor <= device cap on each axis). Measured (`config_tt.
MatmulProgramConfigRegistry`'s docstring, dead end #2 in this doc's section
4): that one hand-picked config is a wash (0.95-1.05x, noise) and fails
outright (L1 circular-buffer overflow) at the MLP shape -- it was never a
real search, just one guess. Only 64 of the device's 110 cores end up doing
work at these shapes under that guess.

This script is the real search the doc's plan calls for: for each real
linear-op shape in the model (`SHAPES` below), it times every candidate from
`ops_tt.tuned_linear.iter_candidate_program_configs` on real hardware,
verifies each surviving candidate's output against the `program_config=None`
default via PCC, and keeps the fastest PCC-passing one. Winners are cached
to `ops_tt/mm_program_config_cache.json`
(`ops_tt.tuned_linear.DEFAULT_CACHE_PATH`), which `TunedLinearLayer` reads
automatically -- running this script and committing its output is the whole
integration step, no code changes needed elsewhere.

## Why some candidates are expected to fail

`iter_candidate_program_configs` only enforces the exact-grid-coverage
constraints that are true `TT_FATAL`s regardless of L1 budget (per-core
tile counts must multiply back to the op's real M/N/K tile counts). It
cannot see the L1 circular-buffer budget a given (grid, block, subblock)
combination will need -- confirmed by hand (see that function's docstring)
that exceeding it raises a catchable `RuntimeError`, not a crash. This
script's `_time_candidate` catches exactly that and skips the candidate,
same idiom as `main_pretrain_tt.train()` and every other script in this
project: real failures are logged, not swallowed silently or retried.

## Real-hardware cost

Each *distinct* program config triggers a fresh kernel JIT compile on its
first use (same cold-cache cost `bench_step.py`'s docstring describes for
`train()` itself) -- `--iters`/`--warmup` per candidate are deliberately
small (default 3/1) since the goal is a relative ranking across many
candidates, not a precise absolute number for one. Run with `--shapes
encoder` first (encoder depth=24 dominates total matmul FLOPs over
decoder's depth=4, per this doc's section 4 op-time table), then `--shapes
decoder`, rather than `--shapes all` in one sitting -- each full-shape sweep
can take several minutes.

Run with:
    python -m smri_mae_tt.scripts.sweep_mm_block_sizes --shapes encoder
    python -m smri_mae_tt.scripts.sweep_mm_block_sizes --shapes decoder
    python -m smri_mae_tt.scripts.sweep_mm_block_sizes --shapes all --max-candidates 20

## Not a unit test

Same convention as `bench_step.py`/`measure_occupancy.py`: filename does not
match `test_*.py`, opens a real device, not CI-safe. `ops_tt/
test_tuned_linear.py` unit-tests `iter_candidate_program_configs`/cache
load/lookup host-side (no device) -- this script is the device-timing loop
built on top of that already-tested logic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import numpy as np
import ttml
import ttnn

from smri_mae_tt import parity as PP
from smri_mae_tt.layout import TILE
from smri_mae_tt.ops_tt import tuned_linear as TL

SEED = 2026 + 730

# (batch, seq_len, in_features, out_features) per full_vitl_real_contract()
# (main_pretrain_tt.py): encoder D=1024/heads=16/seq_len=544/mlp_ratio=4,
# decoder D=512/heads=16/seq_len=2816/mlp_ratio=4. batch=24 matches the
# production shape these ops actually run at.
SHAPES: dict[str, dict[str, dict[str, int]]] = {
    "encoder": {
        "encoder_qkv": dict(batch=24, seq_len=544, in_features=1024, out_features=3072),
        "encoder_proj": dict(batch=24, seq_len=544, in_features=1024, out_features=1024),
        "encoder_fc1": dict(batch=24, seq_len=544, in_features=1024, out_features=4096),
        "encoder_fc2": dict(batch=24, seq_len=544, in_features=4096, out_features=1024),
    },
    "decoder": {
        "decoder_qkv": dict(batch=24, seq_len=2816, in_features=512, out_features=1536),
        "decoder_proj": dict(batch=24, seq_len=2816, in_features=512, out_features=512),
        "decoder_fc1": dict(batch=24, seq_len=2816, in_features=512, out_features=2048),
        "decoder_fc2": dict(batch=24, seq_len=2816, in_features=2048, out_features=512),
    },
}


def _build_inputs(rng: np.random.Generator, batch: int, seq_len: int, in_f: int, out_f: int):
    x_np = (rng.standard_normal((batch, 1, seq_len, in_f)) * 0.1).astype(np.float32)
    w_np = (rng.standard_normal((1, 1, out_f, in_f)) * 0.02).astype(np.float32)
    b_np = (rng.standard_normal((1, 1, 1, out_f)) * 0.01).astype(np.float32)
    x = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16).get_value()
    w = ttml.autograd.Tensor.from_numpy(w_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16).get_value()
    b = ttml.autograd.Tensor.from_numpy(b_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16).get_value()
    return x, w, b


def _to_numpy(raw_value: "ttnn.Tensor") -> np.ndarray:
    """Raw `ttnn.Tensor` -> fp32 numpy, via the same `ttml.autograd.
    create_tensor` wrap-then-`to_numpy` workaround `params.py::
    _read_tensor_value` uses for BFLOAT8_B (`to_numpy` is only exposed on
    `ttml.autograd.Tensor`, not the raw `ttnn.Tensor` this script's `ttnn.
    linear` calls return)."""
    wrapped = ttml.autograd.create_tensor(raw_value, requires_grad=False)
    return np.asarray(wrapped.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32)


def _time_forward(x, w, b, program_config, device, *, iters: int, warmup: int) -> list[float] | None:
    """Returns per-iter ms (post-warmup) for `ttnn.linear(..., program_config)`,
    or `None` if the candidate raised (caught and reported by the caller)."""
    times = []
    out = None
    for i in range(warmup + iters):
        t0 = time.monotonic()
        out = ttnn.linear(x, w, bias=b, transpose_a=False, transpose_b=True, program_config=program_config)
        ttnn.synchronize_device(device)
        t1 = time.monotonic()
        if i >= warmup:
            times.append((t1 - t0) * 1000.0)
    return times, out


def sweep_shape(
    name: str,
    *,
    batch: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    device,
    iters: int,
    warmup: int,
    max_candidates: int,
    pcc_gate: float,
) -> dict | None:
    print(f"\n=== {name}: batch={batch} seq_len={seq_len} in={in_features} out={out_features} ===", flush=True)
    rng = np.random.default_rng(SEED + hash(name) % 10_000)
    x, w, b = _build_inputs(rng, batch, seq_len, in_features, out_features)

    baseline_times, baseline_out = _time_forward(x, w, b, None, device, iters=iters, warmup=warmup)
    baseline_ms = statistics.mean(baseline_times)
    baseline_np = _to_numpy(baseline_out)
    print(f"  baseline (program_config=None): {baseline_ms:.3f} ms", flush=True)

    grid = device.compute_with_storage_grid_size()
    candidates = TL.iter_candidate_program_configs(
        batch=batch,
        seq_len=seq_len,
        in_features=in_features,
        out_features=out_features,
        grid_x=grid.x,
        grid_y=grid.y,
        max_candidates=max_candidates,
    )
    print(f"  {len(candidates)} candidates to try", flush=True)

    best = None  # (ms, pcc, program_config)
    n_failed = 0
    for i, pc in enumerate(candidates):
        pc_grid = pc.compute_with_storage_grid_size
        label = (
            f"grid=({pc_grid.x},{pc_grid.y}) in0_block_w={pc.in0_block_w} "
            f"subblock=({pc.out_subblock_h},{pc.out_subblock_w}) per_core=({pc.per_core_M},{pc.per_core_N})"
        )
        try:
            times, out = _time_forward(x, w, b, pc, device, iters=iters, warmup=warmup)
        except Exception as e:  # real TT_FATAL/L1-overflow candidates are expected -- see module docstring
            n_failed += 1
            print(f"  [{i + 1}/{len(candidates)}] FAILED  {label}  ({type(e).__name__})", flush=True)
            continue

        ms = statistics.mean(times)
        cand_np = _to_numpy(out)
        report = PP.compare(f"{name}[{i}]", baseline_np, cand_np, pcc_gate=0.0)
        status = "ok" if report.pcc >= pcc_gate else "LOW-PCC (rejected)"
        speedup = baseline_ms / ms
        print(f"  [{i + 1}/{len(candidates)}] {ms:7.3f} ms  ({speedup:.2f}x)  pcc={report.pcc:.6f}  {status}  {label}", flush=True)

        if report.pcc < pcc_gate:
            continue
        if best is None or ms < best[0]:
            best = (ms, report.pcc, pc)

    print(f"  {len(candidates) - n_failed}/{len(candidates)} candidates ran; {n_failed} failed (expected, see docstring)", flush=True)

    if best is None:
        print(f"  no PCC-passing candidate beat the fail/skip bar for {name}; leaving this shape uncached", flush=True)
        return None

    best_ms, best_pcc, best_pc = best
    best_grid = best_pc.compute_with_storage_grid_size
    speedup = baseline_ms / best_ms
    print(f"  BEST: {best_ms:.3f} ms ({speedup:.2f}x vs baseline), pcc={best_pcc:.6f}", flush=True)

    return {
        "compute_with_storage_grid_size": [best_grid.x, best_grid.y],
        "in0_block_w": best_pc.in0_block_w,
        "out_subblock_h": best_pc.out_subblock_h,
        "out_subblock_w": best_pc.out_subblock_w,
        "per_core_M": best_pc.per_core_M,
        "per_core_N": best_pc.per_core_N,
        "ms": best_ms,
        "baseline_ms": baseline_ms,
        "speedup": speedup,
        "pcc": best_pcc,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", choices=[*SHAPES, "all"], default="encoder")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=25)
    parser.add_argument("--pcc-gate", type=float, default=0.999, help="matches perf_parity.OP_FWD_PCC_GATE")
    parser.add_argument("--cache-path", type=str, default=str(TL.DEFAULT_CACHE_PATH))
    args = parser.parse_args(argv)

    if args.shapes == "all":
        shape_groups = list(SHAPES.values())
    else:
        shape_groups = [SHAPES[args.shapes]]
    shapes_to_run: dict[str, dict[str, int]] = {}
    for group in shape_groups:
        shapes_to_run.update(group)

    for shape in shapes_to_run.values():
        if shape["seq_len"] % TILE != 0 or shape["in_features"] % TILE != 0 or shape["out_features"] % TILE != 0:
            raise ValueError(f"sweep_mm_block_sizes: non-tile-aligned shape {shape}")

    cache = TL.load_program_config_cache(args.cache_path)
    print(f"sweep_mm_block_sizes: {len(shapes_to_run)} shapes, existing cache has {len(cache)} entries", flush=True)

    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()
    try:
        device = auto_ctx.get_device()
        for name, shape in shapes_to_run.items():
            result = sweep_shape(
                name,
                **shape,
                device=device,
                iters=args.iters,
                warmup=args.warmup,
                max_candidates=args.max_candidates,
                pcc_gate=args.pcc_gate,
            )
            auto_ctx.reset_graph()
            if result is not None:
                key = TL.cache_key(shape["batch"], shape["seq_len"], shape["in_features"], shape["out_features"])
                cache[key] = result
                with open(args.cache_path, "w") as f:
                    json.dump(cache, f, indent=2, sort_keys=True)
                print(f"  wrote {key} to {args.cache_path}", flush=True)
    finally:
        auto_ctx.close_device()

    print(f"\ndone. cache now has {len(cache)} entries at {args.cache_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
