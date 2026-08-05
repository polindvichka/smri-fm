"""Reusable end-to-end step-time benchmark, via `main_pretrain_tt.train()`'s
`on_step_timing` hook, against real MRI data on real hardware.

## Why this script exists

Every M6 kernel/op/dtype-level change needs an honest "before vs after"
full-training-step number, not just an isolated-op microbenchmark -- a
change that speeds up one op in isolation can still be invisible (or
misleading) at full-step scale if measured incorrectly (e.g. bypassing the
`prefetch=True` data-loading path that `main()` actually uses in production,
which was a real mistake made once while measuring the `AttentionMaskType::
None` SDPA kernel fix -- see `tenstorrent/docs/m6-optimization-results.md`).
Rather than hand-write a new ad hoc timing script per change (which is how
that mistake happened -- an ad hoc script skipped prefetch without anyone
deciding to), this script goes through the *exact* production entry point
(`train()`, same `on_step_timing` hook regardless of what changed under the
hood) so every future measurement is automatically apples-to-apples with
whatever `train()` actually does by default.

Nothing in this file is specific to any one optimization -- it takes a
preset name and step count and reports timing, the same way regardless of
whether the thing being A/B'd is a data-loading change, a kernel swap, a
dtype experiment, or anything else. Compare two runs (e.g. before/after a
change, or `--no-prefetch` vs prefetch) by running this script twice.

## Step-0 JIT warmup

Step 0 of any fresh device/kernel-cache combination pays first-run kernel
JIT compilation cost on top of real per-step work (compare the JIT cache
hit-rate log line tt-metal prints at process exit: near-0% on step 0 of a
cold cache, ~100% by the last step). `--warmup-steps` (default 1) excludes
that many leading steps from the reported average -- raw per-step numbers
for every step, including the excluded ones, are still printed so nothing
is hidden.

## Not a unit test

Filename does not match `test_*.py` (same convention as `measure_occupancy.
py`/`memory_probe.py`) -- run directly with the tt-train Python interpreter.
Opens a real device and trains on real shards (`real_data.RealTrainSource`);
this is not a fast/CI-safe script.
"""

from __future__ import annotations

import argparse
import functools
import json
import statistics
import sys

from smri_mae_tt import real_data
from smri_mae_tt.config_tt import TTMAEConfig
from smri_mae_tt.main_pretrain_tt import _PRESETS, train


def bench(
    preset: str,
    *,
    steps: int,
    warmup_steps: int,
    prefetch: bool,
    seed: int,
    use_tuned_gelu: bool = True,
) -> list[float]:
    """Runs `steps` real training steps at `preset`'s shape/depth, returns
    the per-step wall-clock seconds measured by `train()`'s `on_step_timing`
    hook (index 0 = step 0, in order) -- callers decide how to slice/average
    (see `main()` below for the "drop `warmup_steps` leading entries"
    convention this file uses)."""
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}, choose from {sorted(_PRESETS)}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if warmup_steps < 0 or warmup_steps >= steps:
        raise ValueError(f"warmup_steps must be in [0, steps), got warmup_steps={warmup_steps} steps={steps}")

    contract_fn, encoder_depth, decoder_depth = _PRESETS[preset]
    contract = contract_fn()
    cfg = TTMAEConfig(shape_contract=contract)

    train_source = real_data.RealTrainSource(contract)
    train_batch_source_factory = functools.partial(real_data.RealTrainSource, contract)

    step_times: list[float] = []
    train(
        contract,
        encoder_depth,
        decoder_depth,
        cfg,
        steps,
        warmup_steps=0,
        seed=seed,
        log_every=steps,  # only print the final step's loss line; timing is reported separately below
        print_log=True,
        train_batch_fn=train_source.next_batch,
        train_batch_source_factory=train_batch_source_factory,
        prefetch=prefetch,
        on_step_timing=lambda step, elapsed: step_times.append(elapsed),
        use_tuned_gelu=use_tuned_gelu,
    )
    return step_times


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(_PRESETS), default="full-vitl-real")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=1, help="leading steps excluded from the reported average (JIT warmup, not the LR-schedule warmup_steps)")
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=str, default=None, help="optional path to dump raw per-step seconds as JSON")
    parser.add_argument("--no-tuned-gelu", dest="use_tuned_gelu", action="store_false", default=True, help="M6: disable ops_tt.tuned_gelu (ttnn.gelu variant=FastLut, train()'s default since it's a verified win) to isolate the pre-M6-GELU-fix baseline")
    args = parser.parse_args(argv)

    print(
        f"bench_step: preset={args.preset} steps={args.steps} warmup_steps={args.warmup_steps} prefetch={args.prefetch} "
        f"seed={args.seed} use_tuned_gelu={args.use_tuned_gelu}",
        flush=True,
    )
    step_times = bench(
        args.preset,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        prefetch=args.prefetch,
        seed=args.seed,
        use_tuned_gelu=args.use_tuned_gelu,
    )

    print("\nper-step timing:", flush=True)
    for i, t in enumerate(step_times):
        tag = " (warmup, excluded from avg)" if i < args.warmup_steps else ""
        print(f"  step {i:3d}: {t * 1000:8.1f} ms{tag}", flush=True)

    measured = step_times[args.warmup_steps :]
    avg_ms = statistics.mean(measured) * 1000
    min_ms = min(measured) * 1000
    max_ms = max(measured) * 1000
    print(
        f"\navg (steps {args.warmup_steps}..{len(step_times) - 1}, n={len(measured)}): "
        f"{avg_ms:.1f} ms/step  (min={min_ms:.1f} max={max_ms:.1f})",
        flush=True,
    )

    if args.log_file:
        with open(args.log_file, "w") as f:
            json.dump(
                {
                    "preset": args.preset,
                    "prefetch": args.prefetch,
                    "seed": args.seed,
                    "use_tuned_gelu": args.use_tuned_gelu,
                    "step_seconds": step_times,
                    "warmup_steps": args.warmup_steps,
                    "avg_ms": avg_ms,
                },
                f,
                indent=2,
            )
        print(f"wrote raw timing to {args.log_file}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
