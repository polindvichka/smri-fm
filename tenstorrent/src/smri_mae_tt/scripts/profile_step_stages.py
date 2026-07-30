"""Reusable per-stage step-time breakdown, via `main_pretrain_tt.train()`'s
`on_stage_timing` hook, against real MRI data on real hardware.

## Why this script exists

`scripts/bench_step.py` answers "is the whole step faster or slower" but not
"which part of the step is actually slow." Repeated matmul program-config
tuning attempts this session (2D `MatmulMultiCoreReuseMultiCastProgramConfig`,
padded to use the full device grid, block-size swept) gave no measurable
end-to-end gain and, for the decoder's tall-skinny shapes, could not even fit
L1 -- so before trying another lever blind, this script gets a real,
device-synchronized breakdown of where step time actually goes: data
acquisition, forward, backward, grad-clip, optimizer step. Same pattern as
`bench_step.py` -- takes a preset name and step count, not hardcoded to any
one shape or change being investigated.

## Sync overhead is real and intentional

Each stage boundary calls `ttnn.synchronize_device` so the reported time
reflects actual device completion, not host dispatch racing ahead (`data`
and `backward` need an explicit sync since nothing else forces one there;
`forward`/`grad_clip` already end in a scalar readback that forces sync as a
side effect, but an explicit sync keeps every stage measured the same way).
This makes per-stage timing slower than production `train()` -- use
`bench_step.py` for the real step-time number, this script only for relative
stage breakdown.
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import statistics
import sys

from smri_mae_tt import real_data
from smri_mae_tt.config_tt import TTMAEConfig
from smri_mae_tt.main_pretrain_tt import _PRESETS, train

STAGES = ("data", "forward", "backward", "reset_graph", "grad_clip", "optimizer_step")


def profile(
    preset: str,
    *,
    steps: int,
    warmup_steps: int,
    prefetch: bool,
    seed: int,
) -> dict[str, list[float]]:
    """Runs `steps` real training steps at `preset`'s shape/depth, returns
    `{stage_name: [seconds per occurrence, in step order]}` for every stage
    in `STAGES`. `grad_clip`/`optimizer_step` only occur on accumulation-
    boundary steps (same gating as the work itself), so those lists can be
    shorter than `steps`."""
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

    stage_times: dict[str, list[float]] = collections.defaultdict(list)

    def on_stage_timing(step: int, stage: str, elapsed: float) -> None:
        if step >= warmup_steps:
            stage_times[stage].append(elapsed)

    train(
        contract,
        encoder_depth,
        decoder_depth,
        cfg,
        steps,
        warmup_steps=0,
        seed=seed,
        log_every=steps,
        print_log=True,
        train_batch_fn=train_source.next_batch,
        train_batch_source_factory=train_batch_source_factory,
        prefetch=prefetch,
        on_stage_timing=on_stage_timing,
    )
    return dict(stage_times)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(_PRESETS), default="full-vitl-real")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=1, help="leading steps excluded from the reported average (JIT warmup)")
    parser.add_argument("--no-prefetch", dest="prefetch", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-file", type=str, default=None, help="optional path to dump raw per-stage seconds as JSON")
    args = parser.parse_args(argv)

    print(f"profile_step_stages: preset={args.preset} steps={args.steps} warmup_steps={args.warmup_steps} prefetch={args.prefetch} seed={args.seed}", flush=True)
    stage_times = profile(args.preset, steps=args.steps, warmup_steps=args.warmup_steps, prefetch=args.prefetch, seed=args.seed)

    print("\nper-stage breakdown (device-synchronized, warmup steps excluded):", flush=True)
    totals = {}
    for stage in STAGES:
        times = stage_times.get(stage, [])
        if not times:
            print(f"  {stage:>15}: (no occurrences)", flush=True)
            continue
        avg_ms = statistics.mean(times) * 1000
        totals[stage] = avg_ms
        print(f"  {stage:>15}: avg={avg_ms:8.2f} ms  min={min(times)*1000:8.2f} ms  max={max(times)*1000:8.2f} ms  n={len(times)}", flush=True)

    grand_total = sum(totals.values())
    if grand_total > 0:
        print(f"\nsum of stages: {grand_total:.2f} ms (this is slower than a real step -- see module docstring on sync overhead)", flush=True)
        print("share of stage-sum time:", flush=True)
        for stage, avg_ms in sorted(totals.items(), key=lambda kv: -kv[1]):
            print(f"  {stage:>15}: {100 * avg_ms / grand_total:5.1f}%", flush=True)

    if args.log_file:
        with open(args.log_file, "w") as f:
            json.dump({"preset": args.preset, "prefetch": args.prefetch, "seed": args.seed, "stage_seconds": stage_times, "warmup_steps": args.warmup_steps}, f, indent=2)
        print(f"\nwrote raw timing to {args.log_file}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
