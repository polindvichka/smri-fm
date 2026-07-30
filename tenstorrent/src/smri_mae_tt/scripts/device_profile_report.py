"""Real per-op device-kernel-duration breakdown for one real training step.

## What this is

Drives `device_profile_step.py` (a bracketed single real training step, see
that file's docstring) through `tt-metal`'s standard device-profiler
pipeline (`python -m tracy ... -m <module>`, the same mechanism `models/
tt_dit/utils/sweep_mm_block_sizes.py` uses via `run_device_profiler`, but
invoked directly here since the target is a plain module, not a pytest
node-id) and parses the resulting `ops_perf_results_*.csv` (mirrors `tools/
tracy/process_model_log.py`'s `parse_ops_log`, adapted to report per-OP-CODE
aggregates instead of per-call-count-grouped raw durations).

This answers the question every prior profiling attempt in this project
could not: not "how long did forward/backward take" (host wall-clock,
`profile_step_stages.py`) but "which specific device op, by name, actually
consumed that time" -- genuine `DEVICE KERNEL DURATION [ns]` per op
dispatch, summed/grouped by `OP CODE`.

## Usage

    python -m smri_mae_tt.scripts.device_profile_report
    python -m smri_mae_tt.scripts.device_profile_report --preset full-vitl-real --warmup-steps 1
    python -m smri_mae_tt.scripts.device_profile_report --top 40 --csv-out /tmp/step_ops.csv

Must be run with the tt-train Python interpreter
(`tt-metal/python_env/bin/python3`) from `tenstorrent/src/smri_mae_tt`, same
as every other script in this directory. Takes real device time (one JIT
warmup step + one profiled step at full production scale, plus tracy's own
capture/export overhead) -- budget ~1-3 minutes.

## Why not just call `tracy.process_model_log.run_device_profiler`?

`run_device_profiler` assembles a command string of the form `pytest
<node-id> ...` and passes it through `_build_profiler_cmd`'s `-m
{shlex.quote(command)}` -- inspecting `tools/tracy/__main__.py`, `-m` there
means "treat the next *single* positional argument as a `runpy`-importable
module name" (`runpy.run_module(modname, run_name="__main__")`), not "run
this shell command." A `pytest foo.py::test -x -s`-shaped string, even
shell-quoted into one token, is not a valid Python module name -- this
appears to work in practice for that file only insofar as pytest itself
exposes no CLI surface that collides with being misinterpreted as a broken
import (untested here, out of scope to fix `sweep_mm_block_sizes.py`).
Since our target here is a real importable module (`smri_mae_tt.scripts.
device_profile_step`), this script builds the `python -m tracy ... -m
<module> [module args...]` command directly instead of going through
`run_device_profiler`'s pytest-shaped assumption, verified empirically first
against a trivial `ttnn.add` probe (100% real profiler data, zero rebuild
needed -- see `device_profile_step.py`'s docstring for that verification).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# scripts -> smri_mae_tt -> src -> tenstorrent ; tt-metal lives alongside src/.
TT_METAL_ROOT = SCRIPT_DIR.parents[2] / "tt-metal"
PYTHON_ENV = TT_METAL_ROOT / "python_env" / "bin" / "python3"


def run_profiler(
    *,
    preset: str,
    warmup_steps: int,
    seed: int,
    output_dir: Path,
    op_support_count: int | None,
) -> Path:
    """Runs `device_profile_step` under `python -m tracy`, returns the path
    to the generated `ops_perf_results_*.csv`."""
    output_dir.mkdir(parents=True, exist_ok=True)

    module_args = [
        "smri_mae_tt.scripts.device_profile_step",
        "--preset",
        preset,
        "--warmup-steps",
        str(warmup_steps),
        "--seed",
        str(seed),
    ]

    cmd = [
        str(PYTHON_ENV),
        "-m",
        "tracy",
        "-p",  # partial: only profile explicitly-bracketed (signpost) zones
        "-r",  # generate the ops_perf_results_*.csv report
        "-o",
        str(output_dir),
        "-a",
        "device_kernel_duration",
        "-t",
        "5000",
    ]
    if op_support_count is not None:
        cmd += ["--op-support-count", str(op_support_count)]
    cmd += ["-m", *module_args]

    print(f"running: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    # cwd must be the tt-metal root: `tools/tracy/common.py`'s `TT_METAL_HOME`
    # defaults to `Path.cwd()` when the env var isn't set (it isn't, on this
    # host), and `PROFILER_BIN_DIR` (where `tracy-capture`/`tracy-csvexport`
    # live) is derived from it -- get this wrong and `run_report_setup` can't
    # find the capture tool. `smri_mae_tt.scripts.device_profile_step` is
    # importable regardless of cwd (this python env's site config already
    # puts `tenstorrent/src` on `sys.path` -- confirmed empirically), so
    # there's no competing requirement pulling cwd toward `smri_mae_tt/`.
    # Required for `device_profile_step.py`'s per-stage `ttnn.ReadDeviceProfiler`
    # flush calls to actually take effect mid-run (not just at process exit) --
    # see that file's "Periodic mid-run profiler flush" docstring section.
    env = dict(os.environ)
    env["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
    result = subprocess.run(cmd, cwd=str(TT_METAL_ROOT), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"tracy-wrapped run failed with exit code {result.returncode}")

    reports_dir = output_dir / "reports"
    if not reports_dir.exists():
        raise RuntimeError(f"no reports dir produced at {reports_dir} -- profiler run likely failed silently")
    run_dirs = sorted(p for p in reports_dir.iterdir() if p.is_dir())
    if not run_dirs:
        raise RuntimeError(f"no run subdirectory under {reports_dir}")
    latest = run_dirs[-1]
    csvs = list(latest.glob("ops_perf_results_*.csv"))
    if not csvs:
        raise RuntimeError(f"no ops_perf_results_*.csv under {latest}")
    return csvs[0]


def load_bracketed_ops(csv_path: Path):
    """Parses the ops CSV, returns the DataFrame rows strictly between the
    `start`/`stop` signposts emitted by `device_profile_step.py` -- i.e.
    exactly the one measured training step's device ops, nothing from the
    warmup step(s) or tracy's own setup/teardown."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    signpost_rows = df[df["OP TYPE"] == "signpost"]
    starts = signpost_rows[signpost_rows["OP CODE"] == "start"]
    stops = signpost_rows[signpost_rows["OP CODE"] == "stop"]
    if starts.empty or stops.empty:
        raise RuntimeError(
            f"no start/stop signposts found in {csv_path} -- did device_profile_step.py run to completion?"
        )
    start_idx = starts.index[0]
    stop_idx = stops.index[0]
    bracketed = df.iloc[start_idx + 1 : stop_idx]
    bracketed = bracketed[bracketed["OP TYPE"] != "signpost"]
    bracketed = bracketed[bracketed["DEVICE KERNEL DURATION [ns]"] != "-"]
    bracketed = bracketed.copy()
    bracketed["DEVICE KERNEL DURATION [ns]"] = bracketed["DEVICE KERNEL DURATION [ns]"].astype(float)
    return bracketed


def summarize(bracketed, top: int) -> None:
    import pandas as pd

    total_ns = bracketed["DEVICE KERNEL DURATION [ns]"].sum()
    n_ops = len(bracketed)
    print(f"\nmeasured step: {n_ops} device op dispatches, {total_ns / 1e6:.2f} ms total DEVICE KERNEL DURATION\n")

    by_op = (
        bracketed.groupby("OP CODE")["DEVICE KERNEL DURATION [ns]"]
        .agg(["count", "sum", "mean"])
        .sort_values("sum", ascending=False)
    )
    by_op["pct_of_total"] = 100 * by_op["sum"] / total_ns

    print(f"{'OP CODE':<45} {'count':>7} {'sum_ms':>10} {'mean_us':>10} {'% of total':>10}")
    print("-" * 86)
    for op_code, row in by_op.head(top).iterrows():
        print(
            f"{op_code:<45} {int(row['count']):>7} {row['sum'] / 1e6:>10.3f} "
            f"{row['mean'] / 1e3:>10.2f} {row['pct_of_total']:>9.1f}%"
        )
    if len(by_op) > top:
        rest = by_op.iloc[top:]
        print(
            f"{'... (' + str(len(rest)) + ' more OP CODEs)':<45} {int(rest['count'].sum()):>7} "
            f"{rest['sum'].sum() / 1e6:>10.3f} {'':>10} {100 * rest['sum'].sum() / total_ns:>9.1f}%"
        )

    pd.set_option("display.max_columns", None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", default="full-vitl-real")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=30, help="how many top OP CODEs to print")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="profiler artifacts dir (default: scratch dir under /tmp)",
    )
    parser.add_argument("--op-support-count", type=int, default=4000, help="max distinct program hashes the device profiler tracks (default generous for a 28-block model)")
    parser.add_argument("--csv-out", type=str, default=None, help="optional path to dump the raw bracketed-step CSV")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir) if args.output_dir else Path("/tmp/device_profile_report_out")

    csv_path = run_profiler(
        preset=args.preset,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        output_dir=output_dir,
        op_support_count=args.op_support_count,
    )
    print(f"\nparsing {csv_path}", flush=True)
    bracketed = load_bracketed_ops(csv_path)

    if args.csv_out:
        bracketed.to_csv(args.csv_out, index=False)
        print(f"wrote bracketed-step raw CSV to {args.csv_out}", flush=True)

    summarize(bracketed, top=args.top)


if __name__ == "__main__":
    main(sys.argv[1:])
