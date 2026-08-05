"""Golden-baseline regression gate for M6 (performance) work.

## What this catches that `perf_parity.py` cannot

`perf_parity.run_op_parity` checks an *isolated* candidate op against its
reference in a synthetic harness -- it can miss a bug that only shows up
once the op is actually wired into the real training loop (a shape/dtype
mismatch at a call boundary, a config value that differs between the
isolated test and the real model, an autograd-graph connectivity gap like
the one `model_tt._SqueezeDecoderPredAxis` had to fix). This file is the
other half: a small, real `main_pretrain_tt.train()` run against the
*current* (slow but already-verified-correct, M2/M3/M4-gated) implementation,
with per-step loss and grad-norm hardcoded below and checked with a tight,
explicit tolerance. Once M6 swaps in a tuned op, this test must still pass
unchanged -- if it doesn't, something end-to-end broke even if that op's own
`perf_parity` numbers looked fine in isolation.

## Where `GOLDEN_STEP_LOSSES`/`GOLDEN_STEP_GRAD_NORMS` came from

Captured by running exactly the `test_golden_baseline_smoke_run_matches_recorded_values`
call below (`main_pretrain_tt.train`, `main_pretrain_tt.smoke_contract()`,
`encoder_depth=2, decoder_depth=1`, `TTMAEConfig(base_lr=0.15)`, `num_steps=5,
warmup_steps=1, accum_iter=1, seed=2026`) on this host's real Blackhole
hardware, on the current implementation (`ttml`'s fused SDPA, `ttml.ops.linear
.linear` -- i.e. nothing this M6 pass changed). Wall clock: ~0.8-1.0s for the
5 steps (well under this project's "small passes not full cycles" budget).

**Recaptured** when `train()`'s `use_tuned_gelu` default flipped `False` ->
`True` (`ops_tt.tuned_gelu`, `ttnn.gelu`'s `FastLut` variant -- PCC-verified
0.99998 against a PyTorch reference, `tenstorrent/docs/m6-optimization-
results.md`'s "Non-SDPA/matmul op survey" section) -- per this file's own
"How to regenerate" policy below (recapture, don't loosen tolerance, after a
legitimate, PCC-gated implementation change). Old vs new values differ only
in the 4th-5th significant bf16 digit of `LOSS` (e.g. step 0:
`0.9970703125` -> `0.996826171875`, a `~4e-4` shift -- an approximation
changing which bf16 rounding bucket a value lands in, not a broken
computation) and are **bit-identical** in `GRAD_NORM` at this precision;
confirmed bit-exact reproducible across two separate recapture runs before
being recorded here, same standard as the original capture.

**Recaptured again (2026-07-30)** when `train()`'s `use_tuned_linear` default
flipped `False` -> `True` (`ops_tt.tuned_linear.TunedLinearLayer`, real
~2% end-to-end step-time reduction at production scale, `TENSTORRENT_PORT.md`
section 5 item 1 -- this smoke fixture's small `smoke_contract()` shapes have
no sweep-cache entry, so every linear call here is the documented cache-miss
`program_config=None` path, still numerically distinct from `ttml.ops.linear.
linear` per `use_tuned_linear`'s own docstring). Same pattern as the
`use_tuned_gelu` recapture above: `LOSS` shifts by ~1-5e-4 (e.g. step 0:
`0.996826171875` -> `0.9970703125`), `GRAD_NORM` shifts by 0 to ~2e-3 (e.g.
step 3: `0.2119140625` -> `0.2099609375`) -- bf16 rounding-bucket noise from
routing through a different (but numerically equivalent by construction)
code path, not a broken computation. Confirmed bit-exact reproducible across
two separate recapture runs before being recorded here.

**Recaptured again (2026-07-31)** for the vendored `sdpa_bw_compute_utils.hpp`
change (`compute_u_scalar_row` now uses `sfpu_reduce<SUM, Float32, REDUCE_ROW>`
directly on DST instead of a matmul-with-ones-vector CB roundtrip -- upstream
Tenstorrent optimization B17, `tt-train/sources/ttml/metal/ops/
SDPA_OPTIMIZATION_PROPOSALS.md`; see `TENSTORRENT_PORT.md` section 5). `LOSS`
shifts by 0 to ~5e-4 (step 2: `0.880859375` -> `0.88037109375`), `GRAD_NORM`
shifts by 0 to ~2e-3 (step 2: `0.1533203125` -> `0.154296875`, step 3:
`0.2099609375` -> `0.2119140625`) -- same bf16 rounding-bucket noise pattern
as both prior recaptures, not a broken computation (full gtest suite 10/10
including production-scale PCC-gated coverage, and the rest of the full
pytest suite, both passed unaffected). Confirmed bit-exact reproducible
across two separate recapture runs before being recorded here.

**Recaptured again (2026-08-05)** after a stability pass reverted three M6
optimizations judged not worth their added complexity/risk for production
use: `use_tuned_linear` (removed entirely -- `ops_tt/tuned_linear.py`, its
sweep script, and its per-shape program_config cache all deleted) and the
two vendored SDPA backward-KV kernel micro-optimizations (Q/dO group-cache
reader, 7->6-cycle fused compute kernel -- both reverted to the original
kernel `ttml`/tt-train shipped, still selected for every mask type). The
B17 `sfpu_reduce` fusion (previous entry) was kept -- see
`TENSTORRENT_PORT.md` section 5 for the full rationale. Net result at this
fixture's tiny `smoke_contract()` scale: `LOSS` reverts exactly to its
pre-`use_tuned_linear` values (step 0: `0.9970703125` -> `0.996826171875`,
step 2: `0.88037109375` -> `0.881103515625` -- both match the recorded
`use_tuned_gelu`-only recapture above bit-for-bit), `GRAD_NORM` is
**unchanged at every step** -- neither the SDPA kernel reverts nor keeping
B17 perturb this shape's bf16 rounding at all; `use_tuned_linear` was the
only one of the three ever-shipped M6 changes that did. Confirmed bit-exact
reproducible across two separate recapture runs before being recorded here.

**Real shards, deterministic order**: `main_pretrain_tt.py` has no synthetic
data path (its module docstring), so this fixture's `train_batch_fn` reads
real MRI volumes -- `real_data.RealValSource(contract,
urls=real_data.TRAIN_SHARD_URLS[:1]).eval_batches(rng)`, a single train
shard (150 samples) iterated with `shuffle=False`. `RealValSource` (not
`RealTrainSource`) is deliberate here even though this is a *train* fixture:
`shuffle=False` makes `wds.WebDataset` iterate shard/sample order exactly
(`resampled=False`, per `real_data.py`'s module docstring), whereas
`RealTrainSource`'s `shuffle=True` path draws shard/sample order from an
internal `random.Random()` not seeded by anything this file controls --
fine for training, not reproducible enough for a golden fixture pinned to
bit-exact values. Confirmed bit-exact identical across repeated runs before
recording the numbers below (same as the "Reproducibility and tolerance"
section further down already required of the old synthetic-data version).

**Scale**: `smoke_contract()` (`embed_dim=64`, `decoder_embed_dim=32`,
`encoder_heads=2`, `decoder_heads=1`, `l_vis=31`, `q_pred=32`,
`img_size=(208,240,208)`, matching the real shards) -- the same tiny config
`test_main_pretrain_tt.py`'s own M5-gate smoke test already runs 20 steps of,
chosen here for the same
reason: fast, already load-bearing elsewhere in this suite, and (unlike a
per-*parameter* gradient PCC check) this file only ever compares *scalar*
aggregates (one loss, one grad-norm per step) -- the tiny-`embed_dim`
bf16-noise-doesn't-average-out pitfall documented in `test_model_tt.py`'s
`_grad_parity_contract` docstring is specifically a per-parameter problem
(individual small LayerNorm gradients), not a whole-model-aggregate one; that
same docstring notes the *global* grad-norm still agreed to within ~0.1-0.6%
even at this tiny width. Not "close to full ViT-L scale" (task guidance's
default preference) because the user explicitly asked for smoke-test-scale,
seconds-not-minutes runs for anything in the committed suite; a full-ViT-L
-scale one-off capture is out of scope for a fixture that must stay fast to
re-run on every commit.

## Reproducibility and tolerance

Repeated runs of the *identical* call (same process and separate fresh
processes, checked while building this file) produced **bit-exact identical**
per-step loss and grad-norm values -- this training loop has no dropout, no
sampling inside the device graph, and (empirically) no run-to-run kernel
nondeterminism at this shape on this hardware/software stack. `LOSS_ATOL`/
`GRAD_NORM_ATOL` below (`1e-4`) are therefore *not* "how much bf16 slop to
tolerate" (there is none, observed) -- they are a deliberately tight but
non-zero margin, far tighter than the M4 whole-model gate's 2% grad-norm
tolerance, so a future run isn't spuriously flagged over an irrelevant
float-formatting difference while still failing loudly on any real
regression (a tuned op that changes even one bf16 rounding step's outcome
would move these values by much more than 1e-4 at this precision).

## How to regenerate

Re-run this file's own steps with `print_log=True` (or copy the call and
`print(entry.loss, entry.grad_norm)` per step) after intentionally changing
the reference implementation (e.g. to accommodate a legitimate architecture
- neutral refactor); do not "fix" a golden-baseline failure by loosening
`LOSS_ATOL`/`GRAD_NORM_ATOL` without first confirming the new values are
actually still correct (e.g. `perf_parity`/M2-M4 gates still pass) -- a
failure here after an M6 op swap most likely means the swap broke something,
not that this fixture was wrong.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("ttml")

from smri_mae_tt import real_data  # noqa: E402
from smri_mae_tt.config_tt import TTMAEConfig  # noqa: E402
from smri_mae_tt.main_pretrain_tt import smoke_contract, train  # noqa: E402

SEED = 2026
NUM_STEPS = 5

pytestmark = pytest.mark.skipif(
    not real_data.real_shards_available(real_data.TRAIN_SHARD_URLS[:1]),
    reason=f"real FOMO_with_dwi shard not downloaded yet (need {real_data.TRAIN_SHARD_URLS[0]})",
)


def _golden_train_batch_fn(contract):
    """A single-shard, `shuffle=False` real-data source -- deterministic
    shard/sample order (module docstring, "Real shards, deterministic
    order"), required for a golden fixture pinned to bit-exact values."""
    return real_data.DeterministicRealSource(contract, urls=real_data.TRAIN_SHARD_URLS[:1], seed=SEED + 1).next_batch


# Captured exactly as described in the module docstring's "Where ... came
# from" section -- current (M2/M3/M4-gated) implementation, real Blackhole
# hardware, bit-exact reproducible across repeated runs.
GOLDEN_STEP_LOSSES = [
    0.996826171875,
    1.08837890625,
    0.881103515625,
    1.69677734375,
    1.011474609375,
]
GOLDEN_STEP_GRAD_NORMS = [
    0.1748046875,
    0.18359375,
    0.154296875,
    0.2119140625,
    0.150390625,
]

LOSS_ATOL = 1e-4
GRAD_NORM_ATOL = 1e-4


def test_golden_baseline_fixture_shape_is_consistent():
    """Sanity check on the fixture itself, independent of the device: both
    golden lists are `NUM_STEPS` long and every value is finite -- catches a
    transcription typo in the hardcoded numbers above without needing
    hardware."""
    assert len(GOLDEN_STEP_LOSSES) == NUM_STEPS
    assert len(GOLDEN_STEP_GRAD_NORMS) == NUM_STEPS
    assert all(math.isfinite(v) for v in GOLDEN_STEP_LOSSES)
    assert all(math.isfinite(v) for v in GOLDEN_STEP_GRAD_NORMS)


def test_golden_baseline_smoke_run_matches_recorded_values():
    """The regression gate itself: re-run the exact same small training pass
    against whatever implementation is currently checked in, and require its
    per-step loss/grad-norm to match the recorded golden values within
    `LOSS_ATOL`/`GRAD_NORM_ATOL` (module docstring, "Reproducibility and
    tolerance"). A future M6 op swap must leave this passing unchanged.
    """
    contract = smoke_contract()
    cfg = TTMAEConfig(shape_contract=contract, base_lr=0.15)

    history = train(
        contract,
        encoder_depth=2,
        decoder_depth=1,
        cfg=cfg,
        num_steps=NUM_STEPS,
        warmup_steps=1,
        accum_iter=1,
        seed=SEED,
        log_every=1,
        print_log=False,
        train_batch_fn=_golden_train_batch_fn(contract),
    )

    assert len(history) == NUM_STEPS
    losses = [entry.loss for entry in history]
    grad_norms = [entry.grad_norm for entry in history]

    assert all(v is not None for v in grad_norms), f"grad_norm missing on some step: {grad_norms}"
    assert all(math.isfinite(v) for v in losses), f"non-finite loss in {losses}"
    assert all(math.isfinite(v) for v in grad_norms), f"non-finite grad_norm in {grad_norms}"

    loss_diffs = [abs(v - g) for v, g in zip(losses, GOLDEN_STEP_LOSSES)]
    grad_norm_diffs = [abs(v - g) for v, g in zip(grad_norms, GOLDEN_STEP_GRAD_NORMS)]

    assert max(loss_diffs) <= LOSS_ATOL, (
        f"loss diverged from golden baseline beyond tolerance {LOSS_ATOL}: "
        f"got={losses}, golden={GOLDEN_STEP_LOSSES}, diffs={loss_diffs}"
    )
    assert max(grad_norm_diffs) <= GRAD_NORM_ATOL, (
        f"grad_norm diverged from golden baseline beyond tolerance {GRAD_NORM_ATOL}: "
        f"got={grad_norms}, golden={GOLDEN_STEP_GRAD_NORMS}, diffs={grad_norm_diffs}"
    )
