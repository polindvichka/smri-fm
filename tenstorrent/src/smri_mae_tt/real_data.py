"""Real-shard data loading for `main_pretrain_tt.py` (its only data path --
see that module's docstring: no synthetic fallback exists).

Wraps `data.mri_data.make_sparse_wds_dataset`/`collate` (read in full,
**unmodified** -- this project's rules forbid touching anything under
`src/`) and `packing.pack_batch` (also unmodified) to turn the local
`datasets/FOMO_with_dwi/shard.NNNNNN.tar` files into the `PackedBatch`
shapes `main_pretrain_tt.train` consumes.

## The 67 local shards vs. the reference recipe's ~1800

`default_pretrain.yaml`'s `datasets.fomo_train`/`fomo_eval` point at
`shard.{000000..001620}.tar` / `shard.{001621..001800}.tar` -- ~1800 shards
total, ~150 samples each. This host has 67 of those shards downloaded
(see `tenstorrent/src/smri_mae_tt/scripts/download_r2_shards.py`, which
downloaded them from R2 throttled to ~5 MB/s, one shard at a time):
`shard.000000/1/2.tar` plus `shard.000004.tar`-`shard.000050.tar` as TRAIN
(50 shards, 7500 samples) and `shard.000003.tar` plus
`shard.000051.tar`-`shard.000066.tar` as VAL (17 shards, 2550 samples;
`shard.000003.tar` is a genuinely separate shard, not a split of a train
shard) -- 10,050 samples total, a ~74.6%/25.4% train/val split. This
module hardcodes exactly those local names (`TRAIN_SHARD_NAMES`/
`VAL_SHARD_NAMES` below) rather than reusing the yaml's full glob range:
the other ~1733 shards do not exist on this host, and pointing
`webdataset.WebDataset` at a glob that mostly resolves to nothing would
either silently train on a near-empty corpus (`expand_urls`'s glob
filtering drops missing files without complaint) or hang -- fail-fast says
name exactly what's here.

## Train: infinite resampled iteration (cycling), by construction

`make_sparse_wds_dataset(url, shuffle=True, ...)` sets `resampled=True` on
the underlying `wds.WebDataset` -- per webdataset's own semantics this
makes shard *and* sample iteration draw with replacement, indefinitely:
`next()` on `iter(dataset)` never raises `StopIteration`. This is not a
workaround coded here; it is `mri_data.make_sparse_wds_dataset`'s own
existing, unmodified behavior when `shuffle=True`, and it is exactly how
the full ~1800-shard reference recipe already runs too
(`default_pretrain.yaml`: `fomo_train.shuffle: true`,
`samples_per_epoch: 243200` on a much larger but still-necessarily-repeated
corpus over `epochs: 100`). At only 7500 local train samples the repetition
rate is far higher, but the *mechanism* is identical -- so **the only thing
bounding a real-data run's length is the existing `--steps` CLI flag**;
there is no separate "epoch" concept to silently loop past.
`RealTrainSource.samples_seen`/`.epoch_equiv` track an *approximate*
progress signal for logging (not exact epochs, since resampling draws with
replacement rather than partitioning).

## Val: one finite, deterministic pass per eval call

`make_sparse_wds_dataset(url, shuffle=False, ...)` sets `resampled=False`,
so iterating the dataset to exhaustion visits every val sample exactly
once, then raises `StopIteration` -- ordinary finite-iterable behavior.
`RealValSource.eval_batches()`'s policy (documented again on the method
itself): one full pass over all 2550 val samples, in fixed
`contract.batch`-sized packed micro-batches, dropping the final partial
batch (matches `default_pretrain.yaml`'s own `drop_last: true` for both its
train and eval datasets). A fresh `wds.WebDataset` iterator is constructed
on every `eval_batches()` call, so every eval pass sees the same 2550
samples in the same order, independent of how many training steps happened
in between.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Callable, Iterator, Sequence

import numpy as np

from data.mri_data import collate, make_sparse_wds_dataset
from smri_mae_tt.layout import ShapeContract
from smri_mae_tt.packing import PackedBatch, pack_batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(_REPO_ROOT, "datasets", "FOMO_with_dwi")

# Explicit, hardcoded shard lists (not a filesystem glob) -- see module
# docstring's "The 67 local shards vs. the reference recipe's ~1800" section
# for why: these are exactly, and only, the shards this host has downloaded
# via download_r2_shards.py, split ~74.6%/25.4% train/val.
TRAIN_SHARD_NAMES = (
    "shard.000000.tar",
    "shard.000001.tar",
    "shard.000002.tar",
    *(f"shard.{i:06d}.tar" for i in range(4, 51)),  # shard.000004 .. shard.000050
)
VAL_SHARD_NAMES = (
    "shard.000003.tar",
    *(f"shard.{i:06d}.tar" for i in range(51, 67)),  # shard.000051 .. shard.000066
)

TRAIN_SHARD_URLS = tuple(os.path.join(DATA_DIR, name) for name in TRAIN_SHARD_NAMES)
VAL_SHARD_URLS = tuple(os.path.join(DATA_DIR, name) for name in VAL_SHARD_NAMES)

SAMPLES_PER_SHARD = 150  # confirmed by tar listing (see task report), not just assumed
TRAIN_SAMPLES_TOTAL = SAMPLES_PER_SHARD * len(TRAIN_SHARD_URLS)
VAL_SAMPLES_TOTAL = SAMPLES_PER_SHARD * len(VAL_SHARD_URLS)


def _require_shards_present(urls: Sequence[str]) -> None:
    missing = [u for u in urls if not os.path.isfile(u)]
    if missing:
        raise FileNotFoundError(
            "real-data shard file(s) not found: " + ", ".join(missing) + ". These are downloaded "
            "separately from this repo (see .gitignore's /datasets/FOMO_with_dwi/ entry) -- wait "
            "for the download to finish before training."
        )


def real_shards_available(urls: Sequence[str] = TRAIN_SHARD_URLS + VAL_SHARD_URLS) -> bool:
    """True iff every shard file this module needs is already present on
    disk. Used by callers (main() and tests) that want to skip/fail-fast
    gracefully rather than raising deep inside a WebDataset iterator.
    """
    return all(os.path.isfile(u) for u in urls)


def _batch_from_samples(samples: list[dict], contract: ShapeContract, rng: np.random.Generator) -> PackedBatch:
    if not samples:
        raise ValueError("_batch_from_samples got an empty sample list -- nothing to pack")
    batch_dict = collate(samples, include_meta=False)
    return pack_batch(batch_dict["image_values"], batch_dict["img_mask"], contract, rng)


class RealTrainSource:
    """Infinite (resampled, per module docstring) source of `PackedBatch`
    micro-batches from the local train shards (`TRAIN_SHARD_URLS` by
    default, 50 shards / 7500 samples; a caller may pass a smaller `urls`
    subset, e.g. in tests that want a fast, small corpus to exercise the
    same resampling behavior with).
    """

    def __init__(
        self,
        contract: ShapeContract,
        *,
        buffer_size: int = 512,
        urls: Sequence[str] = TRAIN_SHARD_URLS,
    ) -> None:
        _require_shards_present(urls)
        self._contract = contract
        self._urls = list(urls)
        self._buffer_size = buffer_size
        # Instance-level, not the module-global TRAIN_SAMPLES_TOTAL: epoch_equiv
        # must reflect the size of *this* source's own corpus (so it stays
        # correct if a caller passes a smaller `urls` subset), and equals
        # TRAIN_SAMPLES_TOTAL exactly when the default (full) urls are used.
        self._samples_total = SAMPLES_PER_SHARD * len(self._urls)
        self._dataset = make_sparse_wds_dataset(self._urls, shuffle=True, buffer_size=buffer_size)
        self._iter = iter(self._dataset)
        self.samples_seen = 0
        self.batches_seen = 0

    def next_batch(self, rng: np.random.Generator) -> PackedBatch:
        samples = [next(self._iter) for _ in range(self._contract.batch)]
        self.samples_seen += len(samples)
        self.batches_seen += 1
        return _batch_from_samples(samples, self._contract, rng)

    @property
    def epoch_equiv(self) -> float:
        """Approximate "how many times has this source's train corpus been
        seen" -- approximate, not exact, because `shuffle=True` resamples
        with replacement rather than partitioning (module docstring)."""
        return self.samples_seen / self._samples_total


class RealValSource:
    """Finite, one-full-pass-per-call source over the local val shard. See
    module docstring for the drop_last / full-pass policy.
    """

    def __init__(
        self,
        contract: ShapeContract,
        *,
        buffer_size: int = 256,
        urls: Sequence[str] = VAL_SHARD_URLS,
    ) -> None:
        _require_shards_present(urls)
        self._contract = contract
        self._urls = list(urls)
        self._buffer_size = buffer_size

    def eval_batches(self, rng: np.random.Generator) -> Iterator[PackedBatch]:
        """Yield fixed-size `PackedBatch` micro-batches for one full,
        deterministic pass over every val sample (`shuffle=False` ->
        `resampled=False`, so `make_sparse_wds_dataset` yields the val
        shard's samples exactly once before `StopIteration`). The trailing
        partial batch (`VAL_SAMPLES_TOTAL % contract.batch != 0`, in
        general) is dropped -- matches `default_pretrain.yaml`'s
        `fomo_eval.drop_last: true`.
        """
        dataset = make_sparse_wds_dataset(self._urls, shuffle=False, buffer_size=self._buffer_size)
        batch_size = self._contract.batch
        samples: list[dict] = []
        for sample in dataset:
            samples.append(sample)
            if len(samples) == batch_size:
                yield _batch_from_samples(samples, self._contract, rng)
                samples = []
        # Trailing partial batch (len(samples) < batch_size) intentionally
        # dropped -- see docstring.


class DeterministicRealSource:
    """A `next_batch(rng)`-shaped real-data source with a fixed, run-to-run
    reproducible sample order -- for tests/fixtures that need real MRI
    content but must see the *same* batches every run (a golden-value
    regression fixture, a loss-trend assertion), where `RealTrainSource`'s
    `shuffle=True` sample order (drawn from an internally-seeded
    `random.Random()` the caller doesn't control) would be flaky.

    Wraps `RealValSource.eval_batches` (`shuffle=False`, exact fixed order)
    behind `RealTrainSource`'s `next_batch(rng)` call shape, so it drops
    into `main_pretrain_tt.train`'s `train_batch_fn` exactly like either
    source does. One full pass over `urls`' samples, then raises
    `StopIteration` -- pick `urls`/`contract.batch`/`num_steps` so the
    caller's step budget stays within that.
    """

    def __init__(self, contract: ShapeContract, *, urls: Sequence[str], seed: int = 0) -> None:
        self._iter = iter(RealValSource(contract, urls=urls).eval_batches(np.random.default_rng(seed)))

    def next_batch(self, rng: np.random.Generator) -> PackedBatch:
        return next(self._iter)


def _process_prefetch_worker(
    source_factory: Callable[[], object],
    seed: int,
    total_batches: int,
    result_queue: "mp.Queue",
) -> None:
    """Entry point run in the spawned child process (module-level, not a
    closure/bound method, so `multiprocessing`'s `spawn` start method --
    a fresh interpreter, no inherited memory -- can pickle and import it by
    qualified name). Builds its own `rng`/data source locally (never receives
    a live `np.random.Generator` or an already-constructed source across the
    process boundary -- neither is safely picklable/shareable) and pushes
    `("ok", batch)`/`("error", message)` tuples back over `result_queue`.
    """
    rng = np.random.default_rng(seed)
    try:
        source = source_factory()
        for _ in range(total_batches):
            batch = source.next_batch(rng)
            result_queue.put(("ok", batch))
    except BaseException as exc:  # noqa: BLE001 -- propagate to the parent process, not swallowed
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


class ProcessPrefetchingBatchSource:
    """Wraps a *factory* for any `next_batch(rng)`-shaped source
    (e.g. `functools.partial(RealTrainSource, contract)`) so the *next*
    step's batch (WebDataset tar decode + `collate` + `packing.pack_batch` --
    measured at ~757ms/step, 27.8% of step time, at `full_vitl_real_contract`
    scale in `tenstorrent/docs/m6-optimization-results.md`) is produced by a
    background **process** while the *current* step's device compute
    (forward/backward/grad_clip/opt_step) is in flight, instead of running
    serially before every step as `main_pretrain_tt.train`'s loop otherwise
    does.

    **Why a process, not a thread (there was a thread-based version of this
    class first)**: measured at `full_vitl_real_contract` scale, a
    `ThreadPoolExecutor`-based prefetcher produced *zero* measurable
    overlap -- `next_batch()`'s own wait time stayed ~750-800ms on every
    step, identical to the non-prefetched baseline, even though the main
    thread's device-compute window (~1.95s) was nearly 3x longer than the
    background thread needed. The background thread was starved of the GIL:
    a single training step makes a very large number of small Python-level
    `ttml`/`ttnn` calls (per-layer forward/backward ops, a `clip_grad_norm`
    loop over ~hundreds of named parameters), and CPython's GIL handoff
    under that kind of rapid-fire-short-critical-section contention doesn't
    reliably let a second thread in even though each individual call
    nominally releases the GIL. A separate OS process has its own GIL
    entirely, sidestepping this. This was confirmed empirically (not
    assumed) before committing to the rewrite -- see this task's measured
    numbers in `tenstorrent/docs/m6-optimization-results.md`.

    **Cost of the process approach**: results cross the process boundary via
    `multiprocessing.Queue`, which pickles the `PackedBatch` (mostly large
    float32/bool numpy arrays -- ~165MB/batch at `full_vitl_real_contract`
    scale). Measured residual `next_batch()` wait time after switching to a
    process: ~105-110ms/step (pickling + pipe transfer), down from
    ~750-800ms fully serial -- a real, large net win, but not free; this is
    why `next_batch()`'s wait time is worth watching if this is reused at an
    even larger batch/patch-dim scale.

    Uses `multiprocessing.get_context("spawn")`, never `fork` -- by the time
    `main_pretrain_tt.train()` constructs this (inside `MeshContext()`), the
    parent process already holds live TT device driver handles; `fork()`
    would duplicate that native state into the child (undefined behavior for
    a hardware device driver). `spawn` starts a fresh interpreter that only
    imports `real_data`/`packing`/`layout`/`data.mri_data` -- none of which
    import `ttml`/`ttnn` -- so the child process never touches the device at
    all, which is also why this is safe to run concurrently with device work
    in the parent.

    **Determinism**: the child constructs its own `np.random.Generator` from
    the integer `seed` passed in (not a shared `Generator` object -- can't
    cross the process boundary) and calls `source_factory().next_batch(rng)`
    strictly `total_batches` times, in order. Same seed, same call count,
    same order as the non-prefetching path would have produced, in the same
    process-local sequence, deterministic sources included -- moving *which*
    process runs a deterministic computation doesn't change its result.

    `total_batches` must be known upfront (the caller's `num_steps`) so the
    child process terminates on its own after producing exactly that many
    batches. Use as a context manager
    (`with ProcessPrefetchingBatchSource(...) as source:`) so a `train()`
    that raises partway through still tears the child process down.
    """

    def __init__(
        self,
        source_factory: Callable[[], object],
        seed: int,
        total_batches: int,
    ) -> None:
        if total_batches < 1:
            raise ValueError(f"total_batches must be >= 1, got {total_batches}")
        ctx = mp.get_context("spawn")
        self._queue: "mp.Queue" = ctx.Queue(maxsize=1)
        self._process = ctx.Process(
            target=_process_prefetch_worker,
            args=(source_factory, seed, total_batches, self._queue),
            daemon=True,
        )
        self._process.start()

    def next_batch(self, rng: object = None) -> PackedBatch:
        """`rng` accepted (and ignored) only so this matches the
        `next_batch(rng)` call shape `main_pretrain_tt.train` expects for
        `train_batch_fn` -- the real rng is owned by the child process
        (class docstring's "Determinism" section)."""
        kind, payload = self._queue.get()
        if kind == "error":
            raise RuntimeError(f"ProcessPrefetchingBatchSource worker failed: {payload}")
        return payload

    def close(self) -> None:
        """Terminate the child process if it's still running (e.g. `train()`
        raised before consuming all `total_batches`) and reap it. Safe to
        call more than once."""
        if self._process.is_alive():
            self._process.terminate()
        self._process.join(timeout=5.0)

    def __enter__(self) -> "ProcessPrefetchingBatchSource":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
