"""Tests for `real_data.ProcessPrefetchingBatchSource` (M6: overlap
`data_pack`'s ~757ms/step host-CPU cost with device compute -- see that
class's docstring and `tenstorrent/docs/m6-optimization-results.md`).

Split into two groups:

1. Pure process-orchestration tests (`test_prefetch_*` below) -- exercise a
   trivial in-module factory (a picklable top-level function returning an
   object with `next_batch(rng)`) that just counts/echoes seeded rng draws.
   This is not MRI content, just a stand-in payload to test spawn/queue/
   error-propagation mechanics fast without needing real shards on disk --
   not the synthetic-data-generation pattern that was removed from
   `main_pretrain_tt.py` (module docstring's "Data:" section).
2. `test_prefetch_matches_sequential_real_batches` -- the real-data
   integration check: confirms wrapping a `DeterministicRealSource` factory
   in `ProcessPrefetchingBatchSource` yields byte-identical batches, in the
   same order, as calling that same source directly in-process with no
   prefetching -- proving the process boundary changes only timing, never
   correctness.

`multiprocessing.get_context("spawn")` process startup is ~0.3-1s, so these
tests are slower than typical unit tests but still fast in absolute terms
(a handful of seconds total).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from smri_mae_tt import real_data
from smri_mae_tt.main_pretrain_tt import smoke_contract
from smri_mae_tt.real_data import ProcessPrefetchingBatchSource

_REQUIRED_URLS = real_data.TRAIN_SHARD_URLS[:1]
pytestmark_real = pytest.mark.skipif(
    not real_data.real_shards_available(_REQUIRED_URLS),
    reason=f"real FOMO_with_dwi shard not downloaded yet (need {_REQUIRED_URLS})",
)


class _CountingSource:
    """A `next_batch(rng)`-shaped source that just echoes seeded rng draws --
    not MRI data, see module docstring. Defined at module level (not a
    closure) so it (and `_counting_source_factory`) are picklable for
    `multiprocessing`'s `spawn` start method."""

    def next_batch(self, rng: np.random.Generator) -> int:
        return int(rng.integers(0, 1_000_000))


def _counting_source_factory() -> _CountingSource:
    return _CountingSource()


def _failing_source_factory():
    raise ValueError("boom")


class _FailingSource:
    def next_batch(self, rng: np.random.Generator):
        raise ValueError("boom during next_batch")


def _failing_next_batch_source_factory() -> "_FailingSource":
    return _FailingSource()


def test_prefetch_preserves_call_order_and_values():
    """The exact sequence of values produced matches what calling the
    wrapped source directly (no prefetch) with the same seed would give --
    the class docstring's "Determinism" guarantee."""
    direct_rng = np.random.default_rng(0)
    direct_source = _counting_source_factory()
    expected = [direct_source.next_batch(direct_rng) for _ in range(10)]

    with ProcessPrefetchingBatchSource(_counting_source_factory, seed=0, total_batches=10) as source:
        actual = [source.next_batch() for _ in range(10)]

    assert actual == expected


def test_prefetch_propagates_worker_construction_exceptions():
    with pytest.raises(RuntimeError, match="ProcessPrefetchingBatchSource worker failed"):
        with ProcessPrefetchingBatchSource(_failing_source_factory, seed=0, total_batches=3) as source:
            source.next_batch()


def test_prefetch_propagates_worker_next_batch_exceptions():
    with pytest.raises(RuntimeError, match="ProcessPrefetchingBatchSource worker failed"):
        with ProcessPrefetchingBatchSource(_failing_next_batch_source_factory, seed=0, total_batches=3) as source:
            source.next_batch()


def test_prefetch_close_reaps_process_with_batches_still_unconsumed():
    """`close()` must terminate/reap the child promptly even when far more
    `total_batches` were requested than were ever consumed."""
    source = ProcessPrefetchingBatchSource(_counting_source_factory, seed=0, total_batches=1000)
    source.next_batch()  # consume exactly one

    source.close()
    assert not source._process.is_alive()

    source.close()  # idempotent


def test_prefetch_rejects_invalid_total_batches():
    with pytest.raises(ValueError, match="total_batches"):
        ProcessPrefetchingBatchSource(_counting_source_factory, seed=0, total_batches=0)


@pytestmark_real
def test_prefetch_matches_sequential_real_batches():
    contract = smoke_contract()
    n = 5

    direct_source = real_data.DeterministicRealSource(contract, urls=_REQUIRED_URLS, seed=7)
    direct_rng = np.random.default_rng(0)
    expected_batches = [direct_source.next_batch(direct_rng) for _ in range(n)]

    factory = functools.partial(real_data.DeterministicRealSource, contract, urls=_REQUIRED_URLS, seed=7)
    with ProcessPrefetchingBatchSource(factory, seed=0, total_batches=n) as source:
        actual_batches = [source.next_batch() for _ in range(n)]

    for expected, actual in zip(expected_batches, actual_batches):
        np.testing.assert_array_equal(expected.visible_values, actual.visible_values)
        np.testing.assert_array_equal(expected.pred_targets, actual.pred_targets)
        np.testing.assert_array_equal(expected.pred_voxel_mask, actual.pred_voxel_mask)
        np.testing.assert_array_equal(expected.visible_ids, actual.visible_ids)
        np.testing.assert_array_equal(expected.pred_ids, actual.pred_ids)
        assert expected.replaced_samples == actual.replaced_samples
