"""Tests for `real_data.py` and its wiring into `main_pretrain_tt.py`'s
`train()` (the only data path -- see that module's docstring), against the
actual downloaded `datasets/FOMO_with_dwi/shard.*.tar` files.

Download timing is out of this task's control (see project instructions:
"real data is being downloaded ... may still be in progress"), so every
test here starts with `real_data.real_shards_available(...)` and calls
`pytest.skip(...)` with a clear message if the shard(s) it needs aren't on
disk yet, rather than failing confusingly (or hanging) partway through a
`webdataset` iterator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402

from smri_mae_tt import real_data  # noqa: E402
from smri_mae_tt.config_tt import TTMAEConfig  # noqa: E402
from smri_mae_tt.layout import ShapeContract  # noqa: E402
from smri_mae_tt.main_pretrain_tt import smoke_contract, train  # noqa: E402

# Every test below uses either the full TRAIN_SHARD_URLS default or an
# explicit VAL_SHARD_URLS[:1] subset -- not the full 67-file VAL_SHARD_URLS
# set (`real_shards_available()`'s own default), which this host doesn't
# have complete (see real_data.py's module docstring on which val shards are
# actually present). Gating on the stricter, unused-here default would skip
# this whole file even though every test's actual data need is satisfied.
_REQUIRED_URLS = real_data.TRAIN_SHARD_URLS + real_data.VAL_SHARD_URLS[:1]
pytestmark = pytest.mark.skipif(
    not real_data.real_shards_available(_REQUIRED_URLS),
    reason=f"real FOMO_with_dwi shard(s) not downloaded yet (need {_REQUIRED_URLS})",
)


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_real_train_source_yields_expected_shapes():
    contract = smoke_contract()
    source = real_data.RealTrainSource(contract)

    rng = np.random.default_rng(0)
    batch = source.next_batch(rng)

    assert batch.visible_values.shape == (contract.batch, contract.l_vis, contract.patch_dim)
    assert batch.pred_targets.shape == (contract.batch, contract.q_pred, contract.patch_dim)
    assert batch.pred_voxel_mask.shape == (contract.batch, contract.q_pred, contract.patch_dim)
    assert source.samples_seen == contract.batch
    assert source.batches_seen == 1

    # A second batch keeps advancing samples_seen -- and since shuffle=True
    # means resampled=True (module docstring), this never raises
    # StopIteration even after exceeding the 450-sample local corpus.
    source.next_batch(rng)
    assert source.samples_seen == 2 * contract.batch


def test_real_train_source_cycles_past_local_sample_count():
    """A small (1-shard, 150-sample) train corpus is far smaller than a
    plausible step budget; per real_data.py's module docstring,
    `RealTrainSource` must keep yielding batches (resampled, with
    replacement) well past its own corpus size without raising.

    Deliberately uses a 1-shard `urls` subset (not the full 50-shard/7500-
    sample `TRAIN_SHARD_URLS` default) so this stays a fast unit test --
    `epoch_equiv` is instance-scoped (see real_data.py's `RealTrainSource
    .__init__`), so it's still an exact, meaningful check against *this*
    source's own (smaller) corpus.
    """
    contract = smoke_contract()
    # Use a batch size that will cross the 150-sample single-shard corpus
    # boundary within a handful of calls.
    contract = ShapeContract(
        batch=64,
        l_vis=contract.l_vis,
        q_pred=contract.q_pred,
        mask_ratio=contract.mask_ratio,
        img_size=contract.img_size,
        patch_size=contract.patch_size,
        embed_dim=contract.embed_dim,
        decoder_embed_dim=contract.decoder_embed_dim,
        encoder_heads=contract.encoder_heads,
        decoder_heads=contract.decoder_heads,
        class_token=contract.class_token,
    )
    one_shard_urls = real_data.TRAIN_SHARD_URLS[:1]  # 150 samples
    source = real_data.RealTrainSource(contract, urls=one_shard_urls)
    rng = np.random.default_rng(0)
    for _ in range(8):  # 8 * 64 = 512 > 150 = this source's own corpus size
        source.next_batch(rng)
    assert source.samples_seen == 8 * 64
    assert source.epoch_equiv > 1.0


def test_real_val_source_one_full_pass_drops_partial_batch():
    """Deliberately uses a 1-shard `urls` subset (not the full 17-shard/
    2550-sample `VAL_SHARD_URLS` default) so a full val pass -- run twice
    here -- stays a fast unit test.
    """
    contract = smoke_contract()
    one_shard_urls = real_data.VAL_SHARD_URLS[:1]  # 150 samples
    source = real_data.RealValSource(contract, urls=one_shard_urls)
    rng = np.random.default_rng(0)

    batches = list(source.eval_batches(rng))
    expected_num_batches = (real_data.SAMPLES_PER_SHARD * len(one_shard_urls)) // contract.batch
    assert len(batches) == expected_num_batches
    for b in batches:
        assert b.visible_values.shape == (contract.batch, contract.l_vis, contract.patch_dim)

    # Calling eval_batches again gives a fresh, independent pass (same count).
    batches2 = list(source.eval_batches(rng))
    assert len(batches2) == expected_num_batches


def test_train_loop_runs_on_real_data_with_eval():
    """End-to-end: a handful of train steps + one eval pass on the real
    shards, through `main_pretrain_tt.train`'s `train_batch_fn`/
    `eval_batch_iter_fn` hooks -- the same wiring `main()`'s
    `--eval-every` CLI path uses.

    `val_source` deliberately uses a 1-shard `urls` subset (not the full
    17-shard/2550-sample `VAL_SHARD_URLS` default): `eval_every=2` below
    means this test runs a *full* val pass twice, and each val batch here
    is a real train() forward/loss step, not just data loading -- a 1-shard
    (150-sample) val corpus keeps that affordable while still exercising
    the same eval_batch_iter_fn wiring `main()` uses.
    """
    contract = smoke_contract()
    cfg = TTMAEConfig(shape_contract=contract, base_lr=0.05)

    train_source = real_data.RealTrainSource(contract)
    val_source = real_data.RealValSource(contract, urls=real_data.VAL_SHARD_URLS[:1])

    history = train(
        contract,
        encoder_depth=1,
        decoder_depth=1,
        cfg=cfg,
        num_steps=4,
        warmup_steps=1,
        accum_iter=1,
        seed=0,
        log_every=1,
        print_log=False,
        train_batch_fn=train_source.next_batch,
        epoch_equiv_fn=lambda: train_source.epoch_equiv,
        eval_every=2,
        eval_batch_iter_fn=val_source.eval_batches,
    )

    assert len(history) == 4
    assert all(math.isfinite(entry.loss) for entry in history)
    # eval fired at step 1 (0-indexed, (1+1)%2==0) and at the final step (3).
    val_steps = [entry.step for entry in history if entry.val_loss is not None]
    assert val_steps == [1, 3]
    for entry in history:
        if entry.val_loss is not None:
            assert math.isfinite(entry.val_loss)

    assert train_source.samples_seen == 4 * contract.batch


def test_smoke_contract_matches_real_img_size():
    contract = smoke_contract()
    assert contract.img_size == (208, 240, 208)
    assert contract.patch_size == 8
