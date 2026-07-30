"""Tests for `ops_tt/tuned_linear.py`, the M6 tuned-linear Tier-2 op.

Op-level correctness is verified with `perf_parity.run_op_parity` against
`ttml.ops.linear.linear` (the existing, already-verified-correct op this is
meant to be a drop-in replacement for) at real ViT-L-relevant shapes
(`embed_dim=1024` in/out -- QKV/attn-proj scale; `1024 -> 4096` -- MLP fc1
scale), gated at `perf_parity.OP_FWD_PCC_GATE`/`OP_BWD_PCC_GATE` (no gate
loosening, per `perf_parity.py`'s module docstring).

`build_program_config`'s own fail-fast validation is tested directly
(host-side arithmetic, no device needed). `TunedLinearLayer` (the
`ttml.modules.LinearLayer` drop-in) is checked end-to-end against a real
`LinearLayer` built from the same init arrays, at a small-but-tile-aligned
shape, for a fast module-level sanity check on top of the op-level parity
above.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae_tt import perf_parity as PP  # noqa: E402
from smri_mae_tt.ops_tt import tuned_linear as TL  # noqa: E402
from ttml.modules import LinearLayer  # noqa: E402

SEED = 2026 + 906


@pytest.fixture(autouse=False)
def _device():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        yield ctx
    finally:
        ctx.close_device()


@pytest.fixture(autouse=True)
def _reset_state():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


# ---------------------------------------------------------------------------
# build_program_config: fail-fast validation (no device needed).
# ---------------------------------------------------------------------------


def test_build_program_config_rejects_non_tile_aligned_seq_len():
    with pytest.raises(ValueError, match="seq_len"):
        TL.build_program_config(batch=1, seq_len=100, in_features=1024, out_features=1024, grid_x=8, grid_y=8)


def test_build_program_config_rejects_non_tile_aligned_features():
    with pytest.raises(ValueError, match="in_features"):
        TL.build_program_config(batch=1, seq_len=128, in_features=100, out_features=1024, grid_x=8, grid_y=8)


def test_build_program_config_rejects_non_positive_batch():
    with pytest.raises(ValueError, match="batch"):
        TL.build_program_config(batch=0, seq_len=128, in_features=1024, out_features=1024, grid_x=8, grid_y=8)


def test_largest_divisor_leq_finds_exact_divisors():
    assert TL._largest_divisor_leq(32, 8) == 8  # 32 % 8 == 0
    assert TL._largest_divisor_leq(30, 8) == 6  # 30 % 8 != 0, largest divisor <=8 is 6
    assert TL._largest_divisor_leq(7, 8) == 7  # cap larger than n -> n itself


def test_iter_candidate_program_configs_nonempty_at_real_shapes():
    """Encoder QKV (`B=24,S=544,in=1024,out=3072`) and decoder MLP fc1
    (`B=24,S=2816,in=512,out=2048`) both real production shapes
    (`TENSTORRENT_PORT.md` section 2) -- the sweep needs a real candidate
    list at both to be useful."""
    for batch, seq_len, in_f, out_f in ((24, 544, 1024, 3072), (24, 2816, 512, 2048)):
        candidates = TL.iter_candidate_program_configs(
            batch=batch, seq_len=seq_len, in_features=in_f, out_features=out_f, grid_x=11, grid_y=10
        )
        assert len(candidates) > 0


def test_iter_candidate_program_configs_every_candidate_satisfies_exact_coverage():
    """Every candidate must satisfy the same real `TT_FATAL` constraints
    `build_program_config` enforces -- see this test's counterpart above."""
    batch, seq_len, in_f, out_f = 2, 128, 1024, 4096
    m_tiles = (batch * seq_len) // 32
    k_tiles = in_f // 32
    n_tiles = out_f // 32
    candidates = TL.iter_candidate_program_configs(
        batch=batch, seq_len=seq_len, in_features=in_f, out_features=out_f, grid_x=11, grid_y=10
    )
    assert len(candidates) > 1  # more than one distinct config, unlike build_program_config's single guess
    for pc in candidates:
        grid = pc.compute_with_storage_grid_size
        assert pc.per_core_M * grid.y == m_tiles
        assert pc.per_core_N * grid.x == n_tiles
        assert k_tiles % pc.in0_block_w == 0
        assert pc.per_core_M % pc.out_subblock_h == 0
        assert pc.per_core_N % pc.out_subblock_w == 0
        assert pc.out_subblock_h * pc.out_subblock_w <= 8


def test_iter_candidate_program_configs_deduplicates_and_respects_max_candidates():
    batch, seq_len, in_f, out_f = 24, 544, 1024, 3072
    candidates = TL.iter_candidate_program_configs(
        batch=batch, seq_len=seq_len, in_features=in_f, out_features=out_f, grid_x=11, grid_y=10, max_candidates=5
    )
    assert len(candidates) <= 5
    seen = set()
    for pc in candidates:
        grid = pc.compute_with_storage_grid_size
        key = (grid.x, grid.y, pc.in0_block_w, pc.out_subblock_h, pc.out_subblock_w)
        assert key not in seen, "iter_candidate_program_configs must not repeat a config"
        seen.add(key)


def test_iter_candidate_program_configs_rejects_non_tile_aligned_seq_len():
    with pytest.raises(ValueError, match="seq_len"):
        TL.iter_candidate_program_configs(batch=1, seq_len=100, in_features=1024, out_features=1024, grid_x=8, grid_y=8)


# ---------------------------------------------------------------------------
# Sweep-result cache: load/lookup, all host-only (no device needed).
# ---------------------------------------------------------------------------


def test_load_program_config_cache_missing_file_returns_empty(tmp_path):
    assert TL.load_program_config_cache(tmp_path / "does_not_exist.json") == {}


def test_cached_program_config_miss_returns_none():
    assert TL.cached_program_config({}, batch=24, seq_len=544, in_features=1024, out_features=3072) is None


def test_cached_program_config_hit_builds_matching_program_config():
    cache = {
        TL.cache_key(24, 544, 1024, 3072): {
            "compute_with_storage_grid_size": [6, 8],
            "in0_block_w": 4,
            "out_subblock_h": 1,
            "out_subblock_w": 2,
            "per_core_M": 51,
            "per_core_N": 16,
        }
    }
    pc = TL.cached_program_config(cache, batch=24, seq_len=544, in_features=1024, out_features=3072)
    assert pc is not None
    grid = pc.compute_with_storage_grid_size
    assert (grid.x, grid.y) == (6, 8)
    assert pc.in0_block_w == 4
    assert pc.per_core_M == 51
    assert pc.per_core_N == 16


def test_build_program_config_produces_exact_grid_coverage():
    """per_core_M * grid_y == m_tiles and per_core_N * grid_x == n_tiles
    exactly -- the real `TT_FATAL` requirement this function exists to
    satisfy (see module docstring)."""
    batch, seq_len, in_f, out_f = 2, 128, 1024, 4096
    pc = TL.build_program_config(batch=batch, seq_len=seq_len, in_features=in_f, out_features=out_f, grid_x=11, grid_y=10)
    m_tiles = (batch * seq_len) // 32
    n_tiles = out_f // 32
    grid_used = pc.compute_with_storage_grid_size
    assert pc.per_core_M * grid_used.y == m_tiles
    assert pc.per_core_N * grid_used.x == n_tiles


# ---------------------------------------------------------------------------
# Op-level parity vs. ttml.ops.linear.linear, at real ViT-L-relevant shapes.
# ---------------------------------------------------------------------------


def _linear_inputs_4d(rng: np.random.Generator, batch: int, seq: int, in_f: int, out_f: int) -> dict[str, np.ndarray]:
    return {
        "x": (rng.standard_normal((batch, 1, seq, in_f)) * 0.1).astype(np.float32),
        "weight": (rng.standard_normal((1, 1, out_f, in_f)) * 0.02).astype(np.float32),
        "bias": (rng.standard_normal((1, 1, 1, out_f)) * 0.01).astype(np.float32),
    }


def _reference_linear_op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
    return ttml.ops.linear.linear(leaves["x"], leaves["weight"], leaves["bias"])


def _make_tuned_linear_op(batch: int, seq_len: int, in_f: int, out_f: int) -> "PP.OpFn":
    def op(leaves: dict[str, "ttml.autograd.Tensor"]) -> "ttml.autograd.Tensor":
        device = leaves["x"].get_value().device()
        grid = device.compute_with_storage_grid_size()
        program_config = TL.build_program_config(
            batch=batch, seq_len=seq_len, in_features=in_f, out_features=out_f, grid_x=grid.x, grid_y=grid.y
        )
        return TL.tuned_linear(leaves["x"], leaves["weight"], leaves["bias"], program_config)

    return op


def test_tuned_linear_matches_default_linear_at_qkv_scale(_device):
    """embed_dim=1024 in/out -- ViT-L QKV/attn-proj-relevant scale."""
    batch, seq_len, dim = 2, 128, 1024
    rng = np.random.default_rng(SEED)
    inputs = _linear_inputs_4d(rng, batch, seq_len, dim, dim)
    candidate = _make_tuned_linear_op(batch, seq_len, dim, dim)

    result = PP.run_op_parity("tuned_linear-qkv_scale", _reference_linear_op, candidate, inputs)

    assert result.forward.pcc >= PP.OP_FWD_PCC_GATE
    assert set(result.backward) == {"x", "weight", "bias"}
    assert result.min_backward_pcc() >= PP.OP_BWD_PCC_GATE


def test_tuned_linear_matches_default_linear_at_mlp_scale(_device):
    """1024 -> 4096 -- ViT-L MLP fc1-relevant scale."""
    batch, seq_len, in_f, out_f = 2, 128, 1024, 4096
    rng = np.random.default_rng(SEED + 1)
    inputs = _linear_inputs_4d(rng, batch, seq_len, in_f, out_f)
    candidate = _make_tuned_linear_op(batch, seq_len, in_f, out_f)

    result = PP.run_op_parity("tuned_linear-mlp_scale", _reference_linear_op, candidate, inputs)

    assert result.forward.pcc >= PP.OP_FWD_PCC_GATE
    assert set(result.backward) == {"x", "weight", "bias"}
    assert result.min_backward_pcc() >= PP.OP_BWD_PCC_GATE


def test_tuned_linear_layer_survives_real_qkv_shape_with_no_cache_entry(_device):
    """Regression test: `TunedLinearLayer` at the *real* production QKV
    shape (`batch=24, seq_len=544, in=1024, out=3*1024`,
    `full_vitl_real_contract`) with no cache entry (`program_config_cache_path
    =None`, forcing the uncached fallback path unconditionally, regardless
    of whatever `mm_program_config_cache.json` happens to contain when this
    runs).

    Before the fallback fix (see `TunedLinearLayer.forward`'s inline
    comment), the uncached path called `build_program_config`, whose single
    maximal-grid guess crashes at exactly this shape/scale with
    `RuntimeError: TT_THROW ... Statically allocated circular buffers ...
    beyond max L1 size` (confirmed on real hardware while building
    `scripts/sweep_mm_block_sizes.py` -- the earlier tests in this file never
    caught it because they use `batch=2` and `out_features=dim`, not the
    real `batch=24`/`3*dim` QKV shape). This test exists so that regression
    is never silently reintroduced."""
    batch, seq_len, in_f, out_f = 24, 544, 1024, 3072
    rng = np.random.default_rng(SEED + 3)
    weight_np = (rng.standard_normal((1, 1, out_f, in_f)) * 0.02).astype(np.float32)
    bias_np = (rng.standard_normal((1, 1, 1, out_f)) * 0.01).astype(np.float32)
    x_np = (rng.standard_normal((batch, 1, seq_len, in_f)) * 0.1).astype(np.float32)

    def weight_init(shape, mapper=None):
        assert tuple(shape) == weight_np.shape
        return ttml.autograd.Tensor.from_numpy(weight_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    def bias_init(shape, mapper=None):
        assert tuple(shape) == bias_np.shape
        return ttml.autograd.Tensor.from_numpy(bias_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    ref_layer = LinearLayer(in_f, out_f, True, weight_init=weight_init, bias_init=bias_init)
    tuned_layer = TL.TunedLinearLayer(in_f, out_f, True, weight_init=weight_init, bias_init=bias_init, program_config_cache_path=None)

    x_ref = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    x_tuned = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    out_ref = ref_layer(x_ref)
    out_tuned = tuned_layer(x_tuned)  # must not raise -- see docstring

    report = PP.compare(
        "tuned_linear_layer.real_qkv_shape.forward", PP.to_numpy_fp32(out_ref), PP.to_numpy_fp32(out_tuned), pcc_gate=PP.OP_FWD_PCC_GATE
    )
    assert report.pcc >= PP.OP_FWD_PCC_GATE


# ---------------------------------------------------------------------------
# TunedLinearLayer: module-level drop-in-replacement sanity check against a
# real ttml.modules.LinearLayer built from the same init arrays.
# ---------------------------------------------------------------------------


def test_tuned_linear_layer_matches_linear_layer_forward_backward(_device):
    batch, seq_len, in_f, out_f = 1, 64, 128, 128
    rng = np.random.default_rng(SEED + 2)
    weight_np = (rng.standard_normal((1, 1, out_f, in_f)) * 0.02).astype(np.float32)
    bias_np = (rng.standard_normal((1, 1, 1, out_f)) * 0.01).astype(np.float32)
    x_np = (rng.standard_normal((batch, 1, seq_len, in_f)) * 0.1).astype(np.float32)

    def weight_init(shape, mapper=None):
        assert tuple(shape) == weight_np.shape
        return ttml.autograd.Tensor.from_numpy(weight_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    def bias_init(shape, mapper=None):
        assert tuple(shape) == bias_np.shape
        return ttml.autograd.Tensor.from_numpy(bias_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)

    ref_layer = LinearLayer(in_f, out_f, True, weight_init=weight_init, bias_init=bias_init)
    tuned_layer = TL.TunedLinearLayer(in_f, out_f, True, weight_init=weight_init, bias_init=bias_init)

    x_ref = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    x_ref.set_requires_grad(True)
    x_tuned = ttml.autograd.Tensor.from_numpy(x_np, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    x_tuned.set_requires_grad(True)

    out_ref = ref_layer(x_ref)
    out_tuned = tuned_layer(x_tuned)

    fwd_report = PP.compare(
        "tuned_linear_layer.forward", PP.to_numpy_fp32(out_ref), PP.to_numpy_fp32(out_tuned), pcc_gate=PP.OP_FWD_PCC_GATE
    )
    assert fwd_report.pcc >= PP.OP_FWD_PCC_GATE

    PP._scalar_mean_square(out_ref).backward(False)
    PP._scalar_mean_square(out_tuned).backward(False)

    for ref_p, tuned_p, name in (
        (x_ref, x_tuned, "x"),
        (ref_layer.weight.tensor, tuned_layer.weight.tensor, "weight"),
        (ref_layer.bias.tensor, tuned_layer.bias.tensor, "bias"),
    ):
        assert ref_p.is_grad_initialized(), f"{name}: reference has no gradient"
        assert tuned_p.is_grad_initialized(), f"{name}: tuned has no gradient"
        report = PP.compare(
            f"tuned_linear_layer.backward[{name}]",
            PP.to_numpy_fp32(ref_p.get_grad_tensor()),
            PP.to_numpy_fp32(tuned_p.get_grad_tensor()),
            pcc_gate=PP.OP_BWD_PCC_GATE,
        )
        assert report.pcc >= PP.OP_BWD_PCC_GATE
