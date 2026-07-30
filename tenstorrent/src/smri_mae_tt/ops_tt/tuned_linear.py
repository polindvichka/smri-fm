"""M6 candidate: a tuned linear/matmul Tier-2 op using an explicit
`ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` instead of the
`program_config=std::nullopt` default `ttml.ops.linear.linear` always uses
(`tt-train/sources/ttml/ops/linear_op.cpp::linear_op`: `ttnn::linear(...,
/* program_config */ std::nullopt, ..., /* core_grid */ core_grid)` -- it
only ever passes a `core_grid` hint, never a real tuned config, and its
backward (`ttnn_linear_backward`) calls `ttnn_fixed::matmul` with
`program_config` fixed at `std::nullopt` too, see `ttnn_fixed/matmuls.hpp`).

## Why `MatmulMultiCoreReuseMultiCastProgramConfig` on plain interleaved tensors

`models/demos/vision/classification/vit/blackhole/tt/
ttnn_optimized_sharded_vit_hiRes_bh.py` (read-only reference) only ever
builds this program_config for block-*sharded* L1 tensors -- reusing its
config *values* verbatim would have required also sharding every activation,
a much bigger change than "one tuned op." Empirically confirmed instead
(this module, and independently
`tests/ttnn/unit_tests/operations/matmul/test_matmul.py::
run_matmul_2d_multiple_output_blocks_per_core`, whose `in0_sharded=False`
branch builds `in0_memory_config = ttnn.L1_MEMORY_CONFIG` -- interleaved, not
sharded -- and still drives the exact same program_config type) that
`MatmulMultiCoreReuseMultiCastProgramConfig` works perfectly well against
plain interleaved (non-sharded) BFLOAT16 tensors, the same tensors
`ttml.ops.linear.linear` already produces/consumes. So this op stays a
drop-in replacement: no sharding, no changed memory layout anywhere else in
the model.

## Program-config field choices (see `build_program_config`)

Grid, block, and subblock sizes are derived from the op's own shape
(`batch`, `seq_len`, `in_features`, `out_features`) and the *caller's own
device* grid (`ttnn.CoreCoord` from `device.compute_with_storage_grid_size()`
-- queried fresh per call in `TunedLinearLayer.forward`, exactly how
`linear_op.cpp` itself does it: `tensor->get_value().device()->
compute_with_storage_grid_size()`), not hardcoded for one card. Field
semantics/defaults (`out_subblock_h=1`, `transpose_mcast=False`,
`fused_activation=None`) mirror the vit reference file's own
`query_key_value_matmul_program_config`/`ff1_matmul_program_config` shape,
per the task's "use these as a starting point" guidance -- this module does
not invent new field semantics, only computes concrete values for our own
tensor shapes.

Empirically confirmed while building this module (`git log`/session
scratch, not re-derivable from `ttnn`'s own docstrings): the matmul's
"number of output blocks along M" is computed from the tensor's *total*
logical M-volume, i.e. `batch * seq_len` tiled, not `seq_len` alone --
passing a grid sized only from `seq_len` raises `TT_FATAL ... num_blocks_y
<= grid.y` for `batch > 1`. `build_program_config` therefore always folds
`batch` into `per_core_M`'s derivation.

## Shape scope

`seq_len` must already be tile-aligned (multiple of 32) -- this op is meant
to replace the `LinearLayer`s inside `Attention`/`Mlp` (`modules_tt.py`),
which only ever run on `TransformerBlock`'s `seq_len` (encoder/decoder
`ShapeContract.encoder_seq_len`/`decoder_seq_len`, both validated
tile-aligned by `layout.ShapeContract.__post_init__` -- SDPA in the same
block already requires this). It is *not* meant for `patch_embed`/decoder
`proj`/`head`, which run on `l_vis`/`q_pred` directly and are not guaranteed
tile-aligned (`modules_tt.py`'s module docstring measured that those ops
tolerate non-tile-aligned `S` by relying on `ttnn`'s transparent padding --
this tuned op does not attempt that; it fails fast instead, per the
project's no-silent-fallback rule).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import ttml
import ttnn

from smri_mae_tt.layout import TILE
from ttml.autograd import Function
from ttml.modules import AbstractModuleBase, Parameter

# Default location of the sweep's cached winning configs (see
# `scripts/sweep_mm_block_sizes.py`), checked into the repo once populated so
# `TunedLinearLayer` gets sweep-quality configs without re-running the sweep
# on every fresh checkout. Empty/missing file -> every lookup misses -> every
# forward call uses `program_config=None` (see `TunedLinearLayer.forward` /
# `cached_program_config` below), so this module works standalone (as a
# no-op passthrough to the untuned default) even before the sweep has run.
DEFAULT_CACHE_PATH = Path(__file__).parent / "mm_program_config_cache.json"

# ---------------------------------------------------------------------------
# Program-config construction
# ---------------------------------------------------------------------------


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest `d` with `1 <= d <= min(n, cap)` and `n % d == 0`. `n >= 1`
    always has at least `d=1`, so this never fails to find a value."""
    if n <= 0:
        raise ValueError(f"_largest_divisor_leq: n must be positive, got {n}")
    if cap <= 0:
        raise ValueError(f"_largest_divisor_leq: cap must be positive, got {cap}")
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    raise AssertionError("unreachable: d=1 always divides n")  # pragma: no cover


def build_program_config(
    *,
    batch: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    grid_x: int,
    grid_y: int,
    max_in0_block_w: int = 8,
    max_out_subblock_w: int = 8,
) -> "ttnn.MatmulMultiCoreReuseMultiCastProgramConfig":
    """Build a `MatmulMultiCoreReuseMultiCastProgramConfig` for a
    `[batch, 1, seq_len, in_features] @ [1, 1, out_features, in_features]^T`
    linear op on a `(grid_x, grid_y)`-core device.

    Fails fast (no rounding/padding) if `seq_len`/`in_features`/
    `out_features` are not multiples of `TILE` (32) -- a mismatch here means
    the caller built its shapes incorrectly upstream, see this module's
    docstring ("Shape scope").

    Grid usage is capped to the largest divisor of the relevant tile count
    (`batch*seq_len` for M, `out_features` for N) that still fits the
    device's real grid -- this keeps every core doing an equal share of
    work (a real `TT_FATAL` requirement: the multicast kernel needs
    `per_core_M * grid_y == m_tiles` and `per_core_N * grid_x == n_tiles`
    exactly, not just `<=`), rather than guessing a grid and letting some
    cores idle or the op fail outright on non-divisible shapes.
    """
    for name, val in (("seq_len", seq_len), ("in_features", in_features), ("out_features", out_features)):
        if val <= 0 or val % TILE != 0:
            raise ValueError(f"build_program_config: {name}={val} must be a positive multiple of TILE ({TILE})")
    if batch <= 0:
        raise ValueError(f"build_program_config: batch must be positive, got {batch}")
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError(f"build_program_config: grid_x/grid_y must be positive, got ({grid_x}, {grid_y})")

    m_tiles = (batch * seq_len) // TILE
    k_tiles = in_features // TILE
    n_tiles = out_features // TILE

    grid_y_use = _largest_divisor_leq(m_tiles, grid_y)
    grid_x_use = _largest_divisor_leq(n_tiles, grid_x)
    per_core_m = m_tiles // grid_y_use
    per_core_n = n_tiles // grid_x_use
    in0_block_w = _largest_divisor_leq(k_tiles, max_in0_block_w)
    out_subblock_w = _largest_divisor_leq(per_core_n, max_out_subblock_w)

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_x_use, grid_y_use),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=False,
        fused_activation=None,
    )


# ---------------------------------------------------------------------------
# Candidate-config generation for `scripts/sweep_mm_block_sizes.py`.
#
# `build_program_config` above always picks the *single* maximal-grid config
# (largest divisor <= device cap on each axis) -- confirmed in this module's
# and `config_tt.SDPAProgramConfigRegistry`'s docstrings to be a wash
# (0.95-1.05x, noise) and to fail outright (L1 circular-buffer overflow) at
# the MLP shape. That single-config heuristic never explores the actual
# tuning space: other, smaller grid/block/subblock combinations may use more
# of the device's 110 cores in aggregate (TENSTORRENT_PORT.md's measured
# "only 64/110 cores used" gap) or avoid the CB-overflow failure mode
# entirely. `iter_candidate_program_configs` enumerates a bounded set of
# structurally-valid candidates (exact grid coverage, matching
# `build_program_config`'s own `TT_FATAL` constraints) for the sweep script
# to actually time on real hardware -- this function is pure host-side
# arithmetic, no device needed, so it's directly unit-testable (see
# `test_tuned_linear.py`).
# ---------------------------------------------------------------------------


def _divisors(n: int) -> list[int]:
    """All positive divisors of `n`, ascending. `n >= 1`."""
    if n <= 0:
        raise ValueError(f"_divisors: n must be positive, got {n}")
    small, large = [], []
    d = 1
    while d * d <= n:
        if n % d == 0:
            small.append(d)
            if d != n // d:
                large.append(n // d)
        d += 1
    return sorted(small + large)


def _divisors_leq_desc(n: int, cap: int, limit: int) -> list[int]:
    """Up to `limit` largest divisors of `n` that are `<= cap`, descending
    (the sweep prioritizes larger values first: for a grid axis that means
    more cores in use; the caller decides what `n`/`cap` mean)."""
    return [d for d in reversed(_divisors(n)) if d <= cap][:limit]


def iter_candidate_program_configs(
    *,
    batch: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    grid_x: int,
    grid_y: int,
    max_grid_candidates: int = 3,
    max_in0_block_w: int = 8,
    max_subblock_area: int = 8,
    max_candidates: int = 40,
) -> list["ttnn.MatmulMultiCoreReuseMultiCastProgramConfig"]:
    """Bounded, deduplicated list of structurally-valid
    `MatmulMultiCoreReuseMultiCastProgramConfig` candidates for a
    `[batch, 1, seq_len, in_features] @ [1, 1, out_features, in_features]^T`
    linear op on a `(grid_x, grid_y)`-core device -- for
    `scripts/sweep_mm_block_sizes.py` to time on real hardware and PCC-verify
    (this function only builds config *objects*, it runs nothing on device
    and cannot itself validate an L1/CB budget -- see module docstring).

    Every candidate satisfies the same exact-coverage constraint
    `build_program_config` enforces (`per_core_M * grid_y_used == m_tiles`,
    `per_core_N * grid_x_used == n_tiles`, `in0_block_w` divides `k_tiles`,
    `out_subblock_h`/`out_subblock_w` divide `per_core_M`/`per_core_N` with
    `out_subblock_h * out_subblock_w <= max_subblock_area`) -- these are real
    `TT_FATAL` requirements, not sweep-specific choices, so every candidate
    returned here at least *launches*; some may still fail from a resource
    budget this function cannot see (confirmed by hand: L1 CB overflow is a
    catchable `RuntimeError`, not a process crash -- the sweep script simply
    skips a candidate that raises).

    Ordered largest-grid-area first (the actual gap this sweep targets is
    core-count utilization), ties broken by largest subblock area (better
    `DST` register reuse per matmul call, standard tt-metal guidance), then
    largest `in0_block_w` (fewer reader-kernel round trips). Truncated to
    `max_candidates` after sorting, so a caller with a tight time budget
    always gets the most promising candidates, not an arbitrary prefix.
    """
    for name, val in (("seq_len", seq_len), ("in_features", in_features), ("out_features", out_features)):
        if val <= 0 or val % TILE != 0:
            raise ValueError(f"iter_candidate_program_configs: {name}={val} must be a positive multiple of TILE ({TILE})")
    if batch <= 0:
        raise ValueError(f"iter_candidate_program_configs: batch must be positive, got {batch}")
    if grid_x <= 0 or grid_y <= 0:
        raise ValueError(f"iter_candidate_program_configs: grid_x/grid_y must be positive, got ({grid_x}, {grid_y})")

    m_tiles = (batch * seq_len) // TILE
    k_tiles = in_features // TILE
    n_tiles = out_features // TILE

    grid_y_options = _divisors_leq_desc(m_tiles, grid_y, max_grid_candidates)
    grid_x_options = _divisors_leq_desc(n_tiles, grid_x, max_grid_candidates)
    in0_block_w_options = _divisors_leq_desc(k_tiles, max_in0_block_w, max_grid_candidates)

    seen: set[tuple] = set()
    scored: list[tuple[tuple[int, int, int], "ttnn.MatmulMultiCoreReuseMultiCastProgramConfig"]] = []

    for grid_y_use in grid_y_options:
        per_core_m = m_tiles // grid_y_use
        h_options = [h for h in _divisors(per_core_m) if h <= max_subblock_area]
        for grid_x_use in grid_x_options:
            per_core_n = n_tiles // grid_x_use
            for out_subblock_h in h_options:
                w_cap = max_subblock_area // out_subblock_h
                w_options = [w for w in _divisors(per_core_n) if w <= w_cap]
                for out_subblock_w in w_options:
                    for in0_block_w in in0_block_w_options:
                        key = (grid_x_use, grid_y_use, in0_block_w, out_subblock_h, out_subblock_w)
                        if key in seen:
                            continue
                        seen.add(key)
                        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(grid_x_use, grid_y_use),
                            in0_block_w=in0_block_w,
                            out_subblock_h=out_subblock_h,
                            out_subblock_w=out_subblock_w,
                            per_core_M=per_core_m,
                            per_core_N=per_core_n,
                            transpose_mcast=False,
                            fused_activation=None,
                        )
                        score = (grid_x_use * grid_y_use, out_subblock_h * out_subblock_w, in0_block_w)
                        scored.append((score, pc))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [pc for _score, pc in scored[:max_candidates]]


# ---------------------------------------------------------------------------
# Sweep-result cache: `{"<batch>|<seq_len>|<in_features>|<out_features>":
# {program-config fields..., "ms": float, "baseline_ms": float, "pcc": float}}`
# written by `scripts/sweep_mm_block_sizes.py`, read here so
# `TunedLinearLayer` can use a real, PCC-verified, measured-best config
# wherever the sweep has covered a shape and found one. Missing/uncovered
# shapes (or shapes the sweep tried and found no L1-safe candidate for --
# real at production scale, see `TunedLinearLayer.forward`) fall back to
# `program_config=None` (ttnn's own default), never to `build_program_config`
# -- this cache is purely additive, never required for correctness.
# ---------------------------------------------------------------------------


def cache_key(batch: int, seq_len: int, in_features: int, out_features: int) -> str:
    return f"{batch}|{seq_len}|{in_features}|{out_features}"


def load_program_config_cache(path: "Path | str" = DEFAULT_CACHE_PATH) -> dict:
    """Raw `{cache_key: fields-dict}` from disk, or `{}` if the file doesn't
    exist yet (the sweep hasn't been run) -- never raises on a missing file,
    since an empty cache is this module's valid default state."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as f:
        return json.load(f)


def cached_program_config(
    cache: dict, *, batch: int, seq_len: int, in_features: int, out_features: int
) -> Optional["ttnn.MatmulMultiCoreReuseMultiCastProgramConfig"]:
    """Look up a sweep-measured config for this exact shape, or `None` on a
    cache miss (caller falls back to `program_config=None`, i.e. ttnn's own
    default -- not `build_program_config`, see `TunedLinearLayer.forward`)."""
    entry = cache.get(cache_key(batch, seq_len, in_features, out_features))
    if entry is None:
        return None
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=tuple(entry["compute_with_storage_grid_size"]),
        in0_block_w=entry["in0_block_w"],
        out_subblock_h=entry["out_subblock_h"],
        out_subblock_w=entry["out_subblock_w"],
        per_core_M=entry["per_core_M"],
        per_core_N=entry["per_core_N"],
        transpose_mcast=False,
        fused_activation=None,
    )


# ---------------------------------------------------------------------------
# The Tier-2 custom op: raw ttnn.linear forward (explicit program_config),
# raw ttnn.matmul backward (no program_config -- see module docstring:
# ttml.ops.linear.linear's own backward, ttnn_linear_backward, doesn't tune
# its matmuls either, so this is an apples-to-apples comparison, and the
# task's own instructions call backward tuning optional/correctness-first).
# Pattern follows model_tt.py's _SqueezeDecoderPredAxis: forward/backward
# both raw ttnn ops, ctx.save_for_backward for the tensors backward needs.
# ---------------------------------------------------------------------------


class TunedLinearFunction(Function):
    """`y = x @ weight^T + bias` (matches `ttml.ops.linear.linear`'s exact
    convention -- `linear_op.cpp` calls `ttnn::linear(..., transpose_a=false,
    transpose_b=true, ...)`), forward via `ttnn.linear` with an explicit
    tuned `program_config`, backward via plain `ttnn.matmul`/`ttnn.sum`.

    `x`: `[B, 1, S, in_features]`. `weight`: `[1, 1, out_features,
    in_features]`. `bias`: `[1, 1, 1, out_features]` or `None` -- matches
    `params.py`'s documented TT shape convention and `modules_tt.LinearParams`
    exactly, so this is a drop-in replacement for `ttml.ops.linear.linear`
    wherever that convention is already used (`ttml.modules.LinearLayer`,
    `modules_tt.py`'s `_const_init`).

    Backward derivation (all plain 2D matmuls after flattening `B*S` into a
    single `M` row-dimension, exactly mirroring `linear_op.cpp::
    ttnn_linear_backward`'s own reshape-to-2D-then-matmul approach -- *not*
    a batched 4D matmul, which would keep the batch axis unContracted and
    give one `[out,in]` weight-gradient matrix per batch element instead of
    the correctly-summed one):
        dInput  = dOutput @ Weight                  (M,out)@(out,in) = (M,in)
        dWeight = dOutput^T @ Input                  (out,M)@(M,in) = (out,in)
        dBias   = sum_M(dOutput)                                 (1,out)
    """

    @staticmethod
    def forward(ctx, x: "ttml.autograd.Tensor", weight: "ttml.autograd.Tensor", bias, program_config):
        x_val = x.get_value()
        w_val = weight.get_value()
        b_val = bias.get_value() if bias is not None else None

        out = ttnn.linear(
            x_val,
            w_val,
            bias=b_val,
            transpose_a=False,
            transpose_b=True,
            program_config=program_config,
        )

        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        x_val = x.get_value()
        w_val = weight.get_value()

        x_shape = tuple(x_val.shape)
        w_shape = tuple(w_val.shape)
        go_shape = tuple(grad_output.shape)
        in_features = x_shape[-1]
        out_features = w_shape[-2]
        m = 1
        for s in go_shape[:-1]:
            m *= int(s)

        go_2d = ttnn.reshape(grad_output, ttnn.Shape([m, out_features]))
        x_2d = ttnn.reshape(x_val, ttnn.Shape([m, in_features]))
        w_2d = ttnn.reshape(w_val, ttnn.Shape([out_features, in_features]))

        dx_2d = ttnn.matmul(go_2d, w_2d, transpose_a=False, transpose_b=False)  # (M,out)@(out,in) = (M,in)
        dw_2d = ttnn.matmul(go_2d, x_2d, transpose_a=True, transpose_b=False)  # (out,M)@(M,in) = (out,in)

        dx = ttnn.reshape(dx_2d, ttnn.Shape(list(x_shape)))
        dw = ttnn.reshape(dw_2d, ttnn.Shape(list(w_shape)))

        if ctx.has_bias:
            db_2d = ttnn.sum(go_2d, dim=0, keepdim=True)  # (1, out)
            db = ttnn.reshape(db_2d, ttnn.Shape([1, 1, 1, out_features]))
            return dx, dw, db
        return dx, dw


def tuned_linear(
    x: "ttml.autograd.Tensor",
    weight: "ttml.autograd.Tensor",
    bias,
    program_config,
) -> "ttml.autograd.Tensor":
    """Functional entry point -- see `TunedLinearFunction`."""
    return TunedLinearFunction.apply(x, weight, bias, program_config)


# ---------------------------------------------------------------------------
# Module wrapper: drop-in replacement for `ttml.modules.LinearLayer`. Same
# constructor signature/shape convention, so it can be swapped in wherever a
# `LinearLayer` is built from `weight_init`/`bias_init` callables
# (`modules_tt.py`'s `_const_init`) without touching any other code at the
# call site.
# ---------------------------------------------------------------------------


class TunedLinearLayer(AbstractModuleBase):
    """`ttml.modules.LinearLayer`, but forward uses `tuned_linear` with a
    sweep-measured, PCC-verified `program_config` wherever
    `scripts/sweep_mm_block_sizes.py` has covered the exact runtime shape
    (`cached_program_config`), instead of `ttml.ops.linear.linear`'s
    `program_config=None` default.

    `program_config_cache_path` (default `DEFAULT_CACHE_PATH`): loaded once
    at construction (`load_program_config_cache` does file I/O; forward()
    itself does not). On a cache miss, forward() uses `program_config=None`
    -- the same default `ttml.ops.linear.linear` uses -- **not**
    `build_program_config`'s single-guess heuristic (see forward()'s inline
    comment: that heuristic is confirmed to crash, not just underperform, at
    real production QKV/MLP-fc1 shapes -- an L1 circular-buffer overflow).
    This makes `use_tuned_linear=True` strictly safe to enable everywhere:
    covered shapes get a real speedup, uncovered shapes are a no-op, nothing
    crashes. Pass `program_config_cache_path=None` to disable the cache
    entirely (every call is then a `program_config=None` no-op passthrough
    -- e.g. a benchmark that wants the pre-cache baseline for comparison).

    Scope: `x` must be rank-4 `[B, 1, S, in_features]` with `S` a multiple
    of `TILE` (32) -- see this module's docstring, "Shape scope". Not a
    general-purpose linear layer; only meant for the tile-aligned
    `Attention`/`Mlp` call sites this task wires it into.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_init=None,
        bias_init=None,
        program_config_cache_path: "Path | str | None" = DEFAULT_CACHE_PATH,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._config_cache = load_program_config_cache(program_config_cache_path) if program_config_cache_path is not None else {}

        if weight_init is None:
            k = math.sqrt(1.0 / in_features)
            weight_init = ttml.init.uniform(-k, k)
        if bias_init is None:
            k = math.sqrt(1.0 / in_features)
            bias_init = ttml.init.uniform(-k, k)

        weight_shape = (1, 1, out_features, in_features)
        self.weight = Parameter(weight_init(weight_shape))

        if has_bias:
            bias_shape = (1, 1, 1, out_features)
            self.bias = Parameter(bias_init(bias_shape))
        else:
            self.bias = None

    def forward(self, x: "ttml.autograd.Tensor") -> "ttml.autograd.Tensor":
        x_shape = tuple(x.shape())
        if len(x_shape) != 4 or x_shape[1] != 1:
            raise ValueError(f"TunedLinearLayer expects a rank-4 [B,1,S,in_features] input, got shape {x_shape}")
        batch, _one, seq_len, in_f = x_shape
        if in_f != self.in_features:
            raise ValueError(f"TunedLinearLayer input last dim {in_f} != configured in_features {self.in_features}")

        # Cache hit -> sweep-measured, PCC-verified-on-real-hardware config.
        # Cache miss -> `program_config=None` (ttnn's own default choice),
        # NOT `build_program_config`'s heuristic. Empirically confirmed while
        # building `scripts/sweep_mm_block_sizes.py`: that heuristic's single
        # maximal-grid guess is *exactly* the sweep's first-tried candidate,
        # and at real production scale (batch=24, encoder QKV/MLP-fc1 --
        # `full_vitl_real_contract`) it raises `RuntimeError: TT_THROW ...
        # Statically allocated circular buffers ... beyond max L1 size` --
        # `config_tt.MatmulProgramConfigRegistry`'s dead-end #2 ("fails
        # outright at the MLP shape") was this exact failure. Falling back
        # to it here would make `use_tuned_linear=True` *less* safe than
        # `False` at shapes the sweep hasn't covered (or found no safe
        # candidate for) -- falling back to `None` instead guarantees this
        # layer can only match or beat the untuned baseline, never crash it.
        program_config = cached_program_config(
            self._config_cache,
            batch=batch,
            seq_len=seq_len,
            in_features=self.in_features,
            out_features=self.out_features,
        )

        bias_t = self.bias.tensor if self.bias is not None else None
        return tuned_linear(x, self.weight.tensor, bias_t, program_config)
