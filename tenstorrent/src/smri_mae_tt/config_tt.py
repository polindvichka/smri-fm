"""Training configuration, AdamW param-group construction, and per-shape
program-config registries for the TT-NN MAE port (TENSTORRENT_PORT.md
section 6: "`config_tt.py` deserves emphasis: on TT the per-op program
config *is* part of the model definition, and it is shape-dependent.").

This file is organized in three independent pieces:

1. `TTMAEConfig` -- the training hyperparameters relevant to a TT loop,
   mirroring `/home/tt-small/projects/smri-fm/src/smri_mae/config/default_pretrain.yaml`.
2. AdamW param-group construction mirroring `src/smri_mae/utils.py`'s
   `get_param_groups`/`_fuse_param_groups`/`update_lr`/`update_wd`
   (lines 467-508) -- see the long docstring on `MultiGroupAdamW` below for
   a real capability gap this had to work around.
3. `SDPAProgramConfigRegistry`/`MatmulProgramConfigRegistry` -- placeholder
   structure for section 6's "keep it in one file keyed by `(B, S, D, H)`",
   populated with tuning data at M6, not now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Protocol, Sequence

import ttml
import ttnn

from .layout import ShapeContract

# ---------------------------------------------------------------------------
# 1. Training config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TTMAEConfig:
    """Model/optimization hyperparameters relevant to a TT training loop.

    Deliberately narrower than `default_pretrain.yaml`: dataset, wandb,
    eval, and checkpoint-sync fields belong to the not-yet-written
    `main_pretrain_tt.py`, not here (TENSTORRENT_PORT.md's target structure,
    section 6, lists them as separate files/concerns). `img_size`,
    `patch_size`, `mask_ratio`, and micro-`batch` are NOT duplicated here --
    they already live on `layout.ShapeContract` (which additionally pins the
    TT-specific `l_vis`/`q_pred` static-shape constants the yaml has no
    field for at all, since they don't exist in the ragged PyTorch model);
    `shape_contract` is the single source of truth for all of them.

    Every field below that mirrors the yaml uses the yaml's own default
    (`default_pretrain.yaml:16-71`); `adam_epsilon`/`adam_amsgrad` have no
    yaml equivalent (the reference recipe uses `torch.optim.AdamW`'s default
    epsilon, and `amsgrad` is a `ttml`-specific extension with no analog in
    the reference recipe) so they default to PyTorch's AdamW default
    (`eps=1e-8`) and `False` respectively, documented here rather than
    silently inherited from `AdamWFullPrecisionConfig.make`'s own defaults
    (see `MultiGroupAdamW`'s docstring for why `AdamWFullPrecision`, not the
    plain `AdamW` its name might suggest, is the class this config feeds).
    """

    shape_contract: ShapeContract

    # default_pretrain.yaml:17 (`pred_mask_ratio: null`).
    pred_mask_ratio: Optional[float] = None

    # default_pretrain.yaml:66-71.
    base_lr: float = 0.001
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.95)
    clip_grad: float = 1.0

    # default_pretrain.yaml:64 (`accum_iter: 1`; TENSTORRENT_PORT.md section 5
    # notes the 4090 baseline actually ran `accum_iter: 5` at micro-batch 12 --
    # that is a run-specific override, not this yaml's default, so it is not
    # hardcoded here).
    accum_iter: int = 1

    # No yaml equivalent (see class docstring).
    adam_epsilon: float = 1e-8
    adam_amsgrad: bool = False

    def __post_init__(self) -> None:
        if len(self.betas) != 2:
            raise ValueError(f"betas must be a (beta1, beta2) pair, got {self.betas!r}")
        if not (0.0 <= self.betas[0] < 1.0 and 0.0 <= self.betas[1] < 1.0):
            raise ValueError(f"betas must each be in [0, 1), got {self.betas!r}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if self.accum_iter < 1:
            raise ValueError(f"accum_iter must be >= 1, got {self.accum_iter}")
        if self.clip_grad <= 0:
            raise ValueError(f"clip_grad must be > 0, got {self.clip_grad}")
        if self.pred_mask_ratio is not None and not (0.0 <= self.pred_mask_ratio < 1.0):
            raise ValueError(f"pred_mask_ratio must be in [0, 1) or None, got {self.pred_mask_ratio}")

    def lr_for_total_batch(self, total_batch_size: int) -> float:
        """`base_lr * total_batch_size / 256` (`main_pretrain.py:113`) --
        the linear LR scaling rule the reference recipe uses. `total_batch_size`
        is `micro_batch * accum_iter * world_size`
        (`main_pretrain.py:106`; TENSTORRENT_PORT.md section 5: "keep the
        global batch fixed and halve per-card micro-batch, so the LR
        schedule ... transfers unchanged").
        """
        if total_batch_size <= 0:
            raise ValueError(f"total_batch_size must be > 0, got {total_batch_size}")
        return self.base_lr * total_batch_size / 256.0


# ---------------------------------------------------------------------------
# 2. AdamW param groups
# ---------------------------------------------------------------------------


class _NamedTensorLike(Protocol):
    def get_requires_grad(self) -> bool: ...  # noqa: E704


@dataclass(frozen=True)
class ParamGroupSpec:
    """One fused param group: every name in `param_names` shares the same
    `(lr_multiplier, wd_multiplier)` pair. Mirrors one entry of
    `utils._fuse_param_groups`'s output (`utils.py:488-498`).
    """

    lr_multiplier: float
    wd_multiplier: float
    param_names: tuple[str, ...]


def build_param_group_specs(
    named_parameters: Mapping[str, object],
    patch_embed_lr_mult: float = 1.0,
) -> list[ParamGroupSpec]:
    """Reproduce `utils.get_param_groups`/`_fuse_param_groups` (`utils.py:467-498`)
    over a generic `{name: parameter}` mapping.

    Rules, exactly as in `utils.py:471-482`:
      - skip any parameter with `requires_grad == False` (checked via
        `.get_requires_grad()` if present -- `ttml.autograd.Tensor`'s bound
        method, `nb_autograd.cpp:105` -- so this works uniformly whether
        `named_parameters` holds real `ttml.autograd.Tensor`s, once
        `modules_tt.py` exists, or plain test doubles that don't define it,
        in which case every entry is treated as trainable).
      - `wd_multiplier = 0.0` if the name ends with `.bias`, or contains
        `"norm"` or `"gamma"` -- else `1.0`.
      - `lr_multiplier = patch_embed_lr_mult` if the name contains
        `"patch_embed"` -- else `1.0`.
      - names with an identical `(lr_multiplier, wd_multiplier)` pair are
        fused into one `ParamGroupSpec` (`_fuse_param_groups`'s
        `identifier` join, `utils.py:491-495`, specialized to exactly these
        two keys since `utils.get_param_groups` never sets any other key on
        `d` before fusing).

    Written generically against "a mapping of name -> parameter" per
    TENSTORRENT_PORT.md's instruction to not import `modules_tt.py` before
    it exists; `dict(module.named_parameters())` (the convention
    `tt-train/tests/python/test_fsdp.py` already uses throughout, e.g.
    `for name, t in block.named_parameters():`) satisfies this directly
    once that module lands.
    """
    groups: dict[tuple[float, float], list[str]] = {}
    for name, param in named_parameters.items():
        get_requires_grad = getattr(param, "get_requires_grad", None)
        if callable(get_requires_grad) and not get_requires_grad():
            continue
        wd_multiplier = 0.0 if (name.endswith(".bias") or "norm" in name or "gamma" in name) else 1.0
        lr_multiplier = patch_embed_lr_mult if "patch_embed" in name else 1.0
        groups.setdefault((lr_multiplier, wd_multiplier), []).append(name)

    return [
        ParamGroupSpec(lr_multiplier=key[0], wd_multiplier=key[1], param_names=tuple(names))
        for key, names in sorted(groups.items())
    ]


class MultiGroupAdamW:
    """AdamW over multiple param groups with independent `lr`/`weight_decay`,
    built as N cooperating `ttml.optimizers.AdamW` instances.

    ## The capability gap this works around

    `ttml.optimizers.AdamW` has **no param-group concept at all**. Its
    constructor is `AdamW(parameters: NamedParameters, config: AdamWConfig)`
    (confirmed via `help(ttml.optimizers.AdamW)` on the built wheel, and in
    source: `tt-train/sources/ttml/optimizers/adamw.hpp:22-27`,
    `nb_optimizers.cpp:244-248` binds only that one constructor) -- a single
    `AdamWConfig` (one scalar `lr`/`weight_decay`/`beta1`/`beta2`/`epsilon`)
    applied *uniformly* to every parameter in `step()`
    (`adamw.cpp:55-89`: the loop over `m_parameters` calls
    `ttml::metal::adamw(..., m_config.lr, ..., m_config.weight_decay, ...)`
    with the exact same `m_config` for every single named parameter, every
    step). There is no per-parameter or per-group override mechanism
    anywhere in `optimizers/` -- `SGD`, `Muon`, `AdamWFullPrecision`, and
    `RemoteOptimizer` all follow the identical single-`NamedParameters`
    plus-single-`Config` pattern. This is a genuine gap versus PyTorch's
    `torch.optim.AdamW(param_groups)`, which `utils.get_param_groups`
    (`utils.py:467-485`) depends on for "no weight decay on 1D params and
    biases" (TENSTORRENT_PORT.md section 6).

    A second, narrower gap: even `weight_decay` alone has no Python-level
    setter. `AdamW::set_weight_decay` exists in the C++ header
    (`adamw.hpp:50-51`) but `nb_optimizers.cpp:244-248` does not bind it (or
    `get_weight_decay`) to Python at all -- only the inherited
    `OptimizerBase::set_lr`/`get_lr` are exposed
    (`nb_optimizers.cpp:158`, confirmed via `help(ttml.optimizers.AdamW)`
    showing `set_lr`/`get_lr` under "Methods inherited from OptimizerBase"
    and no `weight_decay` accessor at all). So even a *single* `AdamW`
    instance can only ever have its `lr` changed after construction; its
    `weight_decay` is frozen at whatever `AdamWConfig` it was built with.

    ## The fix: one AdamW instance per fused group, not one AdamW with groups

    Since an AdamW parameter update depends only on that parameter's own
    gradient/state and the config used for *its* group -- there is no
    cross-group term in the update rule itself (global-norm gradient
    clipping is the only cross-parameter computation in this pipeline, and
    it is intentionally done as one call over every parameter regardless of
    grouping; see `clip_grad_norm` below) -- constructing one
    `ttml.optimizers.AdamW` per `ParamGroupSpec`, each scoped to only that
    group's `NamedParameters` subset and given its own `lr = base_lr *
    lr_multiplier` / `weight_decay = weight_decay * wd_multiplier`
    `AdamWConfig`, is an exact algebraic reformulation of "one optimizer, N
    param groups" as "N optimizers, each with one group" -- not an
    approximation of `get_param_groups`'s semantics, a correct
    implementation of them. The cost is N separate `exp_avg`/`exp_avg_sq`
    state trees (same total size, just partitioned by group instead of by
    parameter-within-one-object) and N fused-kernel dispatch loops per
    optimizer step instead of one; with `patch_embed_lr_mult=1.0` (this
    yaml's implicit default -- `main_pretrain.py:118` calls
    `get_param_groups(model)` with no override) `build_param_group_specs`
    collapses the whole 316M-parameter model to exactly **2** groups (decay,
    no-decay), so this is 2 `AdamW` instances, not hundreds.

    `set_lr` (live schedule support, e.g. `WarmupThenCosine`, `utils.py:391-421`)
    works per-group via this class's own `set_lr`. `weight_decay` cannot be
    live-updated per-group for the reason above -- this matches the
    reference recipe, which never changes `weight_decay` mid-run
    (`default_pretrain.yaml` sets one constant `weight_decay: 0.05` for the
    whole run; only `base_lr` follows a per-step schedule via
    `utils.update_lr`).

    ## A third, separately-discovered gotcha: plain `AdamW` silently no-ops on fp32 parameters

    `ttml.optimizers.AdamW::step()` (`adamw.cpp:55-91`) fetches
    `theta_ptr->get_value(PreferredPrecision::HALF)` and passes that tensor
    straight into `ttml::metal::adamw` for an **in-place** update -- it never
    calls `theta_ptr->set_value(...)` afterward. For a parameter already
    stored as bf16, `get_value(HALF)` is a no-op accessor returning the same
    underlying buffer, so the in-place kernel writes land in the real
    parameter. For a parameter stored as fp32 (or anything else), the same
    call implicitly casts to a *new, temporary* bf16 tensor, and the
    in-place write lands in that throwaway copy -- the actual parameter
    never changes, silently, with no error or warning. Verified empirically
    (see `test_config_tt.py`'s device tests): an fp32-stored parameter is
    bit-for-bit unchanged after `AdamW.step()`; the identical setup with a
    bf16-stored parameter updates as expected.

    `ttml.optimizers.AdamWFullPrecision` (`adamw_full_precision.cpp:20-101`)
    does not have this hazard -- it builds its own explicit fp32 master-weight
    copy at construction (`ttnn::typecast(bf16_weights, FLOAT32)`,
    `adamw_full_precision.cpp:26-27`), updates *that* in place, and then
    **always** does `theta_ptr->set_value(updated_bf16_weights)`
    (`adamw_full_precision.cpp:98-100`) regardless of what dtype the
    parameter tensor happened to be. This also happens to be the one that
    matches TENSTORRENT_PORT.md section 6's numerics mandate verbatim --
    "AdamW state at fp32 (m, v, master)" -- which plain `AdamW`'s bf16
    `exp_avg`/`exp_avg_sq` (`adamw.cpp:26-33`: `zeros_like(tensor_ptr->get_value(HALF))`)
    does not satisfy at all. For both reasons, `MultiGroupAdamW` below is
    built on `ttml.optimizers.AdamWFullPrecision`/`AdamWFullPrecisionConfig`,
    not the plain `AdamW` this class's own name might suggest -- "AdamW" is
    kept in this wrapper's name because that is what the training recipe and
    TENSTORRENT_PORT.md call the algorithm, not because it wraps `ttml`'s
    class literally named `AdamW`. `AdamWFullPrecisionConfig.make` has no
    `stochastic_rounding` parameter (confirmed: `adamw_full_precision.hpp:12-18`
    has no such field, and `nb_optimizers.cpp:224-241`'s binding has no such
    argument) -- that knob is specific to the bf16-state `AdamW` class and
    does not apply here.
    """

    def __init__(
        self,
        named_parameters: Mapping[str, "ttml.autograd.Tensor"],
        group_specs: Sequence[ParamGroupSpec],
        *,
        base_lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        epsilon: float = 1e-8,
        amsgrad: bool = False,
    ) -> None:
        beta1, beta2 = betas
        seen: set[str] = set()
        self._group_specs: list[ParamGroupSpec] = []
        self._optimizers: list["ttml.optimizers.AdamWFullPrecision"] = []

        for spec in group_specs:
            if not spec.param_names:
                continue
            for name in spec.param_names:
                if name in seen:
                    raise ValueError(f"parameter {name!r} claimed by more than one param group")
                if name not in named_parameters:
                    raise KeyError(f"param group references {name!r}, not present in named_parameters")
                seen.add(name)

            group_params = {name: named_parameters[name] for name in spec.param_names}
            cfg = ttml.optimizers.AdamWFullPrecisionConfig.make(
                lr=base_lr * spec.lr_multiplier,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                weight_decay=weight_decay * spec.wd_multiplier,
                amsgrad=amsgrad,
            )
            self._group_specs.append(spec)
            self._optimizers.append(ttml.optimizers.AdamWFullPrecision(group_params, cfg))

        if not self._optimizers:
            raise ValueError("no non-empty param groups given -- nothing to optimize")

    def step(self) -> None:
        for opt in self._optimizers:
            opt.step()

    def zero_grad(self) -> None:
        for opt in self._optimizers:
            opt.zero_grad()

    def set_lr(self, base_lr: float) -> None:
        """Mirrors `utils.update_lr`: `group['lr'] = base_lr * group['lr_multiplier']`
        (`utils.py:501-503`), applied per underlying `AdamW` instance via its
        (Python-exposed) `set_lr`.
        """
        for opt, spec in zip(self._optimizers, self._group_specs):
            opt.set_lr(base_lr * spec.lr_multiplier)

    @property
    def group_specs(self) -> tuple[ParamGroupSpec, ...]:
        return tuple(self._group_specs)


def clip_grad_norm(
    named_parameters: Mapping[str, "ttml.autograd.Tensor"],
    max_norm: float,
    *,
    p_norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> "ttml.autograd.Tensor":
    """Global gradient-norm clipping, over every parameter regardless of
    `MultiGroupAdamW` grouping (matches PyTorch's `clip_grad_norm_` scope,
    and `utils.clip_grad`'s `nn.utils.clip_grad_norm_(params, max_norm, ...)`
    called over ALL of `optimizer.param_groups`, `utils.py:455-463`).

    TENSTORRENT_PORT.md section 6: "Gradient clipping at 1.0 maps to
    `moreh.clip_grad_norm`." Already conveniently bound to Python -- not
    `ttml.moreh.*` (no such submodule) and not `ttnn.operations.moreh.*`
    directly, but `ttml.core.clip_grad_norm`
    (`nb_core.cpp:118-131`: `m.def("clip_grad_norm", ...)` inside the `core`
    submodule, itself a thin wrapper over
    `ttnn::moreh_clip_grad_norm` via `ttml::core::clip_grad_norm`,
    `core/clip_grad_norm.cpp`). This function is a thin, config-bound
    convenience over that existing binding -- not a new implementation --
    mirroring `utils.clip_grad`'s call shape
    (`clip_grad(optimizer, max_norm=args.clip_grad)`, `main_pretrain.py:360`)
    for this port's `TTMAEConfig.clip_grad`. Returns the (pre-clip) total
    norm, as `ttml.core.clip_grad_norm` and `utils.clip_grad` both do.
    """
    return ttml.core.clip_grad_norm(
        dict(named_parameters), max_norm, p_norm_type=p_norm_type, error_if_nonfinite=error_if_nonfinite
    )


# ---------------------------------------------------------------------------
# 3. Per-shape program config registries (placeholder structure, M6 fills it in)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpShapeKey:
    """`(B, S, D, H)` per TENSTORRENT_PORT.md section 6: batch, sequence
    length, embedding dim, head count. This is the full set of static shape
    parameters that determine both the encoder's and decoder's SDPA/matmul
    program configs under this port's static-shape contract (`layout.py`'s
    `ShapeContract` -- `B=batch`, `S=encoder_seq_len`/`decoder_seq_len`,
    `D=embed_dim`/`decoder_embed_dim`, `H=encoder_heads`/`decoder_heads`).
    """

    batch: int
    seq_len: int
    embed_dim: int
    heads: int


@dataclass
class SDPAProgramConfigRegistry:
    """`{(B, S, D, H): ttnn.SDPAProgramConfig | None}`, keyed by `OpShapeKey`.

    `ttnn.SDPAProgramConfig` is confirmed a real, already-shipped `ttnn` type
    (not a `ttml` one): `help(ttnn.SDPAProgramConfig)` on the built wheel
    shows `__init__(self, *, compute_with_storage_grid_size: CoreCoord,
    sub_core_grids=None, q_chunk_size: int, k_chunk_size: int,
    exp_approx_mode=None, max_cores_per_head_batch: int = 16)`, matching
    TENSTORRENT_PORT.md section 0's measured `q_chunk=k_chunk=512` config
    that gave the 7.4x SDPA speedup.

    **This registry is unpopulated by design** -- TENSTORRENT_PORT.md
    section 7 (M6) is explicit that tuning is future work ("Levers, in
    order: (1) tune `SDPAProgramConfig`"), and this task has no tuning data
    yet. Every lookup returns `None` ("let `ttnn` pick a default") unless
    explicitly registered.

    **Important caveat, checked while building this**: as of this build,
    `ttml.ops.attention.scaled_dot_product_attention`
    (`ops/scaled_dot_product_attention.cpp`/`.hpp`) has no `program_config`
    parameter anywhere in its signature or implementation (grepped for
    `program_config`/`ProgramConfig` in both files: zero matches) -- nor do
    `ttml.ops.linear.linear`/`ttml.ops.matmul.matmul`
    (`linear_op.hpp`/`matmul_op.hpp`: same, zero matches). So a value stored
    here cannot yet be threaded into the fused training-SDPA kernel or
    `ttml`'s matmul/linear wrappers through any existing Python or C++ hook
    in `ttml` -- this registry is the *structure* section 6 asks for ("keep
    it in one file keyed by `(B, S, D, H)`"), ready to be populated and
    wired in once M6 either extends `ttml`'s ops-layer wrappers to accept a
    program config (a small, scoped C++/nanobind change) or a tier-2/3
    custom SDPA (section 4, item 4) is written with one built in directly.

    **M6 first-pass finding (2026-07-27), deeper than the caveat above**:
    the gap isn't just a missing nanobind passthrough -- it's structural.
    `ttml::metal::sdpa_fw`/`sdpa_bw` (`tt-train/sources/ttml/metal/ops/
    sdpa_fw{,/device}/sdpa_fw.{hpp,cpp}`, the fused kernel actually used for
    training via `ops_tt/attention.py`) accept **no `program_config`
    parameter at the C++ API level at all** -- not merely un-exposed to
    Python. Its internal K/V chunk size is picked automatically by
    `pick_sk_chunk_t()` (`sdpa_fw_program_factory.cpp`), a heuristic capped
    at 4 tiles (128 elements) to fit `DST` under `fp32_dest_acc_en=true`,
    with no override hook. `ttnn.SDPAProgramConfig` (`q_chunk_size`/
    `k_chunk_size` up to 512, matching TENSTORRENT_PORT.md section 0's
    measured 7.4x) is a *different* op's config type --
    `ttnn.transformer.scaled_dot_product_attention`
    (`ttnn/cpp/ttnn/operations/transformer/sdpa/`), a separate
    FlashAttention-2 implementation with no `ttml`/autograd wrapper and,
    confirmed by grepping its nanobind bindings and the rest of `ttnn`, **no
    backward op at all** (`help()` on every `ttnn.transformer.*sdpa*`
    function shows forward-only signatures; used in this build only by
    inference demos, e.g. `models/demos/vision/classification/vit/
    blackhole/tt/ttnn_optimized_sharded_vit_hiRes_bh.py`, which calls it
    with `is_causal=False` and a `q_chunk_size=k_chunk_size=256`
    `SDPAProgramConfig`).

    Measured on real Blackhole hardware, B=8/H=16/S=2816/head_dim=32
    (`full_vitl_real_contract`'s decoder shape at reduced batch), 5-iter
    average, all `ttnn.synchronize_device` between iterations:
    - `ttml` fused SDPA (the actual training path): forward-only 80.36 ms,
      forward+backward 130.98 ms/iter.
    - `ttnn.transformer.scaled_dot_product_attention` (forward only, no
      backward exists): default `program_config=None` 21.91 ms, tuned
      `q_chunk=k_chunk=512` 3.35 ms -- **6.54x measured** (close to the
      doc's 7.4x, same op family, same hardware), but this is a forward-only
      number on an op this port cannot use for training without first
      writing a correct backward for it (flash-attention-style, with proper
      chunked recomputation to stay memory-bound-safe -- materializing the
      full `[B,H,S,S]` attention matrix instead would very likely erase the
      speedup at ViT-L's real sequence lengths). Forward outputs agree:
      PCC(`ttml` fused fwd, `ttnn` tuned fwd) = 0.999626, same math.
    - Net implication: the *currently used* training kernel (`ttml` fused,
      unconfigurable) is itself ~3.7x slower on its forward pass alone than
      even `ttnn.transformer.scaled_dot_product_attention`'s *untuned*
      default, and ~24x slower than its tuned config -- but closing that gap
      for training needs a real backward for the tunable op, which is Tier
      2/3 kernel work (TENSTORRENT_PORT.md section 4, item 4, already
      flagged there as "the largest piece of work on this list... a stretch
      item"), not a program-config plumbing fix. No `program_config` value
      can be usefully registered here yet for the training path; this
      registry stays empty pending that Tier 2/3 op.

    **2026-07-28 follow-up**: a Tier-1 alternative was built and measured
    (`ops_tt/flash_attention.py::chunked_bidirectional_attention` --
    FlashAttention-2-style, composed entirely from existing `ttnn`
    primitives in a Python loop, no vendored edits). Numerically correct
    (passes all parity gates) but 3-10x *slower* than the fused kernel above
    at the real decoder shape, even with zero chunking (single-shot) --
    composing separate `ttnn` ops cannot beat one real fused kernel here.
    Also investigated and declined: patching the vendored
    `constexpr bool fp32_dest_acc_en = true` in
    `sdpa_fw_program_factory.cpp` (would only affect forward, ~46ms of the
    op's ~224ms; `sdpa_bw` has no equivalent chunk cap at all). Full detail
    and the measured numbers: `TENSTORRENT_PORT.md`'s M6 section. This
    registry remains empty; closing the gap still needs a real Tier-2/3
    fused Metal kernel, not reachable via config tuning or composition.
    """

    _table: dict[OpShapeKey, Optional["ttnn.SDPAProgramConfig"]] = field(default_factory=dict)

    def get(self, key: OpShapeKey) -> Optional["ttnn.SDPAProgramConfig"]:
        return self._table.get(key)

    def register(self, key: OpShapeKey, program_config: Optional["ttnn.SDPAProgramConfig"]) -> None:
        self._table[key] = program_config


@dataclass
class MatmulProgramConfigRegistry:
    """`{(B, S, D, H): ttnn matmul program config | None}`, same structure
    and same M6-placeholder status as `SDPAProgramConfigRegistry` -- see its
    docstring, including the "no existing `ttml` hook to thread this
    through yet" caveat, which applies identically here (`linear_op.hpp`/
    `matmul_op.hpp` accept no program config parameter). The concrete
    per-entry type is intentionally left as `object` rather than one fixed
    `ttnn.Matmul*ProgramConfig` class: `ttnn` ships several distinct matmul
    program config types (`MatmulMultiCoreReuseMultiCast1DProgramConfig`,
    `MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig`, etc,
    all present in this build), and which one is correct for a given
    `(B, S, D, H)` is exactly the M6 tuning question this registry defers.
    """

    _table: dict[OpShapeKey, Optional[object]] = field(default_factory=dict)

    def get(self, key: OpShapeKey) -> Optional[object]:
        return self._table.get(key)

    def register(self, key: OpShapeKey, program_config: Optional[object]) -> None:
        self._table[key] = program_config


def encoder_shape_key(contract: ShapeContract) -> OpShapeKey:
    """`(B, S, D, H)` for the encoder's SDPA/matmul ops, from a `ShapeContract`."""
    return OpShapeKey(
        batch=contract.batch,
        seq_len=contract.encoder_seq_len,
        embed_dim=contract.embed_dim,
        heads=contract.encoder_heads,
    )


def decoder_shape_key(contract: ShapeContract) -> OpShapeKey:
    """`(B, S, D, H)` for the decoder's SDPA/matmul ops, from a `ShapeContract`."""
    return OpShapeKey(
        batch=contract.batch,
        seq_len=contract.decoder_seq_len,
        embed_dim=contract.decoder_embed_dim,
        heads=contract.decoder_heads,
    )
