"""Tests for `config_tt.py`.

Split into two groups:
  - Pure-Python logic (`TTMAEConfig` validation/LR scaling,
    `build_param_group_specs`, the program-config registries) needs no
    device and runs even without `ttml` importable.
  - `MultiGroupAdamW`/`clip_grad_norm` integration needs real
    `ttml.autograd.Tensor` parameters and a device, so those tests are
    gated behind `pytest.importorskip("ttml")` and open/close a device,
    matching `ops_tt/test_attention.py`'s pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

from smri_mae_tt.config_tt import (
    MultiGroupAdamW,
    OpShapeKey,
    SDPAProgramConfigRegistry,
    TTMAEConfig,
    ParamGroupSpec,
    build_param_group_specs,
    clip_grad_norm,
    decoder_shape_key,
    encoder_shape_key,
)
from smri_mae_tt.layout import ShapeContract


def _contract(**overrides) -> ShapeContract:
    # class_token=False: with class_token=True, encoder_seq_len = l_vis + 1
    # and decoder_seq_len = l_vis + q_pred + 1 can never be tile-aligned
    # (l_vis/q_pred must themselves be tile-aligned per ShapeContract's own
    # __post_init__, and a multiple of 32 plus 1 is never itself a multiple
    # of 32) -- layout.py documents this openly as an unresolved M2 design
    # question ("where the CLS token enters the pipeline ... is an open M2
    # design question"), not something for this file to route around.
    defaults = dict(batch=2, l_vis=64, q_pred=64, mask_ratio=0.8, class_token=False)
    defaults.update(overrides)
    return ShapeContract(**defaults)


# ---------------------------------------------------------------------------
# TTMAEConfig
# ---------------------------------------------------------------------------


def test_ttmaeconfig_defaults_match_yaml():
    cfg = TTMAEConfig(shape_contract=_contract())
    assert cfg.pred_mask_ratio is None
    assert cfg.base_lr == 0.001
    assert cfg.min_lr == 1e-6
    assert cfg.warmup_epochs == 10
    assert cfg.weight_decay == 0.05
    assert cfg.betas == (0.9, 0.95)
    assert cfg.clip_grad == 1.0
    assert cfg.accum_iter == 1


def test_ttmaeconfig_lr_scaling_matches_reference_formula():
    cfg = TTMAEConfig(shape_contract=_contract(), base_lr=0.001)
    # main_pretrain.py:113: args.lr = args.base_lr * total_batch_size / 256
    assert cfg.lr_for_total_batch(256) == pytest.approx(0.001)
    assert cfg.lr_for_total_batch(60) == pytest.approx(0.001 * 60 / 256)
    with pytest.raises(ValueError):
        cfg.lr_for_total_batch(0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(betas=(0.9,)),
        dict(betas=(1.0, 0.95)),
        dict(warmup_epochs=-1),
        dict(accum_iter=0),
        dict(clip_grad=0.0),
        dict(pred_mask_ratio=1.0),
    ],
)
def test_ttmaeconfig_rejects_invalid_fields(kwargs):
    with pytest.raises(ValueError):
        TTMAEConfig(shape_contract=_contract(), **kwargs)


# ---------------------------------------------------------------------------
# build_param_group_specs
# ---------------------------------------------------------------------------


class _Param:
    def __init__(self, requires_grad: bool = True):
        self._requires_grad = requires_grad

    def get_requires_grad(self) -> bool:
        return self._requires_grad


def test_build_param_group_specs_matches_utils_get_param_groups_rules():
    named_parameters = {
        "encoder.patch_embed.weight": _Param(),
        "encoder.patch_embed.bias": _Param(),
        "encoder.block0.norm1.weight": _Param(),
        "encoder.block0.attn.qkv.weight": _Param(),
        "encoder.block0.attn.qkv.bias": _Param(),
        "decoder.pos_embed.gamma": _Param(),
        "frozen.thing": _Param(requires_grad=False),
    }
    specs = build_param_group_specs(named_parameters, patch_embed_lr_mult=0.1)

    by_key = {(s.lr_multiplier, s.wd_multiplier): set(s.param_names) for s in specs}

    # patch_embed.weight: lr *= 0.1 (patch_embed), wd stays 1.0 (not bias/norm/gamma)
    assert by_key[(0.1, 1.0)] == {"encoder.patch_embed.weight"}
    # patch_embed.bias: lr *= 0.1 (patch_embed) AND wd = 0.0 (.bias)
    assert by_key[(0.1, 0.0)] == {"encoder.patch_embed.bias"}
    # norm.weight and .gamma: wd = 0.0, lr untouched (not patch_embed)
    assert by_key[(1.0, 0.0)] == {"encoder.block0.norm1.weight", "decoder.pos_embed.gamma", "encoder.block0.attn.qkv.bias"}
    # plain weight: full lr, full wd
    assert by_key[(1.0, 1.0)] == {"encoder.block0.attn.qkv.weight"}

    # frozen param excluded entirely
    all_named = {n for s in specs for n in s.param_names}
    assert "frozen.thing" not in all_named


def test_build_param_group_specs_default_patch_embed_lr_mult_is_noop():
    named_parameters = {"a.weight": _Param(), "a.patch_embed.weight": _Param()}
    specs = build_param_group_specs(named_parameters)  # default patch_embed_lr_mult=1.0
    assert len(specs) == 1
    assert specs[0].lr_multiplier == 1.0
    assert specs[0].wd_multiplier == 1.0
    assert set(specs[0].param_names) == {"a.weight", "a.patch_embed.weight"}


def test_build_param_group_specs_accepts_plain_objects_without_get_requires_grad():
    # No get_requires_grad attribute at all -- treated as trainable.
    named_parameters = {"x.weight": object(), "x.bias": object()}
    specs = build_param_group_specs(named_parameters)
    total = sum(len(s.param_names) for s in specs)
    assert total == 2


# ---------------------------------------------------------------------------
# Program config registries
# ---------------------------------------------------------------------------


def test_sdpa_program_config_registry_defaults_to_none():
    registry = SDPAProgramConfigRegistry()
    key = OpShapeKey(batch=8, seq_len=5632, embed_dim=512, heads=16)
    assert registry.get(key) is None
    registry.register(key, "sentinel")
    assert registry.get(key) == "sentinel"
    other_key = OpShapeKey(batch=8, seq_len=1152, embed_dim=1024, heads=16)
    assert registry.get(other_key) is None


def test_encoder_decoder_shape_keys_from_contract():
    contract = _contract(batch=8, l_vis=1152, q_pred=4096)
    enc = encoder_shape_key(contract)
    dec = decoder_shape_key(contract)
    assert enc == OpShapeKey(batch=8, seq_len=contract.encoder_seq_len, embed_dim=1024, heads=16)
    assert dec == OpShapeKey(batch=8, seq_len=contract.decoder_seq_len, embed_dim=512, heads=16)
    assert enc.seq_len == 1152
    assert dec.seq_len == 1152 + 4096


# ---------------------------------------------------------------------------
# Device integration: MultiGroupAdamW + clip_grad_norm
# ---------------------------------------------------------------------------

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402


@pytest.fixture(autouse=False)
def _device():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        yield ctx
    finally:
        ctx.close_device()


def _make_param(value: float, requires_grad: bool = True, dtype=ttnn.DataType.BFLOAT16) -> "ttml.autograd.Tensor":
    data = np.full((1, 1, 32, 32), value, dtype=np.float32)
    t = ttml.autograd.Tensor.from_numpy(data, layout=ttnn.Layout.TILE, new_type=dtype)
    t.set_requires_grad(requires_grad)
    return t


def _set_constant_grad(t: "ttml.autograd.Tensor", value: float) -> None:
    grad = np.full((1, 1, 32, 32), value, dtype=np.float32)
    grad_t = ttml.autograd.Tensor.from_numpy(grad, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.BFLOAT16)
    t.add_grad(grad_t.get_value())


def test_plain_adamw_silently_noops_on_fp32_parameter_regression_guard(_device):
    """Regression guard for the finding documented in `MultiGroupAdamW`'s
    docstring: `ttml.optimizers.AdamW.step()` updates a parameter in place
    by fetching `get_value(PreferredPrecision.HALF)`; for an fp32-stored
    parameter that returns a throwaway bf16 cast, so the real parameter is
    silently left unchanged. This is exactly why `MultiGroupAdamW` is built
    on `AdamWFullPrecision` instead. If a future `ttml` fixes this (e.g. by
    writing back regardless of dtype), this test starts failing at the
    first assertion, which is the signal to reconsider using plain `AdamW`.
    """
    fp32_param = _make_param(1.0, dtype=ttnn.DataType.FLOAT32)
    _set_constant_grad(fp32_param, 0.1)
    cfg = ttml.optimizers.AdamWConfig.make(lr=0.5, beta1=0.9, beta2=0.95, epsilon=1e-8, weight_decay=0.5)
    opt = ttml.optimizers.AdamW({"p": fp32_param}, cfg)
    opt.step()
    after = float(np.asarray(fp32_param.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])
    assert after == pytest.approx(1.0), (
        f"expected plain AdamW.step() to silently no-op on an fp32 parameter (got {after}); "
        "if this now changes the value, AdamWFullPrecision may no longer be required here"
    )

    bf16_param = _make_param(1.0, dtype=ttnn.DataType.BFLOAT16)
    _set_constant_grad(bf16_param, 0.1)
    opt2 = ttml.optimizers.AdamW({"p": bf16_param}, cfg)
    opt2.step()
    after_bf16 = float(np.asarray(bf16_param.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])
    assert after_bf16 != pytest.approx(1.0), "bf16-stored parameter should actually update in place"


def test_multi_group_adamw_applies_independent_weight_decay_per_group(_device):
    """Two params, identical initial value and identical gradient, one in
    a wd_multiplier=1.0 group and one in a wd_multiplier=0.0 group. Both
    should move in the gradient's direction, but the decayed one should
    move further (decoupled weight decay adds an extra `-lr*wd*theta`
    pull in the same direction here, since the initial value is positive
    and the gradient is also positive).
    """
    weight = _make_param(1.0)
    bias = _make_param(1.0)
    _set_constant_grad(weight, 0.1)
    _set_constant_grad(bias, 0.1)

    named_parameters = {"layer.weight": weight, "layer.bias": bias}
    specs = build_param_group_specs(named_parameters)
    assert {(s.lr_multiplier, s.wd_multiplier) for s in specs} == {(1.0, 1.0), (1.0, 0.0)}

    opt = MultiGroupAdamW(
        named_parameters,
        specs,
        base_lr=0.1,
        weight_decay=0.5,
        betas=(0.9, 0.95),
    )
    opt.step()

    weight_after = float(np.asarray(weight.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])
    bias_after = float(np.asarray(bias.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])

    assert weight_after < 1.0, "weight (grad>0) should decrease after an AdamW step"
    assert bias_after < 1.0, "bias (grad>0) should decrease after an AdamW step"
    assert weight_after < bias_after, (
        f"weight (wd_multiplier=1.0) should shrink more than bias (wd_multiplier=0.0): "
        f"weight_after={weight_after}, bias_after={bias_after}"
    )


def test_multi_group_adamw_set_lr_updates_each_group_by_its_multiplier():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.open_device()
    try:
        weight = _make_param(1.0)
        bias = _make_param(1.0)
        _set_constant_grad(weight, 0.1)
        _set_constant_grad(bias, 0.1)
        named_parameters = {"layer.weight": weight, "layer.bias": bias}
        specs = build_param_group_specs(named_parameters)

        opt = MultiGroupAdamW(named_parameters, specs, base_lr=0.1, weight_decay=0.5, betas=(0.9, 0.95))
        opt.set_lr(0.2)
        for sub_opt in opt._optimizers:  # white-box: confirm the live update actually took
            assert sub_opt.get_lr() == pytest.approx(0.2)
    finally:
        ctx.close_device()


def test_multi_group_adamw_rejects_param_referenced_by_no_group(_device):
    weight = _make_param(1.0)
    _set_constant_grad(weight, 0.1)
    named_parameters = {"layer.weight": weight}
    # Deliberately mis-specified group referencing a name absent from named_parameters.
    bad_specs = [ParamGroupSpec(lr_multiplier=1.0, wd_multiplier=1.0, param_names=("layer.weight", "layer.missing"))]
    with pytest.raises(KeyError):
        MultiGroupAdamW(named_parameters, bad_specs, base_lr=0.1, weight_decay=0.0, betas=(0.9, 0.95))


def test_clip_grad_norm_shrinks_large_gradients(_device):
    weight = _make_param(1.0)
    bias = _make_param(1.0)
    _set_constant_grad(weight, 100.0)
    _set_constant_grad(bias, 100.0)
    named_parameters = {"layer.weight": weight, "layer.bias": bias}

    total_norm_before = clip_grad_norm(named_parameters, max_norm=1.0)
    total_norm_value = float(np.asarray(total_norm_before.to_numpy(ttnn.DataType.FLOAT32)).reshape(-1)[0])
    assert total_norm_value > 1.0, "synthetic gradient of 100.0 everywhere should have a large pre-clip norm"

    weight_grad = np.asarray(weight.get_grad_tensor().to_numpy(ttnn.DataType.FLOAT32), dtype=np.float64)
    bias_grad = np.asarray(bias.get_grad_tensor().to_numpy(ttnn.DataType.FLOAT32), dtype=np.float64)
    post_clip_norm = float(np.sqrt((weight_grad**2).sum() + (bias_grad**2).sum()))
    assert post_clip_norm <= 1.0 + 1e-1, f"post-clip combined grad norm should be <= max_norm=1.0, got {post_clip_norm}"
