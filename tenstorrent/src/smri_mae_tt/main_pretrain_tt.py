"""Training loop for the TT-NN MAE port (TENSTORRENT_PORT.md section 7, M5:
"AdamW, clipping, accumulation, schedule, checkpointing. Gate: 200 steps on
a shard subset, loss curve tracking the 4090 baseline's first 200 steps
within noise.").

This file wires together every M0-M4 piece that already exists and is
tested (`layout.py`, `packing.py`, `model_tt.py`, `params.py`, `config_tt.py`,
`mesh.py`) into an actual optimizer loop -- it is glue, not new model code,
with exactly one exception documented in detail below.

## Data: real shards only, no synthetic fallback

This module has no synthetic data path. `train_batch_fn` is a required
argument -- there is no default that fabricates fake MRI content.
`real_data.py` wraps `data.mri_data.make_sparse_wds_dataset`/`collate`
(unmodified) + `packing.pack_batch` (unmodified) over the local
`datasets/FOMO_with_dwi/shard.*.tar` files (`real_data.py` hardcodes 50
train shard names / 7500 samples and 17 val shard names / 2550 samples; on
any given host only the ones actually downloaded from R2 so far are
present -- `RealTrainSource`/`RealValSource` fail fast with
`FileNotFoundError` if a `urls` set they're given isn't fully on disk, see
`real_data.py`'s module docstring for exactly which files that is on this
host at time of writing). `main()` always wires
`real_data.RealTrainSource`/`RealValSource` into `train()`'s
`train_batch_fn`/`eval_batch_iter_fn` hooks. See `real_data.py`'s module
docstring for the train-side infinite-resampled-cycling policy and the
val-side one-full-pass-per-eval policy.

(Earlier revisions of this file had a synthetic fallback --
`_synthetic_sparse_batch`/`random_packed_batch`, fabricating dense MRI-shaped
noise via `rng.standard_normal` purely so the training loop had *something*
to run against before real shards were downloaded. Once real data was
available, profiling found that synthetic-batch construction -- a dense
`(208,240,208)` `rng.standard_normal` draw per sample, 24x/step at real
batch scale -- was consuming the majority of measured step wall-clock time
on its own, a pure host-CPU benchmark artifact with nothing to do with
device compute. It was removed rather than optimized: real shards were
already available, and a synthetic path that exists only to be fast is not
a goal here.)

Once real data replaced the synthetic path, re-profiling
(`tenstorrent/docs/m6-optimization-results.md`) found real `data_pack`
(WebDataset tar decode + `collate` + `packing.pack_batch`) still costs
~757ms/step (27.8%) at full ViT-L scale -- genuine host-CPU work, but with
no device dependency, so `main()` wires a `train_batch_source_factory`
and `train()` defaults to `prefetch=True`
(`real_data.ProcessPrefetchingBatchSource`): a background **process**
prepares step N+1's batch while step N's async-dispatched device compute is
in flight, instead of running data_pack serially before every step.
(A thread-based version was tried first and measured ~0% real overlap --
GIL contention from the sheer number of small Python-level `ttml`/`ttnn`
calls per step; see that class's docstring.) Measured net effect: real
`next_batch()` wait time drops from ~750-800ms/step to ~105-110ms/step
(pickling + IPC across the process boundary), for a ~600-700ms/step
(~25%) reduction in total step time at full ViT-L scale. Pass
`--no-prefetch` (`main()`) or `prefetch=False` (`train()`) to isolate the
non-overlapped baseline for comparison.

## Decoder -> loss autograd: now shared with `model_tt.py`, not duplicated here

`model_tt.MaskedAutoencoderViTTT.forward`/`forward_loss` used to be unusable
for training: closing the `[B,1,Q_pred,patch_dim]` (Decoder's raw output
rank) -> `[B,Q_pred,patch_dim]` (loss_tt's required rank) gap went through a
*host round trip* (fresh leaf tensor, no autograd parents), so
`loss.backward()` could not propagate gradients past that seam into
`Decoder`'s (or, transitively, `Encoder`'s) parameters.

This module originally worked around that at its own call site with a
hand-written Tier-1 `ttml.autograd.Function`, `_SqueezeDecoderPredAxis`
(TENSTORRENT_PORT.md section 4: "Tier 1 -- compose in Python on
`ttml.autograd.Function`... No build step, full Python debuggability"):

    forward:  ttnn.reshape(pred.get_value(), [B, Q_pred, patch_dim])
    backward: ttnn.reshape(grad_output, [B, 1, Q_pred, patch_dim])

deliberately *not* a call to `ttml.ops.reshape.reshape` (whose Python-level
backward `tt-train/tests/python/test_reshape_op.py::test_reshape_multiple_reshapes`
documents as unverified for rank-reducing below-4D reshapes -- exactly this
transition). **Empirically verified on this host's real Blackhole hardware**
before this training loop was trusted: built the tiny
`test_modules_tt._contract()`-scale encoder+decoder, ran `forward_encoder` ->
`forward_decoder` -> `_SqueezeDecoderPredAxis` ->
`per_scan_valid_voxel_mse_loss` -> `.backward(False)`, and confirmed
`is_grad_initialized()` is `True` with **100% nonzero, finite-magnitude**
gradients on the decoder head weight, a decoder block's QKV weight, *and* an
encoder block's QKV weight.

`model_tt.py` has since adopted this exact fix directly (its own
`_SqueezeDecoderPredAxis`/`squeeze_decoder_pred`, wired into its own
`forward()`/`forward_loss()` -- see `model_tt.py`'s module docstring), so
this module now **imports** `squeeze_decoder_pred` from `model_tt` instead of
defining its own copy, keeping one implementation shared by both call sites.
`forward_and_loss()` below still calls `model.forward_encoder()` +
`model.forward_decoder()` directly rather than `model.forward()`/
`model.forward_loss()`, since this training loop's accumulation/logging
structure wants the intermediate `pred` tensor and the loss separately; that
remains a legitimate reason to compose the pieces here rather than a
workaround for a broken seam.

## Numerics / recipe fidelity vs. deliberate smoke-test simplifications

- AdamW betas=(0.9, 0.95), weight_decay=0.05, no-decay-on-1D-params/bias,
  gradient clipping at 1.0: all taken verbatim from `config_tt.TTMAEConfig`'s
  defaults (already mirroring `default_pretrain.yaml`) and
  `config_tt.build_param_group_specs`/`MultiGroupAdamW`/`clip_grad_norm` --
  used as-is, not reimplemented.
- LR scaling rule `base_lr * total_batch_size / 256`
  (`TTMAEConfig.lr_for_total_batch`, matching `main_pretrain.py:113`): used
  as-is.
- **Deviation, documented per task instructions**: the reference recipe's
  `WarmupThenCosine` (`utils.py:391-421`) schedules over *epochs* (
  `warmup_epochs` x `steps_per_epoch`, where `steps_per_epoch` depends on
  real dataset length). This smoke-test script has no epoch/dataset-length
  concept (synthetic data, fixed step count), so `warmup_cosine_lr()` below
  reimplements the same linear-warmup-then-cosine-to-`min_lr` *shape*
  directly over `(step, total_steps, warmup_steps)`, with `warmup_steps` a
  CLI parameter instead of derived from `warmup_epochs`. This is a
  step-counting simplification, not a numerical algorithm change -- the
  schedule shape (linear warmup, cosine decay to `min_lr`) is identical.
- Gradient accumulation (`accum_iter`) is implemented (mirrors
  `main_pretrain.py:325-361`'s zero_grad-at-window-start /
  scale-loss-by-group-size / step-at-window-end structure, and
  `tt-train/sources/examples/nano_gpt/nanogpt_primitives_example.py`'s
  `reset_graph()`-immediately-after-`backward()` idiom), but defaults to
  `accum_iter=1` (the yaml's own default; `accum_iter=5` was the 4090
  baseline *run's* override, per `config_tt.TTMAEConfig`'s own docstring,
  not the recipe default) since a single-device smoke test has no
  communication cost to amortize.
- Model dims: far smaller than real ViT-L (302.8M encoder / 13.4M decoder
  params) -- per task instructions, this is expected and documented, not a
  numerical deviation to apologize for. `--preset smoke`/`--preset gate`
  (see `main()`) pick, respectively, a `pytest`-fast tiny config (matching
  `test_modules_tt._contract()`'s scale) and a moderately larger config
  (still nowhere near ViT-L) used for this task's standalone 200-step M5
  gate run.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import ttml
import ttnn

from smri_mae_tt import checkpoint_tt, real_data, wandb_tt
from smri_mae_tt.config_tt import (
    MultiGroupAdamW,
    TTMAEConfig,
    build_param_group_specs,
    clip_grad_norm,
)
from smri_mae_tt.layout import ShapeContract
from smri_mae_tt.loss_tt import (
    per_scan_valid_voxel_mse_loss,
    pred_targets_to_tensor,
    voxel_mask_to_tensor,
)
from smri_mae_tt.mesh import MeshContext
from smri_mae_tt.model_tt import MaskedAutoencoderViTTT, ModelParams, squeeze_decoder_pred
from smri_mae_tt.packing import PackedBatch
from smri_mae_tt.params import init_tt_params, named_tt_parameters

# tenstorrent/src/smri_mae_tt/main_pretrain_tt.py -> parents[2] == tenstorrent/
_TENSTORRENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_CHECKPOINT_ROOT = os.path.join(_TENSTORRENT_ROOT, "checkpoints")

# ---------------------------------------------------------------------------
# Forward + loss, using model_tt.squeeze_decoder_pred (module docstring:
# "Decoder -> loss autograd: now shared with model_tt.py, not duplicated
# here") in place of model_tt.MaskedAutoencoderViTTT.forward/forward_loss.
# ---------------------------------------------------------------------------


def _validate_packed_batch(contract: ShapeContract, batch: PackedBatch) -> int:
    """Fail-fast batch-shape validation, mirroring
    `model_tt.MaskedAutoencoderViTTT.forward`'s own cross-array batch-size
    check plus `forward_loss`'s per-array shape checks -- reimplemented here
    (not called from `model_tt.py`) because this module does not use
    `model_tt.forward`/`forward_loss` at all (see module docstring). Returns
    the validated batch size.
    """
    shapes = {
        "visible_values": tuple(batch.visible_values.shape),
        "visible_ids": tuple(batch.visible_ids.shape),
        "pred_ids": tuple(batch.pred_ids.shape),
        "pred_targets": tuple(batch.pred_targets.shape),
        "pred_voxel_mask": tuple(batch.pred_voxel_mask.shape),
    }
    batch_sizes = {name: shape[0] for name, shape in shapes.items()}
    if len(set(batch_sizes.values())) != 1:
        raise ValueError(f"inconsistent batch size across PackedBatch fields: {batch_sizes}")
    b = next(iter(batch_sizes.values()))
    expected = {
        "visible_values": (b, contract.l_vis, contract.patch_dim),
        "visible_ids": (b, contract.l_vis),
        "pred_ids": (b, contract.q_pred),
        "pred_targets": (b, contract.q_pred, contract.patch_dim),
        "pred_voxel_mask": (b, contract.q_pred, contract.patch_dim),
    }
    for name, shape in shapes.items():
        if shape != expected[name]:
            raise ValueError(f"PackedBatch.{name} shape {shape} != expected {expected[name]}")
    return b


def forward_and_loss(model: MaskedAutoencoderViTTT, batch: PackedBatch) -> "ttml.autograd.Tensor":
    """Encode -> decode -> (autograd-connected squeeze) -> loss, the
    training-loop replacement for `model.forward()` (module docstring).
    Returns the `[1, 1]` scalar loss tensor, gradient-connected all the way
    back to every `Encoder`/`Decoder` parameter.
    """
    contract = model.contract
    b = _validate_packed_batch(contract, batch)
    _cls_embed, patch_embeds = model.forward_encoder(batch.visible_values, batch.visible_ids)
    pred = model.forward_decoder(patch_embeds, batch.visible_ids, batch.pred_ids)
    pred3 = squeeze_decoder_pred(pred, contract, b)
    targets_t = pred_targets_to_tensor(batch.pred_targets)
    mask_t = voxel_mask_to_tensor(batch.pred_voxel_mask)
    return per_scan_valid_voxel_mse_loss(pred3, targets_t, mask_t)


def _tensor_to_scalar(tensor: "ttml.autograd.Tensor") -> float:
    arr = np.asarray(tensor.to_numpy(ttnn.DataType.FLOAT32), dtype=np.float32).reshape(-1)
    return float(arr[0])


# ---------------------------------------------------------------------------
# LR schedule (deliberate step-based simplification of WarmupThenCosine -- see
# module docstring)
# ---------------------------------------------------------------------------


def warmup_cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError(f"warmup_steps must be in [0, total_steps], got {warmup_steps} (total_steps={total_steps})")
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    denom = max(1, total_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# The training loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepLog:
    step: int
    loss: float
    lr: float
    grad_norm: Optional[float]
    replaced_samples: int
    val_loss: Optional[float] = None


def run_eval(
    model: MaskedAutoencoderViTTT,
    eval_batch_iter_fn: Callable[[np.random.Generator], Iterator[PackedBatch]],
    rng: np.random.Generator,
    auto_ctx: "ttml.autograd.AutoContext",
) -> tuple[float, int]:
    """One full no-grad pass over `eval_batch_iter_fn(rng)`'s batches:
    forward + loss only (no `.backward()`), `auto_ctx.reset_graph()` after
    every batch (mirroring the train loop's own reset-after-backward
    pattern, just with nothing to back-propagate through here) so the
    autograd graph does not grow across eval batches. Returns
    `(mean_loss, num_batches)` -- an unweighted mean over batches (every
    batch has the same fixed `contract.batch` size by construction, so this
    is also the per-sample mean). Raises if the iterator yields zero
    batches or any batch's loss is non-finite -- never silently skipped.
    """
    losses: list[float] = []
    for batch in eval_batch_iter_fn(rng):
        loss = forward_and_loss(model, batch)
        loss_value = _tensor_to_scalar(loss)
        auto_ctx.reset_graph()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"non-finite eval loss in batch {len(losses)}: {loss_value}")
        losses.append(loss_value)
    if not losses:
        raise RuntimeError("run_eval got zero batches from eval_batch_iter_fn -- nothing to evaluate")
    return float(sum(losses) / len(losses)), len(losses)


def train(
    contract: ShapeContract,
    encoder_depth: int,
    decoder_depth: int,
    cfg: TTMAEConfig,
    num_steps: int,
    *,
    mlp_ratio: float = 4.0,
    warmup_steps: int = 0,
    accum_iter: int = 1,
    patch_embed_lr_mult: float = 1.0,
    seed: int = 0,
    log_every: int = 1,
    print_log: bool = True,
    train_batch_fn: Callable[[np.random.Generator], PackedBatch],
    epoch_equiv_fn: Optional[Callable[[], float]] = None,
    eval_every: int = 0,
    eval_batch_iter_fn: Optional[Callable[[np.random.Generator], Iterator[PackedBatch]]] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 0,
    checkpoint_at_end: bool = False,
    resume_checkpoint: Optional[str] = None,
    wandb_run: object = None,
    prefetch: bool = True,
    train_batch_source_factory: Optional[Callable[[], object]] = None,
    on_step_timing: Optional[Callable[[int, float], None]] = None,
    on_stage_timing: Optional[Callable[[int, str, float], None]] = None,
    use_tuned_gelu: bool = True,
) -> list[StepLog]:
    """Run `num_steps` optimizer steps of `MaskedAutoencoderViTTT` on a
    single-device `MeshContext`, per TENSTORRENT_PORT.md section 7's M5
    gate. See module docstring for the data source (real shards only, no
    synthetic fallback, through the real `packing.pack_batch`) and the
    decoder->loss autograd fix this loop depends on (`forward_and_loss`).

    `train_batch_fn` is required -- there is no synthetic default. Pass
    `real_data.RealTrainSource(contract).next_batch` (`main()` does exactly
    this).

    `train_batch_source_factory` (optional): a zero-arg callable that
    *constructs* a fresh `next_batch(rng)`-shaped source (e.g.
    `functools.partial(real_data.RealTrainSource, contract)`), used only
    when `prefetch=True` to run `real_data.ProcessPrefetchingBatchSource` --
    see that class's docstring for why this needs a *factory* rather than
    `train_batch_fn` itself (a bound method with live file handles can't
    safely cross a `multiprocessing` process boundary; the child must build
    its own source). When `None` (most callers -- tests, smoke/gate
    presets), `prefetch` has no effect and `train_batch_fn` is called
    directly every step, exactly as before this parameter existed.

    `eval_every > 0` requires `eval_batch_iter_fn` (e.g.
    `real_data.RealValSource(contract).eval_batches`): every `eval_every`
    steps (and always on the final step), a no-grad pass over
    `eval_batch_iter_fn`'s batches (`run_eval`) reports a val loss distinct
    from the train loss, attached to that step's `StepLog.val_loss`.

    `checkpoint_dir` + (`checkpoint_every > 0` and/or `checkpoint_at_end`)
    periodically writes `checkpoint_tt.save_checkpoint(model.encoder,
    model.decoder, step, checkpoint_dir)`. `resume_checkpoint`, if given, is
    loaded (`checkpoint_tt.load_checkpoint`, weights only -- see that
    module's docstring on optimizer state not being saved/restored) right
    after the model is constructed, before the first training step.

    `wandb_run`, if not `None`, gets per-step train metrics
    (`wandb_tt.log_train_step`) and per-eval val metrics
    (`wandb_tt.log_eval`) logged to it -- constructing/finishing the run
    itself is the caller's job (`main()`'s `--wandb` path), not this
    function's, so this stays usable without any wandb/network dependency
    when `wandb_run=None` (the default, and the only thing the fast pytest
    smoke tests ever pass).

    `on_step_timing`, if given, is called once per step as
    `on_step_timing(step, elapsed_seconds)` with the wall-clock cost of that
    step's core work (batch acquisition through optimizer step -- excludes
    eval/checkpoint/logging side effects, which are not part of steady-state
    per-step cost). This is the hook `scripts/bench_step.py` uses for A/B
    step-time comparisons against any code-path change (data loading,
    kernel/op swaps, dtype changes, ...), so it stays a generic
    `(int, float) -> None` callback rather than anything tied to a specific
    change being measured.

    `on_stage_timing`, if given, is called several times per step as
    `on_stage_timing(step, stage_name, elapsed_seconds)` for
    `stage_name in ("data", "forward", "backward", "grad_clip",
    "optimizer_step")`, each bounded by an explicit `ttnn.synchronize_device`
    so the reported time reflects actual device completion of that stage
    (not just host-side dispatch racing ahead). This adds real sync
    overhead, so it's opt-in and meant for one-off profiling
    (`scripts/profile_step_stages.py`), not left on during normal training.
    Stages before `should_step` is true (`grad_clip`/`optimizer_step`) are
    only reported on accumulation-boundary steps, same as the work itself.

    `use_tuned_gelu` (M6, default `True`): forwarded to `MaskedAutoencoderViTTT`
    -- see `modules_tt.Mlp`'s docstring. `ops_tt.tuned_gelu` (`ttnn.gelu`'s
    `FastLut` variant) is PCC-verified at 0.99998 against a PyTorch reference
    (`OP_FWD_PCC_GATE`=0.999) and gives a real, reproducible ~0.5-0.9%
    end-to-end step-time reduction at real production scale
    (`tenstorrent/docs/m6-optimization-results.md`, "Non-SDPA/matmul op
    survey" section) -- defaulted on since it's a verified win with no
    vendored-kernel risk. Pass `False` to recover the exact prior GELU math
    (e.g. to reproduce results captured before this default changed).

    (M6 note: `ops_tt.tuned_linear.TunedLinearLayer` -- a per-shape tuned
    matmul `program_config` cache, ~2% e2e win -- and two vendored SDPA
    backward-KV kernel micro-optimizations (Q/dO group-cache reader,
    7->6-cycle fused kernel; ~1.3%/~0.35% e2e) were tried, shipped, then
    reverted 2026-08-05 in favor of stability: real but small gains, bespoke
    to this project, not upstream-contributable, and (for the two vendored
    kernel changes) touching the exact kernel family responsible for every
    device hang found during this port's optimization work. See
    `TENSTORRENT_PORT.md` section 5 for the full history. `use_tuned_gelu`
    above and the B17 `sfpu_reduce` fusion in the vendored `sdpa_bw` kernel
    were kept: verified wins/precision improvements built on documented
    public APIs, no hang history.)
    """
    if accum_iter < 1:
        raise ValueError(f"accum_iter must be >= 1, got {accum_iter}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    if eval_every > 0 and eval_batch_iter_fn is None:
        raise ValueError("eval_every > 0 requires eval_batch_iter_fn")
    if (checkpoint_every > 0 or checkpoint_at_end) and checkpoint_dir is None:
        raise ValueError("checkpoint_every > 0 / checkpoint_at_end requires checkpoint_dir")

    history: list[StepLog] = []

    with MeshContext() as mesh:
        param_rng = np.random.default_rng(seed)
        encoder_params, decoder_params = init_tt_params(
            param_rng, contract, encoder_depth, decoder_depth, mlp_ratio=mlp_ratio
        )
        model = MaskedAutoencoderViTTT(
            contract,
            encoder_depth,
            decoder_depth,
            ModelParams(encoder=encoder_params, decoder=decoder_params),
            mlp_ratio=mlp_ratio,
            use_tuned_gelu=use_tuned_gelu,
        )

        if resume_checkpoint is not None:
            payload = checkpoint_tt.load_checkpoint(resume_checkpoint, model.encoder, model.decoder)
            if print_log:
                print(f"resumed weights from {resume_checkpoint} (checkpoint step={payload.get('step')})")

        named_params = named_tt_parameters(model.encoder, model.decoder)
        group_specs = build_param_group_specs(named_params, patch_embed_lr_mult=patch_embed_lr_mult)

        total_batch_size = contract.batch * accum_iter  # world_size=1 (single device, section 5)
        base_lr = cfg.lr_for_total_batch(total_batch_size)

        optimizer = MultiGroupAdamW(
            named_params,
            group_specs,
            base_lr=base_lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            epsilon=cfg.adam_epsilon,
            amsgrad=cfg.adam_amsgrad,
        )

        data_rng = np.random.default_rng(seed + 1)
        eval_rng = np.random.default_rng(seed + 2)
        auto_ctx = ttml.autograd.AutoContext.get_instance()

        # `prefetch` + `train_batch_source_factory`: a background *process* (see
        # real_data.ProcessPrefetchingBatchSource's docstring for why a process, not a
        # thread -- threads measured ~0% real overlap here, GIL-starved by the sheer
        # number of small Python-level ttml/ttnn calls this loop makes per step) builds
        # step N+1's batch while step N's device compute is in flight, instead of running
        # data_pack serially before every step. Only active when a factory is given (most
        # callers -- tests, smoke/gate presets -- don't pass one and run exactly as
        # before). Call order (and therefore data_rng consumption order) is unchanged --
        # see that class's docstring for why this keeps golden-baseline/loss-trend
        # determinism intact even though the *process* running the calls differs.
        prefetch_source = (
            real_data.ProcessPrefetchingBatchSource(train_batch_source_factory, seed + 1, num_steps)
            if prefetch and train_batch_source_factory is not None
            else None
        )
        get_batch = prefetch_source.next_batch if prefetch_source is not None else train_batch_fn
        stage_device = auto_ctx.get_device() if on_stage_timing is not None else None

        try:
            for step in range(num_steps):
                step_start = time.monotonic() if on_step_timing is not None else None

                lr = warmup_cosine_lr(step, num_steps, warmup_steps, base_lr, cfg.min_lr)
                optimizer.set_lr(lr)

                step_in_accum = step % accum_iter
                if step_in_accum == 0:
                    optimizer.zero_grad()

                stage_t = time.monotonic() if on_stage_timing is not None else None

                batch = get_batch(data_rng)
                if on_stage_timing is not None:
                    ttnn.synchronize_device(stage_device)
                    now = time.monotonic()
                    on_stage_timing(step, "data", now - stage_t)
                    stage_t = now

                loss = forward_and_loss(model, batch)
                loss_value = _tensor_to_scalar(loss)  # forces device completion of the forward pass
                if not math.isfinite(loss_value):
                    raise RuntimeError(f"non-finite loss at step {step}: {loss_value}")
                if on_stage_timing is not None:
                    now = time.monotonic()
                    on_stage_timing(step, "forward", now - stage_t)
                    stage_t = now

                loss_scaled = loss if accum_iter == 1 else ttml.ops.binary.mul(loss, 1.0 / accum_iter)
                loss_scaled.backward(False)
                if on_stage_timing is not None:
                    ttnn.synchronize_device(stage_device)
                    now = time.monotonic()
                    on_stage_timing(step, "backward", now - stage_t)
                    stage_t = now

                auto_ctx.reset_graph()
                if on_stage_timing is not None:
                    ttnn.synchronize_device(stage_device)
                    now = time.monotonic()
                    on_stage_timing(step, "reset_graph", now - stage_t)
                    stage_t = now

                should_step = step_in_accum == accum_iter - 1
                grad_norm_value: Optional[float] = None
                if should_step:
                    grad_norm_tensor = clip_grad_norm(named_params, cfg.clip_grad)
                    grad_norm_value = _tensor_to_scalar(grad_norm_tensor)  # forces device completion of grad clip
                    if not math.isfinite(grad_norm_value):
                        raise RuntimeError(f"non-finite grad norm at step {step}: {grad_norm_value}")
                    if on_stage_timing is not None:
                        now = time.monotonic()
                        on_stage_timing(step, "grad_clip", now - stage_t)
                        stage_t = now

                    optimizer.step()
                    if on_stage_timing is not None:
                        ttnn.synchronize_device(stage_device)
                        now = time.monotonic()
                        on_stage_timing(step, "optimizer_step", now - stage_t)
                        stage_t = now

                if on_step_timing is not None:
                    on_step_timing(step, time.monotonic() - step_start)

                is_last_step = step == num_steps - 1
                is_log_step = step % log_every == 0 or is_last_step

                val_loss_value: Optional[float] = None
                val_num_batches: Optional[int] = None
                if eval_every > 0 and ((step + 1) % eval_every == 0 or is_last_step):
                    val_loss_value, val_num_batches = run_eval(model, eval_batch_iter_fn, eval_rng, auto_ctx)
                    if print_log:
                        print(f"  eval  @ step {step:5d} | val_loss {val_loss_value:10.6f} over {val_num_batches} batches")
                    wandb_tt.log_eval(wandb_run, step, val_loss=val_loss_value, num_batches=val_num_batches)

                if is_log_step or val_loss_value is not None:
                    entry = StepLog(
                        step=step,
                        loss=loss_value,
                        lr=lr,
                        grad_norm=grad_norm_value,
                        replaced_samples=batch.replaced_samples,
                        val_loss=val_loss_value,
                    )
                    history.append(entry)
                    if print_log and is_log_step:
                        gn = f"{grad_norm_value:.4f}" if grad_norm_value is not None else "-"
                        print(f"step {step:5d} | loss {loss_value:10.6f} | lr {lr:.3e} | grad_norm {gn}")
                    wandb_tt.log_train_step(
                        wandb_run,
                        step,
                        loss=loss_value,
                        lr=lr,
                        grad_norm=grad_norm_value,
                        replaced_samples=batch.replaced_samples,
                        epoch_equiv=epoch_equiv_fn() if epoch_equiv_fn is not None else None,
                    )

                if checkpoint_dir is not None and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
                    ckpt_path = checkpoint_tt.save_checkpoint(model.encoder, model.decoder, step + 1, checkpoint_dir)
                    if print_log:
                        print(f"  checkpoint saved: {ckpt_path}")
        finally:
            if prefetch_source is not None:
                prefetch_source.close()

        if checkpoint_dir is not None and checkpoint_at_end:
            ckpt_path = checkpoint_tt.save_checkpoint(model.encoder, model.decoder, num_steps, checkpoint_dir)
            if print_log:
                print(f"  final checkpoint saved: {ckpt_path}")

    return history


# ---------------------------------------------------------------------------
# Presets + CLI
# ---------------------------------------------------------------------------


def smoke_contract() -> ShapeContract:
    """Tiny-width/depth config for the pytest smoke test
    (`test_main_pretrain_tt.py`): model dims match `test_modules_tt
    ._contract()`'s scale (already verified fast and correct on this
    hardware); `img_size` matches the real FOMO_with_dwi shards
    `(208, 240, 208)` (module docstring: no synthetic data path exists, so
    every contract that goes through `train()` must densify against a real
    shard's actual volume shape). `l_vis=31`/`q_pred=32` stay tiny -- real
    shards have thousands of valid patches (`TENSTORRENT_PORT.md` section 1),
    so this budget is comfortably covered without sample-with-replacement.
    """
    return ShapeContract(
        batch=2,
        l_vis=31,
        q_pred=32,
        mask_ratio=0.8,
        img_size=(208, 240, 208),
        patch_size=8,
        in_chans=1,
        embed_dim=64,
        decoder_embed_dim=32,
        encoder_heads=2,
        decoder_heads=1,
        class_token=True,
    )


def gate_contract() -> ShapeContract:
    """Moderately larger config for the standalone M5 200-step gate run --
    still far smaller than real ViT-L (module docstring), but with more
    depth/width than `smoke_contract()` so the loop exercises more than a
    single-block model. `img_size` matches the real shards, same reasoning
    as `smoke_contract()`.
    """
    return ShapeContract(
        batch=4,
        l_vis=95,
        q_pred=32,
        mask_ratio=0.8,
        img_size=(208, 240, 208),
        patch_size=8,
        in_chans=1,
        embed_dim=128,
        decoder_embed_dim=64,
        encoder_heads=4,
        decoder_heads=2,
        class_token=True,
    )


def full_vitl_real_contract() -> ShapeContract:
    """Real, unshrunk ViT-L MAE config for the full-scale end-to-end demo run
    (TENSTORRENT_PORT.md's "capstone" milestone): encoder D=1024/depth=24/
    heads=16, decoder D=512/depth=4/heads=16, mlp_ratio=4, class_token=True,
    patch_size=8, img_size=(208,240,208), in_chans=1, mask_ratio=0.80 --
    every architecture number matches `src/smri_mae/model_mae.py::mae_vit_large`
    / `src/smri_mae/config/default_pretrain.yaml` verbatim (see
    `scripts/memory_probe.py`'s module docstring for the exact provenance of
    each value); none of them may be changed to make this run cheaper.

    `l_vis=543`/`q_pred=2272` are **not** the memory probe's deliberately
    conservative full-grid ceiling (l_vis=4031/q_pred=4160) -- they come from
    `scripts/measure_occupancy.py --split train`'s real measurement over the
    450 local `datasets/FOMO_with_dwi/shard.{000000,000001,000002}.tar`
    samples: p1=2842.83/p50=3673.5/p99=4729.18 valid patches/sample, giving
    `choose_l_vis(mask_ratio=0.80)` -> 544 (class-token-adjusted: 543) and
    `choose_q_pred` -> 2272 (encoder_seq_len=544, decoder_seq_len=2816, both
    multiples of 32). `batch=24` was chosen from a handful of real-hardware
    `scripts/memory_probe.py --l-vis 543 --q-pred 2272 --batch B` trials at
    this exact shape: batch=4/8/16/24 peaked at 24.6%/35.5%/57.4%/79.3% of
    the 31.83 GiB DRAM budget (all safe), batch=28 hit 90.2% (tight), and
    batch=32 OOM'd outright (`TT_FATAL: Out of Memory ... allocator/
    bank_manager.cpp`) -- so batch=24 is the largest trial with comfortable
    headroom for a real, sustained training run (data loading, checkpoint
    writes, wandb logging all happen host-side and add negligible device
    memory, but a long run is not the place to run within a few points of an
    allocator that already ran out at the very next batch size tried).
    """
    return ShapeContract(
        batch=24,
        l_vis=543,
        q_pred=2272,
        mask_ratio=0.80,
        img_size=(208, 240, 208),
        patch_size=8,
        in_chans=1,
        embed_dim=1024,
        decoder_embed_dim=512,
        encoder_heads=16,
        decoder_heads=16,
        class_token=True,
    )


_PRESETS = {
    "smoke": (smoke_contract, 2, 1),
    "gate": (gate_contract, 4, 2),
    "full-vitl-real": (full_vitl_real_contract, 24, 4),
}


def _default_run_name(preset: str) -> str:
    return f"{preset}-{time.strftime('%Y%m%d-%H%M%S')}"


def main(argv: Optional[Sequence[str]] = None) -> list[StepLog]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(_PRESETS), default="gate")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--accum-iter", type=int, default=1)
    parser.add_argument("--base-lr", type=float, default=TTMAEConfig.base_lr)
    parser.add_argument("--min-lr", type=float, default=TTMAEConfig.min_lr)
    parser.add_argument("--weight-decay", type=float, default=TTMAEConfig.weight_decay)
    parser.add_argument("--clip-grad", type=float, default=TTMAEConfig.clip_grad)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--log-file", type=str, default=None, help="optional path to dump per-logged-step JSON history")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="run a no-grad val pass over the local val shards every N steps "
        "(0 = never; also always runs once on the final step if > 0).",
    )
    parser.add_argument("--run-name", type=str, default=None, help="default: '<preset>-<timestamp>'")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="default: tenstorrent/checkpoints/<run-name>",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="save a checkpoint every N optimizer steps in addition to the end-of-run checkpoint (0 = end-of-run only)",
    )
    parser.add_argument("--no-checkpoint", action="store_true", help="disable checkpoint saving entirely")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to a checkpoint_tt.save_checkpoint() output to load encoder/decoder weights "
        "from before training (weights only -- see checkpoint_tt.py's module docstring on "
        "optimizer state not being restored)",
    )
    parser.add_argument("--wandb", dest="wandb", action="store_true", default=False)
    parser.add_argument("--no-wandb", dest="wandb", action="store_false")
    parser.add_argument("--wandb-project", type=str, default="smri-mae-tt")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument(
        "--no-prefetch",
        dest="prefetch",
        action="store_false",
        default=True,
        help="disable background-process data_pack prefetch (real_data.ProcessPrefetchingBatchSource); "
        "on by default -- pass this to isolate/compare against the non-overlapped baseline.",
    )
    args = parser.parse_args(argv)

    contract_fn, encoder_depth, decoder_depth = _PRESETS[args.preset]
    contract = contract_fn()
    cfg = TTMAEConfig(
        shape_contract=contract,
        base_lr=args.base_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        accum_iter=args.accum_iter,
    )

    run_name = args.run_name or _default_run_name(args.preset)

    train_source = real_data.RealTrainSource(contract)
    train_batch_fn = train_source.next_batch
    train_batch_source_factory = functools.partial(real_data.RealTrainSource, contract)
    epoch_equiv_fn = lambda: train_source.epoch_equiv  # noqa: E731
    print(
        f"real data: training on {len(real_data.TRAIN_SHARD_URLS)} shard(s), "
        f"{real_data.TRAIN_SAMPLES_TOTAL} samples total"
    )
    eval_batch_iter_fn = None
    if args.eval_every > 0:
        val_source = real_data.RealValSource(contract)
        eval_batch_iter_fn = val_source.eval_batches
        print(
            f"real data: evaluating on {len(real_data.VAL_SHARD_URLS)} shard(s), "
            f"{real_data.VAL_SAMPLES_TOTAL} samples total, every {args.eval_every} steps"
        )

    checkpoint_dir = None
    checkpoint_every = 0
    checkpoint_at_end = False
    if not args.no_checkpoint:
        checkpoint_dir = args.checkpoint_dir or os.path.join(_DEFAULT_CHECKPOINT_ROOT, run_name)
        checkpoint_every = args.checkpoint_every
        checkpoint_at_end = True

    wandb_run = None
    if args.wandb:
        wandb_run = wandb_tt.init_wandb(
            project=args.wandb_project,
            run_name=run_name,
            entity=args.wandb_entity,
            notes=None,
            config={
                "preset": args.preset,
                "steps": args.steps,
                "encoder_depth": encoder_depth,
                "decoder_depth": decoder_depth,
                "eval_every": args.eval_every,
                **asdict(cfg),  # includes shape_contract (=contract) nested
            },
        )

    start = time.monotonic()
    try:
        history = train(
            contract,
            encoder_depth,
            decoder_depth,
            cfg,
            args.steps,
            warmup_steps=args.warmup_steps,
            accum_iter=args.accum_iter,
            seed=args.seed,
            log_every=args.log_every,
            print_log=True,
            train_batch_fn=train_batch_fn,
            train_batch_source_factory=train_batch_source_factory,
            epoch_equiv_fn=epoch_equiv_fn,
            eval_every=args.eval_every,
            eval_batch_iter_fn=eval_batch_iter_fn,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=checkpoint_every,
            checkpoint_at_end=checkpoint_at_end,
            resume_checkpoint=args.resume,
            wandb_run=wandb_run,
            prefetch=args.prefetch,
        )
    finally:
        wandb_tt.finish_wandb(wandb_run)
    elapsed = time.monotonic() - start
    print(f"done: {args.steps} steps in {elapsed:.1f}s ({elapsed / args.steps * 1000:.1f} ms/step avg)")

    if args.log_file:
        with open(args.log_file, "w") as f:
            json.dump([asdict(entry) for entry in history], f, indent=2)
        print(f"wrote history to {args.log_file}")

    return history


if __name__ == "__main__":
    main(sys.argv[1:])
