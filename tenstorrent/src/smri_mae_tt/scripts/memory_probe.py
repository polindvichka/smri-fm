"""Standalone device-memory ceiling probe for the *full-size* ViT-L MAE, on
real Blackhole hardware.

This is **not** a unit test (filename deliberately does not match
`test_*.py`/`*_test.py` so pytest never collects it -- run it directly with
the tt-train Python interpreter) and it is not part of the M0-M7 milestone
code path. It exists purely to answer one question raised by
`TENSTORRENT_PORT.md`'s standing instruction ("If you find out ... current
card memory is not enough ... you need to stop and let me know"): does the
*actual, unshrunk* ViT-L config (24-layer D=1024 encoder / 4-layer D=512
decoder, real `img_size`/`patch_size`) fit in this card's ~31 GiB DRAM, and
if so with how much headroom -- measured with the real `ttnn`/`ttml` device
memory APIs, not estimated.

Every other test/module under `smri_mae_tt/` (`test_modules_tt.py`,
`test_model_tt.py`, `main_pretrain_tt.py`'s `smoke_contract`/`gate_contract`)
has only ever exercised `embed_dim`/`decoder_embed_dim`/depth values a few
percent of the real config's size, for speed. This script is the first time
the real numbers are used end-to-end.

## Real ViT-L hyperparameters (do not edit without re-reading the sources)

From `src/smri_mae/model_mae.py::mae_vit_large`:
    embed_dim=1024, depth=24, num_heads=16   (encoder)

From `src/smri_mae/config/default_pretrain.yaml`:
    img_size: [208, 240, 208], patch_size: 8, in_chans=1 (model default)
    mask_ratio: 0.80, pad_to_multiple: 32
    model_kwargs.decoder_depth: 4
    (decoder_embed_dim/decoder_num_heads not overridden in the yaml -> the
    `MaskedAutoencoderViT.__init__` defaults apply: decoder_embed_dim=512,
    decoder_num_heads=16)
    mlp_ratio, class_token: not overridden -> defaults 4, True

So: encoder D=1024/depth=24/heads=16, decoder D=512/depth=4/heads=16,
mlp_ratio=4, class_token=True, patch_size=8, img_size=(208,240,208),
in_chans=1, mask_ratio=0.80. Grid = (26, 30, 26) -> num_patches=20280,
patch_dim = 1*8**3 = 512. This matches TENSTORRENT_PORT.md section 1's
"302.8M encoder / 13.4M decoder, 316.2M total" and "20,280 patches,
patch_dim=512" verbatim -- nothing here is approximated.

## l_vis / q_pred: a memory-ceiling budget, not a measured-occupancy budget

No real MRI shards exist on this host (see `main_pretrain_tt.py`'s module
docstring), so `layout.choose_l_vis`/`choose_q_pred` (which need a measured
per-sample valid-patch distribution) cannot be used. Per this task's
instructions, this probe instead computes a budget directly from the *full*
patch grid (no brain-mask occupancy discount -- deliberately the
conservative/larger-than-realistic direction, since occupancy in the real
data is measured at only 25-35% per TENSTORRENT_PORT.md section 1's table):

    target_l_vis = num_patches * (1 - mask_ratio) = 20280 * 0.20 = 4056

`layout.ShapeContract.__post_init__` requires `encoder_seq_len =
l_vis + class_token` to be tile-aligned (see that file's own long comment:
with `class_token=True`, requiring `l_vis` itself to *also* be tile-aligned
is unsatisfiable, so only `encoder_seq_len`/`decoder_seq_len` are checked).
So `l_vis` is taken as the largest value <= target_l_vis with
`(l_vis + 1) % 32 == 0`:

    l_vis = round_down_to_multiple(target_l_vis, TILE) - 1 = 4032 - 1 = 4031
    encoder_seq_len = 4032  (= 126 * 32)

For `q_pred`: decoding literally every remaining candidate patch
(`num_patches - l_vis = 16249`, `pred_mask_ratio: null`'s real semantics)
would give `decoder_seq_len ~= 20281` -- about 2.9x TENSTORRENT_PORT.md
section 1's own largest worked *plausible-occupancy* example
(valid=7098 -> decoder_seq=7099), and roughly 4x the largest shape this
project's doc has ever measured decoder attention at (S=5632, section 0/8).
That is not "the real full-size config" so much as an artifact of probing
with zero occupancy discount on a rectangular bounding box the brain does
not fill -- decoding all 16249 patches is not a scenario the real training
recipe would ever produce. Per this task's explicit instruction to pick "a
documented smaller fraction if that number is enormous", this probe targets
a round, tile-aligned `decoder_seq_len = 8192` instead -- about 1.15x
section 1's largest documented plausible-occupancy example, comfortably
past everything this project has measured before (still an aggressive,
ceiling-probing choice) but not a 4x leap into completely unmeasured
territory in the very first full-size run:

    q_pred = decoder_seq_len_target - l_vis - 1 = 8192 - 4031 - 1 = 4160
    decoder_seq_len = 8192  (= 256 * 32)

`q_pred=4160` is ~25.6% of the 16249 full remaining candidates -- i.e. this
is exactly the "(2) set Q_pred below the full candidate set" lever
TENSTORRENT_PORT.md section 8 already names as the natural static-shape
performance/memory knob, applied here as a documented, deliberate choice
rather than an approximation forced by ignorance.

Both `l_vis=4031` and `q_pred=4160` individually need not be tile-aligned
(only the two `_seq_len` properties do, and both are exact multiples of 32
by construction above) -- see `layout.py`'s own docstring for why.

## What this script does

`run_probe(batch)`: opens a `MeshContext`, builds the full-size
`ShapeContract` above at the given `batch`, initializes
`MaskedAutoencoderViTTT` + fresh params + the real `MultiGroupAdamW`
optimizer (mirroring `main_pretrain_tt.train()` exactly, just
instrumented), builds one real packed batch (`real_data.RealTrainSource`,
the same real-shard source the M5 training loop uses -- `main_pretrain_tt.py`
has no synthetic data path), and runs one forward + backward + optimizer
step. Each stage prints a
START/DONE (or FAILED, with the exact exception) line with wall time, and a
device DRAM/L1 snapshot via `ttnn.get_memory_view` (the real allocator
statistics API -- see module docstring below for how this was found) is
taken after every stage.

## The memory-reporting API

Investigated by grepping `tenstorrent/tt-metal` (read-only, per project
rules) for `dump_device_memory_state`/`get_memory_view`/`allocator` stats.
Two real, already-shipped APIs were found and are both used here:

  - `ttnn.get_memory_view(device, ttnn.BufferType.DRAM | L1)` ->
    `MemoryView` (`ttnn/ttnn/device.py:211`, backed by
    `tt::tt_metal::detail::GetMemoryView`, itself
    `device->allocator()->get_statistics(buffer_type)`,
    `tt_metal/detail/reports/memory_reporter.cpp`). Fields used here:
    `.num_banks`, `.total_bytes_per_bank`, `.total_bytes_allocated_per_bank`,
    `.total_bytes_free_per_bank`, `.largest_contiguous_bytes_free_per_bank`
    -- all *per bank*; multiply by `.num_banks` for device-wide totals (this
    is exactly what `tt-train/sources/ttml/ttml/common/utils.py`'s own
    `get_available_device_memory_in_bytes` does). Confirmed empirically on
    this host before writing this script: DRAM = 8 banks x 4,272,341,376 B
    = 34,178,731,008 B = 31.83 GiB total allocatable; L1 = 110 banks x
    1,461,504 B = 160,765,440 B = 153.32 MiB total.
  - `ttnn.dump_device_memory_state(device, prefix=...)` (`ttnn/ttnn/device.py:207`,
    backed by `tt::tt_metal::detail::DumpDeviceMemoryState`) writes
    `<prefix>memory_usage_summary.csv` / `<prefix>l1_usage_summary.csv` /
    `<prefix>detailed_memory_usage.csv` under `generated/reports/` (relative
    to CWD) -- called at every stage below as a supplementary on-disk trail,
    in addition to the numeric `get_memory_view` snapshot printed live.

Fail-fast, no fallbacks (project rule): if any stage raises (OOM, allocator
failure, or otherwise), this script prints the exact exception and the
stage/shape it occurred at, attempts one best-effort memory snapshot, and
re-raises -- it does not shrink the contract, retry with a smaller shape, or
swallow the error.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np
import ttml
import ttnn

from smri_mae_tt.config_tt import (
    MultiGroupAdamW,
    TTMAEConfig,
    build_param_group_specs,
    clip_grad_norm,
)
from smri_mae_tt import real_data
from smri_mae_tt.layout import TILE, ShapeContract, round_down_to_multiple
from smri_mae_tt.main_pretrain_tt import _tensor_to_scalar, forward_and_loss
from smri_mae_tt.mesh import MeshContext
from smri_mae_tt.model_tt import MaskedAutoencoderViTTT, ModelParams
from smri_mae_tt.params import init_tt_params, named_tt_parameters

# ---------------------------------------------------------------------------
# Real ViT-L hyperparameters -- see module docstring for exact provenance.
# ---------------------------------------------------------------------------

IMG_SIZE = (208, 240, 208)
PATCH_SIZE = 8
IN_CHANS = 1
MASK_RATIO = 0.80

EMBED_DIM = 1024
ENCODER_DEPTH = 24
ENCODER_HEADS = 16

DECODER_EMBED_DIM = 512
DECODER_DEPTH = 4
DECODER_HEADS = 16

MLP_RATIO = 4.0
CLASS_TOKEN = True

_GRID = tuple(s // PATCH_SIZE for s in IMG_SIZE)
NUM_PATCHES = _GRID[0] * _GRID[1] * _GRID[2]
PATCH_DIM = IN_CHANS * PATCH_SIZE**3

# Deliberately chosen decoder_seq_len ceiling -- see module docstring
# ("q_pred: a memory-ceiling budget") for why this is 8192 and not
# "decode every remaining candidate".
_DECODER_SEQ_LEN_TARGET = 8192


def full_size_contract(batch: int, l_vis: int | None = None, q_pred: int | None = None) -> ShapeContract:
    """Full-size ViT-L contract at the given batch. `l_vis`/`q_pred`
    default to this module's original full-grid-worst-case derivation (see
    module docstring); pass explicit values (e.g. from
    `scripts/measure_occupancy.py`'s real measured-occupancy budget) to
    probe memory at a realistic, smaller shape instead -- everything else
    about the model (depth/heads/embed dims/img_size) is unchanged either
    way, per this project's architecture-immutability rule.
    """
    if l_vis is None:
        target_l_vis = int(NUM_PATCHES * (1.0 - MASK_RATIO))
        l_vis = round_down_to_multiple(target_l_vis, TILE) - 1
    if q_pred is None:
        q_pred = _DECODER_SEQ_LEN_TARGET - l_vis - 1
    return ShapeContract(
        batch=batch,
        l_vis=l_vis,
        q_pred=q_pred,
        mask_ratio=MASK_RATIO,
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_chans=IN_CHANS,
        embed_dim=EMBED_DIM,
        decoder_embed_dim=DECODER_EMBED_DIM,
        encoder_heads=ENCODER_HEADS,
        decoder_heads=DECODER_HEADS,
        class_token=CLASS_TOKEN,
    )


# ---------------------------------------------------------------------------
# Memory reporting
# ---------------------------------------------------------------------------


def mem_snapshot(device, label: str) -> dict:
    dram = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    l1 = ttnn.get_memory_view(device, ttnn.BufferType.L1)

    dram_total = dram.num_banks * dram.total_bytes_per_bank
    dram_alloc = dram.num_banks * dram.total_bytes_allocated_per_bank
    dram_free = dram.num_banks * dram.total_bytes_free_per_bank
    l1_total = l1.num_banks * l1.total_bytes_per_bank
    l1_alloc = l1.num_banks * l1.total_bytes_allocated_per_bank
    l1_free = l1.num_banks * l1.total_bytes_free_per_bank

    print(
        f"[mem] {label:32s} "
        f"DRAM {dram_alloc / 2**30:7.3f} / {dram_total / 2**30:.3f} GiB "
        f"({100 * dram_alloc / dram_total:5.1f}%), free {dram_free / 2**30:6.3f} GiB, "
        f"largest_free_block/bank {dram.largest_contiguous_bytes_free_per_bank / 2**20:8.1f} MiB | "
        f"L1 {l1_alloc / 2**20:7.2f} / {l1_total / 2**20:.2f} MiB ({100 * l1_alloc / l1_total:5.1f}%)",
        flush=True,
    )
    try:
        ttnn.dump_device_memory_state(device, prefix=f"memprobe_{label}_")
    except Exception as e:  # noqa: BLE001 -- reporting best-effort, never masks the real run
        print(f"[mem]   (dump_device_memory_state failed for {label!r}: {e!r})", flush=True)

    return {
        "label": label,
        "dram_alloc": dram_alloc,
        "dram_total": dram_total,
        "dram_free": dram_free,
        "l1_alloc": l1_alloc,
        "l1_total": l1_total,
        "l1_free": l1_free,
    }


# ---------------------------------------------------------------------------
# The probe itself
# ---------------------------------------------------------------------------


def run_probe(batch: int, seed: int = 0, l_vis: int | None = None, q_pred: int | None = None) -> list[dict]:
    contract = full_size_contract(batch, l_vis=l_vis, q_pred=q_pred)
    print(f"=== full-size ViT-L memory probe: batch={batch} ===", flush=True)
    print(
        f"contract: l_vis={contract.l_vis} q_pred={contract.q_pred} "
        f"encoder_seq_len={contract.encoder_seq_len} decoder_seq_len={contract.decoder_seq_len} "
        f"num_patches={contract.num_patches} patch_dim={contract.patch_dim} "
        f"embed_dim={contract.embed_dim} decoder_embed_dim={contract.decoder_embed_dim} "
        f"encoder_depth={ENCODER_DEPTH} decoder_depth={DECODER_DEPTH}",
        flush=True,
    )

    history: list[dict] = []

    def snap(device, label: str) -> None:
        history.append(mem_snapshot(device, label))

    with MeshContext() as mesh:
        device = mesh.device
        snap(device, "00_device_open")

        stage_name = "01_host_param_init"
        print(f"[stage] START {stage_name}", flush=True)
        t0 = time.monotonic()
        try:
            param_rng = np.random.default_rng(seed)
            encoder_params, decoder_params = init_tt_params(
                param_rng, contract, ENCODER_DEPTH, DECODER_DEPTH, mlp_ratio=MLP_RATIO
            )
        except Exception:
            print(f"[stage] FAILED {stage_name} after {time.monotonic() - t0:.1f}s", flush=True)
            traceback.print_exc()
            snap(device, f"FAILED_{stage_name}")
            raise
        print(f"[stage] DONE  {stage_name} in {time.monotonic() - t0:.1f}s", flush=True)
        snap(device, stage_name)

        stage_name = "02_model_construction"
        print(f"[stage] START {stage_name} (weight upload to device)", flush=True)
        t0 = time.monotonic()
        try:
            model = MaskedAutoencoderViTTT(
                contract,
                ENCODER_DEPTH,
                DECODER_DEPTH,
                ModelParams(encoder=encoder_params, decoder=decoder_params),
                mlp_ratio=MLP_RATIO,
            )
        except Exception:
            print(f"[stage] FAILED {stage_name} after {time.monotonic() - t0:.1f}s", flush=True)
            traceback.print_exc()
            snap(device, f"FAILED_{stage_name}")
            raise
        print(f"[stage] DONE  {stage_name} in {time.monotonic() - t0:.1f}s", flush=True)
        snap(device, stage_name)

        named_params = named_tt_parameters(model.encoder, model.decoder)
        n_params = sum(int(np.prod(t.shape())) for t in named_params.values())
        print(f"[info] total trainable params: {n_params:,} ({n_params / 1e6:.1f}M)", flush=True)

        stage_name = "03_optimizer_construction"
        print(f"[stage] START {stage_name}", flush=True)
        t0 = time.monotonic()
        try:
            cfg = TTMAEConfig(shape_contract=contract)
            group_specs = build_param_group_specs(named_params)
            total_batch_size = batch  # accum_iter=1, world_size=1 for this probe
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
        except Exception:
            print(f"[stage] FAILED {stage_name} after {time.monotonic() - t0:.1f}s", flush=True)
            traceback.print_exc()
            snap(device, f"FAILED_{stage_name}")
            raise
        print(f"[stage] DONE  {stage_name} in {time.monotonic() - t0:.1f}s", flush=True)
        snap(device, stage_name)

        stage_name = "04_real_batch_pack"
        print(f"[stage] START {stage_name}", flush=True)
        t0 = time.monotonic()
        try:
            data_rng = np.random.default_rng(seed + 1)
            batch_obj = real_data.RealTrainSource(contract).next_batch(data_rng)
        except Exception:
            print(f"[stage] FAILED {stage_name} after {time.monotonic() - t0:.1f}s", flush=True)
            traceback.print_exc()
            snap(device, f"FAILED_{stage_name}")
            raise
        print(f"[stage] DONE  {stage_name} in {time.monotonic() - t0:.1f}s", flush=True)
        snap(device, stage_name)

        stage_name = "05_forward"
        print(f"[stage] START {stage_name}", flush=True)
        t0 = time.monotonic()
        try:
            loss = forward_and_loss(model, batch_obj)
            loss_value = _tensor_to_scalar(loss)
        except Exception:
            print(f"[stage] FAILED {stage_name} after {time.monotonic() - t0:.1f}s", flush=True)
            traceback.print_exc()
            snap(device, f"FAILED_{stage_name}")
            raise
        print(f"[stage] DONE  {stage_name} in {time.monotonic() - t0:.1f}s, loss={loss_value:.6f}", flush=True)
        snap(device, stage_name)

        stage_name = "06_backward"
        print(f"[stage] START {stage_name}", flush=True)
        t0 = time.monotonic()
        try:
            loss.backward(False)
        except Exception:
            print(f"[stage] FAILED {stage_name} after {time.monotonic() - t0:.1f}s", flush=True)
            traceback.print_exc()
            snap(device, f"FAILED_{stage_name}")
            raise
        print(f"[stage] DONE  {stage_name} in {time.monotonic() - t0:.1f}s", flush=True)
        snap(device, stage_name)

        stage_name = "07_grad_clip_and_optimizer_step"
        print(f"[stage] START {stage_name}", flush=True)
        t0 = time.monotonic()
        try:
            grad_norm_tensor = clip_grad_norm(named_params, cfg.clip_grad)
            grad_norm_value = _tensor_to_scalar(grad_norm_tensor)
            optimizer.step()
        except Exception:
            print(f"[stage] FAILED {stage_name} after {time.monotonic() - t0:.1f}s", flush=True)
            traceback.print_exc()
            snap(device, f"FAILED_{stage_name}")
            raise
        print(
            f"[stage] DONE  {stage_name} in {time.monotonic() - t0:.1f}s, grad_norm={grad_norm_value:.4f}",
            flush=True,
        )
        snap(device, stage_name)

        ttml.autograd.AutoContext.get_instance().reset_graph()
        snap(device, "08_after_reset_graph")

    print("\n=== summary (batch=%d) ===" % batch, flush=True)
    for h in history:
        print(
            f"{h['label']:32s} DRAM {h['dram_alloc'] / 2**30:7.3f} / {h['dram_total'] / 2**30:.3f} GiB "
            f"({100 * h['dram_alloc'] / h['dram_total']:5.1f}%)   "
            f"L1 {h['l1_alloc'] / 2**20:7.2f} / {h['l1_total'] / 2**20:.2f} MiB",
            flush=True,
        )
    peak = max(history, key=lambda h: h["dram_alloc"])
    print(
        f"\npeak DRAM usage: {peak['dram_alloc'] / 2**30:.3f} GiB / {peak['dram_total'] / 2**30:.3f} GiB "
        f"at stage {peak['label']!r}",
        flush=True,
    )
    return history


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1, help="micro-batch size to probe")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--l-vis", type=int, default=None, help="override l_vis (default: full-grid worst-case, see module docstring)"
    )
    parser.add_argument(
        "--q-pred", type=int, default=None, help="override q_pred (default: full-grid worst-case, see module docstring)"
    )
    args = parser.parse_args(argv)

    try:
        run_probe(args.batch, seed=args.seed, l_vis=args.l_vis, q_pred=args.q_pred)
    except Exception as e:  # noqa: BLE001 -- top-level: report verbatim, do not swallow
        print(f"\n[FATAL] probe failed at batch={args.batch}: {type(e).__name__}: {e}", flush=True)
        raise


if __name__ == "__main__":
    main(sys.argv[1:])
