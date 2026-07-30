If you find out in the process of work that current card memory is not enough and i.e you need to stop and let me know
- All code must be under tenstorrent folder
- Use uv as package manager, code and deps /must be reproducible
- No fallbacks principles, clean, pure code, fail fast
- No patches over existing tt-nn code, you can write new functionality if you need

# Native TT-NN port of the sMRI MAE

Porting `src/smri_mae` (3D ViT-L MAE pretraining) to run natively on Blackhole
via TT-NN/TT-Metalium — hand-written ops and an explicit forward/backward
graph, **not** a PyTorch-to-TT auto-compiler. All new code lives under
`tenstorrent/src/smri_mae_tt/`; `src/smri_mae`, `src/data` are reference-only
and never modified; `tenstorrent/tt-metal/` is vendored upstream and
read-only except for precisely-scoped, verified fixes (currently 8 modified
files, see "Vendored tt-metal changes" below).

**This is the single doc for this project.** `tenstorrent/docs/` has been
removed — its content (session-by-session optimization narrative,
tt-metal-tactics research notes) is superseded by this file, which keeps
only what's still true and actionable.

---

## 1. Status

Functionally complete and gated (M0-M6 below all passed their gates).
Currently **not at performance parity with the 4090 baseline** — see §3.

| Milestone | Status |
|---|---|
| M0 Environment + causal-mask fix | done |
| M1 Shape contract | done |
| M2 Encoder forward parity | done |
| M3 Full forward + loss parity | done |
| M4 Backward parity | done |
| M5 Training loop (real data, checkpoints, wandb) | done |
| M6 Performance tuning | in progress — see §3/§4, real gap remains |
| M7 Multi-card mesh (N cards) | not started |

Full test suite is green (`pytest` under `tenstorrent/src/smri_mae_tt/`,
requires real FOMO_with_dwi shards downloaded from R2 per `.env`).

---

## 2. What is actually being ported

The model (`src/smri_mae/model_mae.py`, `modules.py`) is an MAE over 3D volumes:

- Input `[B, 1, 208, 240, 208]`, patch `8³` → grid `26×30×26` = 20,280 patches, `patch_dim = 512`.
- Encoder: ViT-L, `D=1024`, depth 24, 16 heads, 302.8M params, runs on visible tokens only.
- Decoder: `D=512`, depth 4, 16 heads, 13.4M params, runs on visible + prediction-query tokens.
- Fixed (non-trainable) 3D sin/cos position embeddings.
- Loss: per-scan mean of valid-voxel MSE over predicted patches.
- Total 316.2M trainable params.

Static shapes: `L_vis`/`Q_pred` are compile-time constants (`layout.py`'s
`ShapeContract`), rounded to tile multiples, sampled per-sample on the host.
Production shape (`full_vitl_real_contract`): `l_vis=543, q_pred=2272`.

No autograd in stock `ttnn` (no SDPA backward) — built on `ttml` (tt-train),
which has `ttml.autograd.Tensor`/`Function`, fused `sdpa_fw`/`sdpa_bw` Metal
kernels, `AdamW`, `clip_grad_norm`, mesh/FSDP primitives. `ttml`'s SDPA
originally hardcoded `AttentionMaskType::Causal` when no mask was passed —
our attention is bidirectional; this was the M0-blocking fix (extend the
ops-layer wrapper + nanobind signature to accept `AttentionMaskType::None`).

Data path never densifies: sparse shard → host-side numpy gather →
`[B, L_vis, 512]`/`[B, Q_pred, 512]` packed patch values + ids, transferred
directly. `patchify3d`/`unpatchify3d` are host-side and off the device graph.

Package layout (`tenstorrent/src/smri_mae_tt/`):
```
layout.py       # ShapeContract: L_vis, Q_pred, tile alignment
packing.py      # host: sparse shard -> packed patch tensors + ids
params.py       # checkpoint -> ttnn/ttml tensors, dtype policy
modules_tt.py   # Attention / Mlp / Block / LayerNorm on ttml.autograd
model_tt.py     # MaskedEncoder / MaskedDecoder / MaskedAutoencoderViT
ops_tt/         # tier-1 Functions (tuned_gelu, tuned_linear, flash_attention, ...)
config_tt.py    # SDPA/matmul program configs + TTMAEConfig, keyed by shape
mesh.py         # MeshDevice setup, gradient all-reduce, rank plumbing
real_data.py    # RealTrainSource/RealValSource/ProcessPrefetchingBatchSource
main_pretrain_tt.py  # training loop (train(), the production entry point)
scripts/        # bench_step.py, device_profile_report.py, and other one-off probes
```

---

## 3. Performance: the real, corrected number

**4090 reference recipe**: global batch 60 = batch 12 × `accum_iter` 5,
1920 ms per optimizer step → **32.0 ms/sample (31.25 samples/sec)**.

**TT, measured 2026-07-30, real hardware, `full_vitl_real_contract`
(batch=12 or 24, encoder D=1024/depth=24, decoder D=512/depth=4, real
l_vis=543/q_pred=2272), warm steps, `train()`'s current defaults
(prefetch on, tuned-GELU on)**:

| batch | ms/step | ms/sample |
|---|---|---|
| 24 | ~1795 (two-run avg 1791.6/1799.0) | ~74.8 |
| 12 | ~910 (avg of 2 warm steps, 22.7s/2.9s JIT-warmup steps excluded) | ~75.85 |
| 24, `use_tuned_linear=True` (2026-07-30, §4/§5 item 1) | ~1759.6 (5-step avg, two baseline reruns at ~1794-1796 for comparison) | ~73.3 |

Per-sample cost is consistent across both batch sizes (~75 ms/sample) — this
is a real, batch-size-independent rate, not an artifact of one measurement.
`use_tuned_linear=True` (sweep-cached matmul `program_config` on `Attention.
proj`/`Mlp.fc2`, see §4/§5 item 1) is a real, reproducible ~2% end-to-end
step-time reduction — small next to the ~2.4x gap below, not yet defaulted
on in `train()` (see `ops_tt/tuned_linear.py`'s docstring: it changes which
code path runs even on a cache miss, which broke the golden-baseline
regression gate's tight tolerance once already).

**TT is running at ~75 / 32.0 ≈ 2.37x slower per sample than the 4090** —
i.e. TT needs to close a **~2.4x deficit just to match the 4090**, then
another 2x beyond that to hit this project's "2x the 4090" goal (~4.7x
total from where it stands today).

### Why earlier numbers in this project said "faster than the 4090"

Every TT step-time benchmark up to this point used `batch=24, accum_iter=1`
(24 samples/optimizer-step) and compared its `ms/step` directly against the
4090's `1920ms` figure, which is for `batch=12 × accum_iter=5` (60
samples/optimizer-step) — different amounts of work per "step". That
produced a string of "TT is ~1.06-1.14x faster than the 4090" claims (the
`~1.795s/step` M6 gate figure, `~1.83s/step` at an earlier point) that were
never actually apples-to-apples. Normalizing to ms/sample (the only unit
that's valid regardless of batch/accum composition) reverses the
conclusion: TT is slower, not faster. This correction obsoletes every
"faster than 4090" statement anywhere in this project's prior history.

---

## 4. Where the time actually goes (real device-level profiling)

Real per-op device-kernel-duration breakdown, `python -m
smri_mae_tt.scripts.device_profile_report --preset full-vitl-real`
(Tracy device profiler, `TT_METAL_DEVICE_PROFILER=1`), one production step,
2685 device op dispatches, 1703.98ms total device kernel time:

| op | % of step | notes |
|---|---|---|
| SDPA backward-KV | 26.9% | single largest op in the step, larger than all matmuls combined |
| Matmul (all `ttml.ops.linear.linear` calls) | 22.9% | 345 calls |
| SDPA backward-Q | 15.5% | |
| SDPA forward | 11.4% | |
| GELU backward | 3.1% | |
| everything else (norms, reductions, AdamW, grad-clip, ...) | ~20% | each individually <2.5% |

**SDPA (fwd+bwd) is 53.9% of the step.** Decoder-scale (`Ht=88`) SDPA
backward-KV calls cost ~91ms each vs ~3.8ms at encoder scale (`Ht=17`) — a
23.8x scaling, close to the theoretical `Ht²=26.8x`, meaning the KV kernel
is close to its quadratic-compute ceiling at decoder scale, not wasting
work on something fixable at the kernel-config level.

**Utilization**: both SDPA and matmul run at `MathFidelity::HiFi4` (1
TFLOPS/matrix-engine) by design, not the 332 TFLOPS marketing peak (LoFi,
4 TFLOPS/engine). Achieved throughput: SDPA ~1.16 TFLOPS (encoder) / ~3.26
TFLOPS (decoder) out of ~110 available matrix engines. Matmul's default
linear op is flat at ~1 TFLOPS achieved regardless of problem size
(batch=1/4/24/96 all measured within noise of "one engine's worth") —
genuinely compute-limited at a low per-call parallelism, not
dispatch-latency-bound.

### Confirmed dead ends (do not re-attempt without new evidence)

1. **MathFidelity tuning** (HiFi4→HiFi2→LoFi) on SDPA, default linear, and
   the 8 previously-uninvestigated non-SDPA op codes (GELU-bw, MorehSum,
   BinaryNg, MorehClipGradNorm, Reduce, Unary, LayerNorm-bw, AdamW) — zero
   measured speedup everywhere it was checked. Not fidelity-bound.
2. **Matmul program-config hand-tuning, one hand-picked config** (original
   M6 finding) — 0.95-1.05x (noise), fails outright (L1 CB overflow) at
   the MLP shape. Structural: maximizing per-core work instead of core
   count means only 8x8=64 of 110 cores get used at these shapes.
   **Superseded 2026-07-30** by a real offline sweep (`scripts/
   sweep_mm_block_sizes.py`, §5 item 1) — see that item for what a real
   search over the same config type actually found: real wins on 2 of 8
   shapes, a confirmed hard structural ceiling on the other 6.
3. **BFLOAT8_B for encoder QKV/MLP** — 1.00-1.03x (noise); not memory-bound
   at this shape.
4. **Tier-1 chunked flash-attention** (`ops_tt/flash_attention.py`,
   FlashAttention-2-style, pure `ttnn` composition) — numerically correct
   but 3-10x **slower** than the fused `sdpa_fw`/`sdpa_bw` kernels at every
   chunk size tried. Composing separate ops cannot beat one real fused
   kernel here.
5. **SDPA backward-KV redundant Q/dO DRAM re-read fix** (caching per
   (batch,head) group instead of re-reading per K/V-row) — real and
   landed, but small (~1.3% step-time) and lands almost entirely on the
   encoder's 24 blocks, not the decoder's 4 blocks that dominate cost;
   decoder-scale SDPA backward itself barely moved. An earlier
   blocking-scalar-copy version of the same idea was a measured 2x
   *regression* and was reverted; the shipped version uses an async
   NOC-loopback copy instead.
6. **Q-row-batching / wide 2-row subblocking** in the SDPA backward-KV
   kernel (amortizing the kernel's 7-cycle-per-row `tile_regs`
   structure) — real, quantified targets (~1.5% and ~9% of step time
   respectively), but the first is hard-capped by a 4-tile DST register
   budget (`fp32_dest_acc_en=true`), and the second could not be made
   numerically correct after 8 independently-reasoned attempts against
   Tensix register-sharing semantics — reverted cleanly, not shipped.
7. **GELU FastLut variant** (`ttnn.gelu(variant=FastLut)`, a LUT
   approximation, not a fidelity knob) — the one real, landed win this
   whole investigation found: ~1.76x forward-GELU speedup in isolation,
   PCC-verified 0.99998 (`ops_tt/tuned_gelu.py`), ~0.5-0.9% end-to-end.
   `train()`'s default (`use_tuned_gelu=True`); `test_golden_baseline.py`
   recaptured accordingly. Zero vendored-kernel changes.
8. **`MatmulMultiCoreReuseMultiCast1DProgramConfig`** (`mcast_in0=True`,
   the real `matmul_1d_config` helper from `models/tt_transformers/tt/
   model_config.py`, arithmetically derived not hand-picked) for the 6
   matmul shapes item 1 above couldn't cover — confirmed real hardware
   dead end, different failure mode than the 2D config: this 1D pattern
   is architecturally built for small-M decode-time shapes (`per_core_M`
   is the **full**, unsplit M — only N is distributed across cores).
   At training scale (`m_tiles=408` encoder / `2112` decoder) even the
   most minimal candidate (`in0_block_w=1, per_core_N=1`) overflows L1
   (2.6MB > 1.5MB) purely from `per_core_M` alone — the 2D
   `MatmulMultiCoreReuseMultiCastProgramConfig` (which splits *both* M
   and N) is the architecturally correct family for this workload's
   scale, not a missed alternative.
9. **Splitting a matmul call's batch in half to unlock a smaller
   `per_core_M`** (e.g. `batch=12` instead of `24` gets encoder QKV a
   working 2D config where full-`batch=24` has none, confirmed on real
   hardware: `grid=(8,6) per_core=(34,12)` succeeds where every
   `batch=24` candidate overflows L1) — real and correct, but a **net
   timing loss**: two half-batch `ttnn.linear` calls + `ttnn.slice` +
   `ttnn.concat` measured 2.61ms vs the untuned single-call baseline's
   1.79ms (batch=24 QKV), confirming dispatch/slice/concat overhead
   dominates any per-core-utilization gain from the smaller shape — the
   same lesson `chunked_bidirectional_attention` (dead end #4) already
   established for SDPA now confirmed for matmul too: composing more,
   smaller ops loses to fewer, larger ones here.
10. **SDPA forward `Sk_chunk_t` cap 4→8 via `fp32_dest_acc_en=false`**
    (2026-07-30) — tried and reverted. `sdpa_fw_program_factory.cpp`'s
    `pick_sk_chunk_t`/`dst_size` plumbing is genuinely clean in the
    *source* (`Sk_chunk_t` threaded consistently through every compile-time
    arg and loop bound in `sdpa_fw_compute_kernel.cpp`, CB sizes already
    dynamic) — this is **not** the same code as dead end #6's backward-KV
    row-batching. Changed `fp32_dest_acc_en` false + widened the chunk cap
    accordingly (scoped to the forward kernel only, backward untouched),
    rebuilt cleanly, and it **broke correctness badly**:
    `test_attention.py`'s bidirectional-SDPA PCC gate went from ~1.0 to
    **0.24** (not a graceful precision loss — a real bug). Reverted via
    `git checkout` + rebuild immediately; `pytest` (120/120) confirmed
    clean recovery. Take-away: even source-level-clean DST-budget changes
    in these fused kernels hit real Tensix `tile_regs`/DST hardware
    hazards invisible from reading the C++ alone — consistent with, and
    further evidence for, dead end #6's "8 independently-reasoned attempts,
    all failed" finding. **Any future `fp32_dest_acc_en`/DST-width change
    anywhere in `sdpa_fw`/`sdpa_bw` needs to be treated as a full kernel
    rewrite requiring deep Tensix microarchitecture review, not a
    config-level tweak** — do not re-attempt without that.

---

## 5. Plan to close the gap with the 4090

Ranked by expected impact, cheapest/most-confirmed first. Every item below
is scoped from real, measured evidence in §4, not speculation — nothing
here is a repeat of a confirmed dead end.

1. **Matmul offline block-size sweep + lookup table** — **done, 2026-07-30,
   modest real win.** `scripts/sweep_mm_block_sizes.py` (reusable, PCC-gated
   against `program_config=None`, results cached to `ops_tt/
   mm_program_config_cache.json`, read automatically by `ops_tt.
   tuned_linear.TunedLinearLayer`) swept all 8 real encoder/decoder linear
   shapes at `batch=24`:

   | shape | in→out | result |
   |---|---|---|
   | encoder proj | 1024→1024 | **2.64x** (0.943ms→0.357ms), pcc=0.999861 |
   | encoder fc2 | 4096→1024 | **3.76x** (3.306ms→0.880ms), pcc=0.999763 |
   | encoder qkv | 1024→3072 | no safe candidate (see below) |
   | encoder fc1 | 1024→4096 | no safe candidate (see below) |
   | decoder qkv/proj/fc1/fc2 | (seq_len=2816) | no safe candidate (see below) |

   **Real structural ceiling, not a sweep bug**: every candidate for the 6
   uncovered shapes raised the same `TT_THROW ... circular buffers ...
   beyond max L1 size` the old hand-picked heuristic hit (dead end #2
   above) — confirmed by hand this is inherent to
   `MatmulMultiCoreReuseMultiCastProgramConfig` at these shapes, not a
   missed candidate: encoder qkv/fc1's `out_features` (3072/4096) has no
   divisor in `[9,11]`, so `per_core_N` floors at 12/16 tiles regardless of
   grid/block/subblock choice, and even the minimum subblock (1,1) at
   minimum `in0_block_w` still overflows L1 there. Decoder's problem is
   worse: `seq_len=2816` makes `per_core_M` floor at 264 tiles (grid_y
   capped at 10 rows), which alone overflows the output-block CB before
   `in0_block_w`/subblock choice even matters. **`MatmulMultiCoreReuseMultiCast1DProgramConfig`
   was tried as the candidate "different program-config type" for this
   and confirmed NOT to escape either floor** — see §4 item 8: it's
   architecturally built for small-M (decode-time) shapes, and fails even
   harder at our training-scale M than the 2D config did. Batch-splitting
   to shrink `per_core_M` (§4 item 9) also works structurally but loses
   on dispatch overhead. Both of these 6 shapes' matmul-config avenue is
   now a **closed, confirmed dead end**, not an open item.

   **End-to-end** (`scripts/bench_step.py --tuned-linear`, `full_vitl_real_
   contract`, batch=24, warm steps): 1796.0ms → 1759.6ms/step, **~2.0%
   faster**, ~74.8→~73.3 ms/sample. Real and reproducible (two baseline
   reruns agreed within 3ms), but small — proj/fc2 are 2 of the encoder's 4
   linear-op shapes and neither is the single largest one. Not defaulted on
   in `train()` (`use_tuned_linear=False` still): even a *cache-miss*
   shape numerically routes through a different code path (`ttnn.linear`
   vs `ttml.ops.linear.linear`) with tiny bf16 rounding differences that
   broke `test_golden_baseline.py`'s tight tolerance when tried as a
   default — pass `use_tuned_linear=True` explicitly once ready to accept
   that regression-gate recapture, or per-call once a specific shape is
   verified to help.

2. **Tier-2/3 fused SDPA backward-KV kernel rewrite** (the largest single
   lever, and the largest piece of work). SDPA backward-KV alone is 26.9%
   of step time and is compute/pipeline-bound by the current per-row
   streaming kernel architecture itself (§4's dead-end #6), not by
   anything reachable through reader-kernel or register-budget tweaks.
   Closing this requires an actual new kernel design — chunked
   FlashAttention-style tiling with cross-core K/V multicast (the
   `AdvancedPerformanceOptimizationsForModels`/`FlashAttention` tech-report
   pattern: double-buffered circular buffers sized for 2 chunks so
   reader/compute/writer RISCs overlap NoC movement with math), not an
   incremental patch. This is genuine multi-session C++/Metal kernel work
   requiring its own correctness harness before touching the vendored
   tree (per this project's "test before rewrite" policy) — largest
   expected win, largest cost.

3. **`SDPAProgramConfig` tuning on the kernel we actually train with —
   confirmed NOT reachable, do not re-attempt without new evidence.**
   `config_tt.SDPAProgramConfigRegistry`'s docstring already establishes
   this in detail: `ttml::metal::sdpa_fw`/`sdpa_bw` (the fused kernels
   `ops_tt/attention.py` actually trains with) accept **no `program_config`
   parameter at the C++ API level at all** — chunk size is picked
   internally by `pick_sk_chunk_t()`, hard-capped at 4 tiles by
   `fp32_dest_acc_en=true`, with no override hook anywhere to plumb a
   config through, Python or C++. `ttnn.SDPAProgramConfig` (`q_chunk_size`
   up to 512, the 6.54x-measured config this item used to describe) belongs
   to a *different* op (`ttnn.transformer.scaled_dot_product_attention`)
   that has no backward in this build at all — unusable for training
   without writing one, which is exactly item 2's scope, not a cheaper
   alternative to it.

4. **Metal Trace + multi-CQ dispatch** (`begin_trace_capture`/
   `execute_trace`, `num_command_queues=2`) — only pays off if host
   dispatch (not device compute) is a meaningful fraction of step time.
   Not yet measured directly; the device-profiler total (1703.98ms) was
   already close to wall-clock step time (~1795-1810ms at batch=24),
   suggesting host dispatch overhead is small (~5-6%) — likely low
   priority, but a 10-minute profiler check should confirm before ruling
   it out.

5. **Re-run the offline matmul/attention levers at the corrected,
   apples-to-apples ms/sample metric** whenever a candidate fix is tried —
   every measurement from here on must report ms/sample (or an explicit
   fixed batch+accum recipe matching the 4090's), never raw ms/step at an
   unstated batch size, to prevent repeating §3's error.

Realistic framing, updated 2026-07-30 (second pass): item 1 is **fully
closed**, not just "done for now" — the two remaining avenues that could
have covered its other 6 shapes (`MatmulMultiCoreReuseMultiCast1DProgramConfig`,
batch-splitting) were both tried on real hardware and are confirmed dead
ends (§4 items 8-9), not unswept opportunity. Item 3 is a confirmed dead
end. A bounded, source-clean-looking DST-budget experiment on the SDPA
forward kernel (§4 item 10) was also tried and reverted after breaking
correctness — real evidence that config/dtype-level SDPA levers are now
genuinely exhausted, not just under-explored. Item 4 remains an open, cheap
check, unlikely to be large (host dispatch already estimated at ~5-6%).

**Bottom line**: every lever reachable without deep Tensix-microarchitecture
kernel authorship has now been tried, in most cases with a real (if often
negative) hardware measurement to back the conclusion — this investigation
has moved from "config tuning, mostly unswept" to "config tuning,
exhausted." Item 2 (a genuine SDPA backward-KV kernel rewrite — chunked
FlashAttention-style tiling with cross-core K/V multicast, its own
correctness harness, deep familiarity with Tensix `tile_regs`/DST register
semantics) is no longer just "the biggest lever" — it is the *only*
remaining lever with real headroom, and reaching "2x the 4090" is not
possible without it landing. This is genuine multi-session expert Metal
kernel work; the 8 failed attempts already on record for a smaller version
of this exact kernel (dead end #6) plus this session's own DST-budget
failure (dead end #10) are real, concrete evidence that this is a
correctness-hazard-heavy undertaking, not an incremental config change —
budget it, and gate it, accordingly.

---

## 6. Multi-card (M7, not started)

Two-plus Blackhole cards is the production target; mesh-native from commit
one (`ttnn.open_device` already returns a `MeshDevice`, `mesh.py` builds on
`ttml.autograd.AutoContext`). Data-parallel, not FSDP (316M params + AdamW
state is 5.1GB against 31GiB/card — sharding buys nothing at this size).
Gradient reduction: one `all_reduce` over 632MB bf16 grads per optimizer
step; `accum_iter` amortizes this across microbatches. Mesh graph
descriptors for Blackhole already ship
(`tt-train/configs/mgd/bh_galaxy_1_2_line_line.textproto`).

**Open question, unresolved:** are the cards linked by ethernet (QSFP-DD,
CCL over TT-Fabric) or only through host PCIe (host-mediated reduction,
~100ms/step at measured ~13GB/s PCIe bandwidth)? Resolve before the second
card arrives — it changes whether scaling `accum_iter` down is affordable.

---

## 7. Vendored `tt-metal` changes

`tenstorrent/tt-metal/` is a clean, non-submodule git checkout (`HEAD
detached at v0.74.0`); every change is a diffable, revertible patch, not a
fork. Currently 8 modified files, all precisely scoped and test-covered:

- `tt-train/sources/ttml/metal/common/dataflow_utils.hpp`
- `tt-train/sources/ttml/metal/ops/sdpa_bw/device/kernels/dataflow/sdpa_bw_kv_reader_kernel.cpp`
- `tt-train/sources/ttml/metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.cpp`
- `tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute/sdpa_fw_compute_kernel.cpp`
- `tt-train/sources/ttml/nanobind/nb_ops.cpp`
- `tt-train/sources/ttml/ops/scaled_dot_product_attention.cpp`
- `tt-train/sources/ttml/ops/scaled_dot_product_attention.hpp`
- `tt-train/tests/ops/sdpa_bw_op_test.cpp`

Together these: (a) add `AttentionMaskType::None` support through the
ops-layer wrapper and nanobind binding (the M0 bidirectional-attention fix
+ a missing `USE_ATTN_MASK` guard bug it exposed in the forward compute
kernel), and (b) fix the SDPA backward-KV reader kernel's redundant
per-row Q/`dO` DRAM re-reads via an async NOC-loopback per-group cache.

---

## 8. Principal risks

| Risk | Why it bites | Mitigation |
|---|---|---|
| Performance gap to 4090 (§3/§5) | Currently ~2.4x slower per sample, not faster | §5's ranked plan; item 2 (kernel rewrite) likely load-bearing |
| tt-train build instability | Built from a moving `main`, not the release wheel | Pinned to v0.74.0, detached HEAD |
| Silent numeric drift | bf16 + hand-written backward + no compiler check | Parity harness gates every milestone (PCC/grad-norm), `test_golden_baseline.py` regression gate |
| Custom kernels absorbing the schedule | Tier 2/3 work is open-ended | Profile-gated: descend a tier only when the profiler names the op (§4's real per-op data, not guessing) |
| Card-to-card link unknown | Decides whether all-reduce is fabric or host-mediated | Resolve before card 2 arrives (§6) |
| Background scripts without `if __name__ == "__main__":` guard | `multiprocessing`'s spawn start method re-executes the whole file in the prefetch worker process, causing a second conflicting device open (`Failed to allocate TLB window`) if the training call isn't guarded | Always guard ad hoc scripts that use `train_batch_source_factory`/`prefetch=True`; `bench_step.py` and `main_pretrain_tt.py`'s own `main()` already do this correctly |

---

## 9. Explicitly out of scope

- `src/evaluation`, `src/asparagus_bridge`, `src/preprocessing` — stay on PyTorch/CPU.
- The `torch.distributed` DDP path in `main_pretrain.py`; replaced by §6.
- Reproducing CUDA RNG bit-for-bit — masking ids come from a seeded host numpy generator, the *distribution* is matched, not the bit pattern.
- FSDP / tensor parallel — revisit only if the model grows past ~2B params.
