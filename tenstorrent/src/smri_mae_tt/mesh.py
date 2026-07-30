"""`MeshDevice` setup, gradient all-reduce, and rank/mesh-shape plumbing.

TENSTORRENT_PORT.md section 5 ("Multi-card from commit one"): open a
`MeshDevice` from the start, even as a 1x1 mesh on this box, because
"`ttnn.open_device` already returns a `MeshDevice` here, so this costs
nothing today and avoids a rewrite later." Confirmed empirically on this
host: `ttml.autograd.AutoContext.get_instance().open_device()` (no args,
the same call `ops_tt/attention.py`'s M0-gate test uses) then `.get_device()`
returns `<class 'ttnn._ttnn.multi_device.MeshDevice'>` with
`.shape == MeshShape([1, 1])` -- a genuine (degenerate) mesh, not a bare
single-device handle. This module works at that `AutoContext` level (not
the higher-level `ttml.Mesh`/`ttml.open_device_mesh`/`ttml.mesh()` API used
by `tt-train/tests/python/test_fsdp.py` for multi-axis DP/TP/FSDP meshes)
because that is the level the rest of this port's M0-gate code
(`ops_tt/attention.py`, `ops_tt/test_attention.py`) already standardizes on,
and section 5 explicitly rules FSDP/tensor-parallel out for this model
("Data parallel, not FSDP ... keep `ttml.fsdp.fully_shard` in reserve for a
future ViT-H, not for this model") -- there is no tensor-parallel/FSDP axis
here to name, only plain data parallelism across whole-mesh replicas.

## The one non-obvious finding here: `all_reduce` is not a no-op on 1x1

The task briefing for this module assumed `ttml.ops.distributed.all_reduce`
would behave as an identity on a single-device mesh. It does not: it
*raises*. `ttml::ttnn_fixed::distributed::all_reduce`
(`tt-train/sources/ttml/ttnn_fixed/distributed/ttnn_ops.cpp:114-119`) starts
with

    auto num_devices = mesh_device->num_devices();
    if (num_devices == 1U) {
        throw std::logic_error("All reduce should not be called for a single device case");
    }

confirmed empirically (see `test_mesh.py`): calling
`ttml.ops.distributed.all_reduce(t)` on this host's 1x1 mesh raises
`RuntimeError: All reduce should not be called for a single device case`.
So "gradient reduction behind an interface" (section 5) cannot mean "always
forward to `ttml.ops.distributed.all_reduce`" -- on today's single card that
call is a hard error, not a slow/no-op path. `MeshContext.all_reduce_grad`
below is the thin interface the doc asks for: it forwards to the real CCL op
only when the mesh actually has more than one device, and returns the input
tensor unchanged otherwise (a true Python-level no-op, matching the
mathematical identity all_reduce *would* have to compute at world size 1, but
implemented as a skip rather than a call, since the underlying op refuses to
run at size 1). When a second card arrives this same call site starts
exercising the real CCL path with no caller-visible change.

## Mesh graph descriptors (future 1x2, not exercised on this host)

`tt-train/configs/mgd/bh_galaxy_1_2_line_line.textproto` is the shipped 1x2
Blackhole layout, selected via the `TT_MESH_GRAPH_DESC_PATH` env var
(TENSTORRENT_PORT.md section 5; mechanics documented in
`tt-train/tests/python/test_fsdp.py:95-118`, `_ensure_mgd_path`).
`bundled_mgd_path` below resolves that path for a given mesh shape without
opening any device, so it is testable on this single-card host; actually
opening a >(1,1) mesh is not (this host has one card) and is therefore not
attempted anywhere in this module or its tests, per
TENSTORRENT_PORT.md's "open question to resolve before the second card
arrives" and the instruction not to build multi-card code paths that cannot
be exercised today.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import ttml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tt-metal", "tt-train"))

# (arch, mesh_shape) -> bundled mesh graph descriptor. Only entries that
# actually exist under tt-train/configs/mgd are listed; extend this table
# (and TENSTORRENT_PORT.md section 5's cabling note) once a second card's
# topology (ethernet vs. host-PCIe-mediated) is known.
_MGD_FOR_ARCH_AND_SHAPE = {
    ("blackhole", (1, 2)): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_1_2_line_line.textproto"),
    ("blackhole", (2, 1)): os.path.join(_REPO_ROOT, "configs", "mgd", "bh_galaxy_2_1_line_line.textproto"),
}


def bundled_mgd_path(mesh_shape: Sequence[int], arch: str = "blackhole") -> Optional[str]:
    """Path to the bundled mesh graph descriptor for `(arch, mesh_shape)`, or
    `None` if none is bundled for that combination. Pure host-side lookup --
    does not touch a device, so it is exercisable on this single-card host
    even though the meshes it describes are not.
    """
    path = _MGD_FOR_ARCH_AND_SHAPE.get((arch, tuple(mesh_shape)))
    if path is not None and not os.path.isfile(path):
        raise FileNotFoundError(
            f"bundled_mgd_path({mesh_shape!r}, {arch!r}) resolved to {path}, but that file does not "
            "exist -- tt-train/configs/mgd/ layout changed underneath this table."
        )
    return path


@dataclass(frozen=True)
class MeshTopology:
    """A snapshot of the open mesh's shape/device-id plumbing (doc section 5:
    "Rank/mesh-shape plumbing"). Minimal by design -- this host is a 1x1
    mesh today; multi-host rank (via `AutoContext.get_distributed_context()`)
    is out of scope until a second card's link type is known (section 5's
    "open question").
    """

    shape: tuple[int, ...]
    device_ids: tuple[int, ...]

    @property
    def num_devices(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def is_multi_device(self) -> bool:
        return self.num_devices > 1


class MeshContext:
    """Owns the process's `ttml.autograd.AutoContext` `MeshDevice` for the
    lifetime of training, and is the single call site for gradient
    all-reduce (doc section 5: "Gradient reduction behind an interface").

    Usage (mirrors `ops_tt/test_attention.py`'s open/close pattern):

        with MeshContext() as mesh:
            ...
            reduced = mesh.all_reduce_grad(grad_tensor)

    `mesh_shape`/`device_ids` default to `None`, which `AutoContext.open_device`
    itself defaults to a 1x1 mesh with auto-selected device ids
    (`nb_autograd.cpp:247-273`) -- the exact call `ops_tt/test_attention.py`
    already uses. Passing an explicit larger shape is accepted (the API
    shape here is not hardcoded to "1 device", per TENSTORRENT_PORT.md's
    "mesh is not something to retrofit" -- section 6) but is untested on
    this host; see `bundled_mgd_path` for the env var such a shape would
    need, and set it before calling `open()`.
    """

    def __init__(
        self,
        mesh_shape: Optional[Sequence[int]] = None,
        device_ids: Optional[Sequence[int]] = None,
    ) -> None:
        self._mesh_shape_request = tuple(mesh_shape) if mesh_shape is not None else None
        self._device_ids_request = tuple(device_ids) if device_ids is not None else None
        self._opened = False

    def open(self) -> "MeshContext":
        if self._opened:
            raise RuntimeError("MeshContext.open() called twice without an intervening close()")
        ctx = ttml.autograd.AutoContext.get_instance()
        ctx.open_device(
            mesh_shape=list(self._mesh_shape_request) if self._mesh_shape_request is not None else None,
            device_ids=list(self._device_ids_request) if self._device_ids_request is not None else None,
        )
        self._opened = True
        return self

    def close(self) -> None:
        if not self._opened:
            return
        ttml.autograd.AutoContext.get_instance().close_device()
        self._opened = False

    def __enter__(self) -> "MeshContext":
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _require_open(self):
        if not self._opened:
            raise RuntimeError("MeshContext is not open; call .open() or use it as a context manager first")

    @property
    def device(self):
        """The underlying `ttnn.MeshDevice` (`AutoContext.get_device()`)."""
        self._require_open()
        return ttml.autograd.AutoContext.get_instance().get_device()

    @property
    def topology(self) -> MeshTopology:
        self._require_open()
        dev = self.device
        return MeshTopology(shape=tuple(dev.shape), device_ids=tuple(dev.get_device_ids()))

    @property
    def is_multi_device(self) -> bool:
        return self.topology.is_multi_device

    def all_reduce_grad(
        self,
        tensor: "ttml.autograd.Tensor",
        *,
        noop_backward: bool = False,
        cluster_axis: Optional[int] = None,
    ) -> "ttml.autograd.Tensor":
        """Gradient all-reduce, single call site regardless of mesh size.

        On this host's 1x1 mesh, `ttml.ops.distributed.all_reduce` itself
        raises (see module docstring) -- forwarding to it unconditionally
        would make every single-card training step fail. This returns
        `tensor` unchanged when there is only one device (the mathematically
        correct result of an all-reduce across a world of size 1, computed
        by skipping the op rather than calling it), and forwards to the real
        CCL `ttml.ops.distributed.all_reduce` otherwise -- doc section 5:
        "accum_iter: 5 ... already amortizes that reduction", i.e. this is
        meant to be called once per optimizer step, not per micro-batch.
        """
        self._require_open()
        if not self.is_multi_device:
            return tensor
        return ttml.ops.distributed.all_reduce(tensor, noop_backward=noop_backward, cluster_axis=cluster_axis)
