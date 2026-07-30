"""Tests for `mesh.MeshContext` and `mesh.bundled_mgd_path`.

`pytest.importorskip("ttml")` is the one legitimate skip (environment
availability, not a fallback masking a bug -- see `feedback_fail_fast`
memory), matching `ops_tt/test_attention.py`.

This host has exactly one Blackhole card, so only the 1x1-mesh path is
exercised on device; `bundled_mgd_path` is pure host-side lookup and is
tested for shapes this host cannot actually open, per
TENSTORRENT_PORT.md section 5.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("ttml")

import ttml  # noqa: E402
import ttnn  # noqa: E402

from smri_mae_tt.mesh import MeshContext, bundled_mgd_path  # noqa: E402


def test_open_close_reports_1x1_mesh_device():
    """`AutoContext.open_device()` returns a genuine `MeshDevice` even with
    no card beyond this host's single one -- TENSTORRENT_PORT.md section 5's
    claim, confirmed here rather than merely asserted.
    """
    with MeshContext() as mesh:
        assert isinstance(mesh.device, ttnn.MeshDevice)
        topo = mesh.topology
        assert topo.shape == (1, 1)
        assert topo.num_devices == 1
        assert topo.is_multi_device is False
        assert len(topo.device_ids) == 1


def test_double_open_raises():
    mesh = MeshContext()
    mesh.open()
    try:
        with pytest.raises(RuntimeError, match="twice"):
            mesh.open()
    finally:
        mesh.close()


def test_methods_before_open_raise():
    mesh = MeshContext()
    with pytest.raises(RuntimeError, match="not open"):
        _ = mesh.device
    with pytest.raises(RuntimeError, match="not open"):
        mesh.all_reduce_grad(object())


def test_close_without_open_is_a_noop():
    mesh = MeshContext()
    mesh.close()  # must not raise


def test_all_reduce_grad_is_identity_on_1x1_mesh():
    """`all_reduce_grad` must return the tensor unchanged on this host's
    1x1 mesh -- and must NOT forward to `ttml.ops.distributed.all_reduce`,
    which raises at world size 1 (see mesh.py's module docstring; confirmed
    directly in `test_raw_all_reduce_raises_on_1x1_mesh` below). This test
    checks both the identity behavior and, separately, that it is achieved
    without the underlying op ever being called successfully (there would be
    no successful call to check the output of).
    """
    with MeshContext() as mesh:
        data = np.arange(32 * 32, dtype=np.float32).reshape(1, 1, 32, 32)
        t = ttml.autograd.Tensor.from_numpy(data, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.FLOAT32)

        out = mesh.all_reduce_grad(t)

        assert out is t, "all_reduce_grad must be a pure pass-through (identity) on a 1x1 mesh"
        np.testing.assert_array_equal(np.asarray(out.to_numpy(ttnn.DataType.FLOAT32)), data)


def test_raw_all_reduce_raises_on_1x1_mesh():
    """Regression guard for the finding documented in mesh.py: calling
    `ttml.ops.distributed.all_reduce` directly on a 1x1 mesh raises rather
    than behaving as an identity. If this ever stops raising (a future
    `ttml` fixing the single-device case), `MeshContext.all_reduce_grad`'s
    early-return branch becomes dead code but still correct -- this test
    existing is what would tell us to reconsider it.
    """
    with MeshContext() as mesh:
        assert not mesh.is_multi_device
        data = np.ones((1, 1, 32, 32), dtype=np.float32)
        t = ttml.autograd.Tensor.from_numpy(data, layout=ttnn.Layout.TILE, new_type=ttnn.DataType.FLOAT32)
        with pytest.raises(RuntimeError, match="single device"):
            ttml.ops.distributed.all_reduce(t)


def test_bundled_mgd_path_known_shape_exists_on_disk():
    path = bundled_mgd_path((1, 2), arch="blackhole")
    assert path is not None
    assert os.path.isfile(path)
    assert path.endswith("bh_galaxy_1_2_line_line.textproto")


def test_bundled_mgd_path_unknown_shape_returns_none():
    assert bundled_mgd_path((1, 1), arch="blackhole") is None
    assert bundled_mgd_path((3, 7), arch="blackhole") is None
    assert bundled_mgd_path((1, 2), arch="some_future_arch") is None
