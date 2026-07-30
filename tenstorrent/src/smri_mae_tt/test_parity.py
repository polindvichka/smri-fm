"""Tests for `parity.py`, the shared parity-testing harness
(TENSTORRENT_PORT.md section 7: "Build this at M2 and keep it").

This file tests the harness itself, not the model: `compare()`/`pearson_corr`
give the right numbers on inputs whose right answer is known by construction
(identical tensors, and tensors perturbed by a known amount), `ParityReport`
gates and fails loud with all three metrics in the message, and
`make_frozen_fixture`/`FROZEN_FIXTURE` are actually deterministic and
actually independent across `seed_offset`s -- the properties every other
parity test in this package now depends on.

`pytest.importorskip("ttml")` matches every other test file in this package
(`parity.py` imports `ttml`/`ttnn` at module level, same as `modules_tt.py`);
none of the tests below open a device, since the properties under test are
all host-side numpy/dataclass behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ttml")

from smri_mae_tt import parity as P  # noqa: E402


# ---------------------------------------------------------------------------
# pearson_corr / compare metrics
# ---------------------------------------------------------------------------


def test_compare_identical_tensors_gives_perfect_parity():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((4, 5, 6)).astype(np.float32)

    report = P.compare("identical", a, a.copy())

    assert report.pcc == pytest.approx(1.0, abs=1e-9)
    assert report.max_abs_err == pytest.approx(0.0, abs=1e-9)
    assert report.rel_frob_err == pytest.approx(0.0, abs=1e-9)


def test_compare_perturbed_tensors_gives_nontrivial_pcc_and_nonzero_error():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((64, 32)).astype(np.float32)
    noise = rng.standard_normal(a.shape).astype(np.float32) * 0.05
    b = a + noise

    # Small, correlated noise: high but not perfect PCC, and it must clear
    # a loose gate while failing a strict one -- exercising both branches of
    # assert_gate, not just the passing one.
    report = P.compare("perturbed", a, b, pcc_gate=0.9)
    assert 0.9 <= report.pcc < 1.0
    assert report.max_abs_err > 0.0
    assert report.rel_frob_err > 0.0

    with pytest.raises(AssertionError, match="below gate"):
        P.compare("perturbed strict", a, b, pcc_gate=1.0 - 1e-12)


def test_compare_uncorrelated_tensors_gives_low_pcc():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(4096).astype(np.float32)
    b = rng.standard_normal(4096).astype(np.float32)  # independent draw

    report = P.compare("uncorrelated", a, b, pcc_gate=-1.0)
    assert abs(report.pcc) < 0.1


def test_compare_rejects_shape_mismatch():
    a = np.zeros((2, 3))
    b = np.zeros((2, 4))
    with pytest.raises(ValueError, match="element count mismatch"):
        P.compare("mismatch", a, b)


def test_compare_rejects_all_zero_reference():
    a = np.zeros(8)
    b = np.ones(8)
    with pytest.raises(ValueError, match="all-zero"):
        P.compare("zero ref", a, b)


def test_pearson_corr_rejects_constant_input():
    a = np.full(16, 3.0)
    b = np.arange(16, dtype=np.float64)
    with pytest.raises(ValueError, match="undefined"):
        P.pearson_corr(a, b)


def test_parity_report_assert_gate_message_has_all_three_metrics():
    report = P.ParityReport(name="foo", pcc=0.5, max_abs_err=1.25, rel_frob_err=0.75)
    with pytest.raises(AssertionError) as excinfo:
        report.assert_gate(pcc_gate=0.999)
    msg = str(excinfo.value)
    assert "foo" in msg
    assert "0.5" in msg
    assert "1.25" in msg or "1.250000e+00" in msg
    assert "0.75" in msg or "7.500000e-01" in msg


def test_parity_report_dict_like_access_matches_attributes():
    report = P.ParityReport(name="foo", pcc=0.987, max_abs_err=0.01, rel_frob_err=0.02)
    assert report["pcc"] == report.pcc
    assert report["max_abs_err"] == report.max_abs_err
    assert report["rel_frob_err"] == report.rel_frob_err
    assert report.as_dict() == {"pcc": 0.987, "max_abs_err": 0.01, "rel_frob_err": 0.02}


# ---------------------------------------------------------------------------
# Frozen fixture
# ---------------------------------------------------------------------------


def test_frozen_fixture_default_matches_documented_dims():
    contract = P.FROZEN_FIXTURE.contract
    assert (contract.batch, contract.l_vis, contract.q_pred) == (2, 31, 32)
    assert (contract.embed_dim, contract.encoder_heads) == (64, 2)
    assert (contract.decoder_embed_dim, contract.decoder_heads) == (32, 1)
    assert contract.class_token is True
    # head_dim == 32 (the tile size) on both sides -- the load-bearing
    # constraint documented on FrozenFixture and modules_tt.Attention.
    assert contract.embed_dim // contract.encoder_heads == 32
    assert contract.decoder_embed_dim // contract.decoder_heads == 32
    # encoder/decoder sequence lengths tile-aligned.
    assert contract.encoder_seq_len % 32 == 0
    assert contract.decoder_seq_len % 32 == 0
    assert P.FROZEN_FIXTURE.encoder_depth == 2
    assert P.FROZEN_FIXTURE.decoder_depth == 1


def test_make_frozen_fixture_is_deterministic():
    fx1 = P.make_frozen_fixture(seed_offset=7)
    fx2 = P.make_frozen_fixture(seed_offset=7)
    assert fx1.contract == fx2.contract
    assert fx1.param_seed == fx2.param_seed == fx2.data_seed - 1

    draw1 = fx1.param_rng().standard_normal(32)
    draw2 = fx2.param_rng().standard_normal(32)
    np.testing.assert_array_equal(draw1, draw2)

    data1 = fx1.data_rng().standard_normal(32)
    data2 = fx2.data_rng().standard_normal(32)
    np.testing.assert_array_equal(data1, data2)


def test_make_frozen_fixture_seed_offsets_are_independent():
    fx_a = P.make_frozen_fixture(seed_offset=0)
    fx_b = P.make_frozen_fixture(seed_offset=100)
    assert fx_a.param_seed != fx_b.param_seed
    assert fx_a.data_seed != fx_b.data_seed

    draw_a = fx_a.param_rng().standard_normal(32)
    draw_b = fx_b.param_rng().standard_normal(32)
    assert not np.allclose(draw_a, draw_b)

    # param stream and data stream within one fixture are also independent
    # of each other (different seeds), not the same stream read twice.
    param_draw = fx_a.param_rng().standard_normal(32)
    data_draw = fx_a.data_rng().standard_normal(32)
    assert not np.allclose(param_draw, data_draw)


def test_make_frozen_fixture_depth_override():
    fx = P.make_frozen_fixture(seed_offset=0, encoder_depth=4, decoder_depth=2)
    assert fx.encoder_depth == 4
    assert fx.decoder_depth == 2
    # Depth override doesn't perturb the frozen contract itself.
    assert fx.contract == P.FROZEN_FIXTURE.contract
