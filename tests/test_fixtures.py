"""Sanity tests that the .mat fixtures load with the shapes the README claims."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from qge.io import _validate

J, N = 26, 50


def test_raw_shapes(raw):
    """README spec: J=26 sectors, N=50 regions; xbilat stacked as (J*N, N)."""
    assert raw.xbilat.shape == (J * N, N), raw.xbilat.shape
    assert raw.L_j_n.shape == (J, N), raw.L_j_n.shape
    assert raw.IO.shape == (J, J), raw.IO.shape
    assert raw.gamma.shape == (J, N), raw.gamma.shape
    assert raw.B.shape == (N,), raw.B.shape
    assert raw.alphas.shape == (J, N), raw.alphas.shape
    assert raw.io.shape == (N,), raw.io.shape


def test_raw_finite(raw):
    """No NaNs / infs in any raw input — would short-circuit later debugging."""
    import numpy as np

    for name in ("xbilat", "L_j_n", "IO", "gamma", "B", "alphas", "io"):
        arr = getattr(raw, name)
        assert np.isfinite(arr).all(), f"{name} has non-finite entries"


def test_validate_rejects_alphas_not_summing_to_one(raw):
    """A region whose final-demand shares don't sum to 1 is rejected."""
    bad_alphas = raw.alphas.copy()
    bad_alphas[0, 0] += 0.5  # Alabama's shares no longer sum to 1
    with pytest.raises(ValueError, match="alphas column"):
        _validate(replace(raw, alphas=bad_alphas))


def test_validate_rejects_zero_xbilat_row(raw):
    """A (sector, destination) with zero total expenditure is rejected."""
    bad_xbilat = raw.xbilat.copy()
    bad_xbilat[0, :] = 0.0
    with pytest.raises(ValueError, match="xbilat row"):
        _validate(replace(raw, xbilat=bad_xbilat))


def test_validate_rejects_wrong_shape(raw):
    """Shape mismatch with sector/region labels is rejected."""
    truncated = raw.L_j_n[:-1, :]  # drop a sector
    with pytest.raises(ValueError, match="L_j_n.*shape"):
        _validate(replace(raw, L_j_n=truncated))


def test_handles_zero_production_cells(raw):
    """A (sector, region) cell with truly zero gross output should produce
    Ljn = 0 (not NaN), and aggregate quantities should be finite.

    This case doesn't arise in CPRHS data (every cell has positive output
    via measurement quirks) but is expected for Canadian data — e.g. PEI
    has no petroleum production.
    """
    from qge.models.benchmark import compute_baseline, compute_regional_shock

    sec, reg = 4, 10  # Petroleum and Coal, Hawaii — a CPRHS-positive cell we'll zero out
    J = len(raw.sectors)
    xbilat_3d = raw.xbilat.reshape(J, raw.N, raw.N).copy()
    xbilat_3d[sec, :, reg] = 0.0  # the source state produces nothing in this sector
    raw_zero = replace(raw, xbilat=xbilat_3d.reshape(J * raw.N, raw.N))

    baseline = compute_baseline(raw=raw_zero, tol=1e-10)
    assert baseline.Ljn[sec, reg] == 0.0
    assert not np.isnan(baseline.Ljn).any()

    shock = compute_regional_shock(region=4, raw=raw_zero, baseline=baseline, tol=1e-10)
    assert not np.isnan(shock.Ljn_hat).any()
    assert np.isfinite(shock.TFP_hat) and np.isfinite(shock.GDP_hat)


def test_benchmark_golden_loads(benchmark_golden):
    """Smoke-test: the Benchmark golden state has the variables we need."""
    expected = {
        "J", "N", "T", "Ln_RS", "xbilat_RS", "B", "G", "gamma",
        "alphas", "VAR_RS", "VAL_RS", "io",
    }
    missing = expected - set(benchmark_golden)
    assert not missing, f"Missing in Base_Year_Benchmark.mat: {sorted(missing)}"
