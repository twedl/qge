"""Sanity tests that the .mat fixtures load with the shapes the README claims."""

from __future__ import annotations

J, N = 26, 50


def test_raw_shapes(raw):
    """README spec: J=26 sectors, N=50 regions; xbilat stacked as (J*N, N)."""
    assert raw.xbilat.shape == (J * N, N), raw.xbilat.shape
    assert raw.L_j_n.shape == (J, N), raw.L_j_n.shape
    assert raw.IO.shape == (J, J), raw.IO.shape
    assert raw.gamma.shape == (J, N), raw.gamma.shape
    assert raw.B.shape == (J, N), raw.B.shape
    assert raw.alphas.shape == (J, N), raw.alphas.shape
    assert raw.io.shape == (N,), raw.io.shape


def test_raw_finite(raw):
    """No NaNs / infs in any raw input — would short-circuit later debugging."""
    import numpy as np

    for name in ("xbilat", "L_j_n", "IO", "gamma", "B", "alphas", "io"):
        arr = getattr(raw, name)
        assert np.isfinite(arr).all(), f"{name} has non-finite entries"


def test_benchmark_golden_loads(benchmark_golden):
    """Smoke-test: the Benchmark golden state has the variables we need."""
    expected = {
        "J", "N", "T", "Ln_RS", "xbilat_RS", "B", "G", "gamma",
        "alphas", "VAR_RS", "VAL_RS", "io",
    }
    missing = expected - set(benchmark_golden)
    assert not missing, f"Missing in Base_Year_Benchmark.mat: {sorted(missing)}"
