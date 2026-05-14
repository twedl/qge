"""Golden-master test for the Benchmark baseline equilibrium.

Compares qge.models.benchmark.compute_baseline() against the .mat snapshot
shipped with the replication kit (`Base_Year_Benchmark.mat`).
"""

from __future__ import annotations

import numpy as np
import pytest

from qge.models.benchmark import compute_baseline


@pytest.fixture(scope="module")
def baseline():
    return compute_baseline()


def _flat(x):
    return np.asarray(x).ravel()


def test_Ln_matches(baseline, benchmark_golden):
    np.testing.assert_allclose(
        _flat(baseline.Ln), _flat(benchmark_golden["Ln_RS"]),
        rtol=1e-6, atol=1e-10,
    )


def test_xbilat_matches(baseline, benchmark_golden):
    np.testing.assert_allclose(
        baseline.xbilat, benchmark_golden["xbilat_RS"],
        rtol=1e-6, atol=1e-8,
    )


def test_VAL_matches(baseline, benchmark_golden):
    np.testing.assert_allclose(
        _flat(baseline.VAL), _flat(benchmark_golden["VAL_RS"]),
        rtol=1e-6, atol=1e-8,
    )


def test_VAR_matches(baseline, benchmark_golden):
    np.testing.assert_allclose(
        _flat(baseline.VAR), _flat(benchmark_golden["VAR_RS"]),
        rtol=1e-6, atol=1e-8,
    )
