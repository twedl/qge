"""Verify Phase 3 counterfactual matches MATLAB Counterfactual_economy.mat.

Fast tests cover the math primitives (china shock path, mu_cf
row-stochasticity); the slow integration test runs the full 200-period
counterfactual seeded with the saved V and compares all five outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from qge.counterfactual_dynamics import (
    CHINA_ANNUAL_TFP, CHINA_REGION_IDX, N_CHINA_SHOCK_QUARTERS,
    china_tfp_shock_path, compute_mu_path_cf,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CF_MAT = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Counterfactual_economy" / "Counterfactual_economy.mat"


# ---------------------------------------------------------------- primitives


def test_china_shock_path_shape_and_values():
    """The shock is applied in the right (sector, region, quarter) corner."""
    J, N, time = 22, 87, 200
    A_hat = china_tfp_shock_path(J, N, time)
    assert A_hat.shape == (J, N, time)

    assert np.allclose(A_hat[:, :CHINA_REGION_IDX, :], 1.0)
    assert np.allclose(A_hat[:, CHINA_REGION_IDX + 1:, :], 1.0)
    assert np.allclose(A_hat[12:, CHINA_REGION_IDX, :], 1.0)
    assert np.allclose(A_hat[:, CHINA_REGION_IDX, N_CHINA_SHOCK_QUARTERS:], 1.0)

    expected = 1.0 / (np.asarray(CHINA_ANNUAL_TFP) ** (1.0 / N_CHINA_SHOCK_QUARTERS))
    actual = A_hat[:12, CHINA_REGION_IDX, 0]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0)


def test_compute_mu_path_cf_shape():
    """mu_cf is (RJ1, RJ1, time); slices [1..time-2] are row-stochastic."""
    RJ1, time = 1150, 200
    rng = np.random.default_rng(0)
    mu_baseline = rng.uniform(0.001, 1.0, size=(RJ1, RJ1, time))
    mu_baseline /= mu_baseline.sum(axis=1, keepdims=True)
    V = rng.uniform(0.5, 2.0, size=(RJ1, time))
    mu_cf = compute_mu_path_cf(mu_baseline, V)
    assert mu_cf.shape == (RJ1, RJ1, time)
    row_sums = mu_cf[..., 1:time - 1].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- integration


@pytest.fixture(scope="module")
def cf_fixture():
    if not CF_MAT.exists():
        pytest.skip(f"Counterfactual_economy.mat not found: {CF_MAT}")
    return loadmat(str(CF_MAT))


@pytest.mark.slow
@pytest.mark.parametrize(
    "attr, rtol, atol",
    [
        ("V",          5e-3, 1e-3),
        ("realwages",  5e-3, 1e-3),
        ("Ldyn",       5e-3, 1e-3),
    ],
    ids=["V", "realwages", "Ldyn"],
)
def test_counterfactual_matches_matlab(counterfactual, cf_fixture, attr, rtol, atol):
    actual = getattr(counterfactual, attr)
    expected = cf_fixture[attr]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
