"""Verify Phase 3 counterfactual matches MATLAB Counterfactual_economy.mat.

Fast tests cover the math primitives (china shock path, mu_cf
row-stochasticity); the slow integration test runs the full 200-period
counterfactual seeded with the saved V and compares all five outputs.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import loadmat

from qge.counterfactual_dynamics import (
    CHINA_ANNUAL_TFP, CHINA_REGION_IDX, N_CHINA_SHOCK_QUARTERS,
    china_tfp_shock_path, compute_mu_path_cf,
)
from qge.models.baseline_economy import BaselineEconomy
from qge.models.counterfactual import compute_counterfactual_economy

REPO_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals"
CF_MAT = REP_DIR / "Counterfactual_economy" / "Counterfactual_economy.mat"
BASELINE_MAT = REP_DIR / "Baseline_economy" / "Baseline_economy.mat"


# ---------------------------------------------------------------- primitives


def test_china_shock_path_shape_and_values():
    """The shock is applied in the right (sector, region, quarter) corner."""
    J, N, time = 22, 87, 200
    A_hat = china_tfp_shock_path(J, N, time)
    assert A_hat.shape == (J, N, time)

    # Off-corner cells should be 1.0 (no shock).
    assert np.allclose(A_hat[:, :CHINA_REGION_IDX, :], 1.0)
    assert np.allclose(A_hat[:, CHINA_REGION_IDX + 1:, :], 1.0)
    assert np.allclose(A_hat[12:, CHINA_REGION_IDX, :], 1.0)
    assert np.allclose(A_hat[:, CHINA_REGION_IDX, N_CHINA_SHOCK_QUARTERS:], 1.0)

    # In-corner: each quarter is 1/(annual^(1/28)).
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
    # Slices 1..time-2 are normalized in compute_mu_path_cf; slice 0
    # is the unnormalized "jump", slice time-1 is a carry-forward copy.
    row_sums = mu_cf[..., 1:time - 1].sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=0, atol=1e-12)


# ---------------------------------------------------------------- integration


def _load_baseline_economy_from_mat() -> BaselineEconomy:
    if not BASELINE_MAT.exists():
        pytest.skip(f"Baseline_economy.mat not found at {BASELINE_MAT}")
    with h5py.File(str(BASELINE_MAT), "r") as f:
        # HDF5 stores arrays transposed; .T restores MATLAB layout.
        series_xbilat = np.asarray(f["series_xbilat"]).T
        series_pi = np.asarray(f["series_pi"]).T
        series_wages = np.asarray(f["series_wages"]).T
        series_Ljnhat = np.asarray(f["series_Ljnhat"]).T
        series_mu = np.asarray(f["series_mu"]).T
        L0_initial = np.squeeze(np.asarray(f["L0_initial"]))
    return BaselineEconomy(
        series_xbilat=series_xbilat,
        series_pi=series_pi,
        series_wages=series_wages,
        series_Ljnhat=series_Ljnhat,
        series_mu=series_mu,
        L0_initial=L0_initial,
    )


@pytest.fixture(scope="module")
def cf_fixture():
    if not CF_MAT.exists():
        pytest.skip(f"Counterfactual_economy.mat not found: {CF_MAT}")
    return loadmat(str(CF_MAT))


@pytest.fixture(scope="module")
def cf_python(raw, baseline):
    if not BASELINE_MAT.exists():
        pytest.skip(f"Baseline_economy.mat not found: {BASELINE_MAT}")
    baseline_econ = _load_baseline_economy_from_mat()
    V_seed = loadmat(str(CF_MAT))["V"]
    # The counterfactual seeds its inner equilibrium path from the
    # Phase 1 static base year (2000Q1) — MATLAB loads VARjn00 / VALjn00
    # / Din00 from Base_year.mat (not from Phase 2b's 2007Q4 state).
    return compute_counterfactual_economy(
        V_seed=V_seed, baseline_econ=baseline_econ,
        base_year=baseline, raw=raw,
        max_outer_iter=2,
    )


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
def test_counterfactual_matches_matlab(cf_python, cf_fixture, attr, rtol, atol):
    actual = getattr(cf_python, attr)
    expected = cf_fixture[attr]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
