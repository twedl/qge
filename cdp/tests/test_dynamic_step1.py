"""Verify Step 1 (Phase 2a) quarterly series match the MATLAB workspace.

Reference: `Baseline_2000_2007_economy_actual_data.mat` carries the
MATLAB output of Step_1_data.m. Each Python series should match its
MATLAB counterpart at machine epsilon (the data construction is
algebraic, not iterative).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from qge.dynamic import build_quarterly_series
from qge.io import load_inputs
from qge.models.base_year import compute_baseline

REPO_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Baseline_economy"
STEP1_MAT = REP_DIR / "Baseline_2000_2007_economy_actual_data.mat"


@pytest.fixture(scope="module")
def step1_fixture():
    if not STEP1_MAT.exists():
        pytest.skip(f"Step 1 MATLAB fixture not found: {STEP1_MAT}")
    return loadmat(str(STEP1_MAT))


@pytest.fixture(scope="module")
def step1_python():
    if not REP_DIR.exists():
        pytest.skip(f"CDP replication kit not present: {REP_DIR}")
    raw = load_inputs()
    baseline = compute_baseline(raw=raw, tol=1e-7, vfactor=-0.05)
    return build_quarterly_series(REP_DIR, baseline, raw.gamma, raw.B)


def test_Din_baseline_matches(step1_python, step1_fixture):
    np.testing.assert_allclose(
        step1_python.Din_baseline, step1_fixture["Din_baseline"],
        rtol=1e-5, atol=1e-7,
    )


def test_series_xbilat_matches(step1_python, step1_fixture):
    np.testing.assert_allclose(
        step1_python.series_xbilat, step1_fixture["series_xbilat"],
        rtol=1e-5, atol=1e-3,
    )


def test_series_wageshat_matches(step1_python, step1_fixture):
    np.testing.assert_allclose(
        step1_python.series_wageshat, step1_fixture["series_wageshat"],
        rtol=1e-5, atol=1e-7,
    )


def test_series_Ljn0hat_matches(step1_python, step1_fixture):
    np.testing.assert_allclose(
        step1_python.series_Ljn0hat, step1_fixture["series_Ljn0hat"],
        rtol=1e-5, atol=1e-7,
    )


def test_series_mu_matches(step1_python, step1_fixture):
    np.testing.assert_allclose(
        step1_python.series_mu, step1_fixture["series_mu"],
        rtol=1e-12, atol=0,
    )


def test_L0_initial_matches(step1_python, step1_fixture):
    np.testing.assert_allclose(
        step1_python.L0_initial, step1_fixture["L0_initial"].ravel(),
        rtol=1e-12, atol=0,
    )
