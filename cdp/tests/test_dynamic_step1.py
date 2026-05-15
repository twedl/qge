"""Verify Step 1 (Phase 2a) quarterly series match the MATLAB workspace.

Reference: `Baseline_2000_2007_economy_actual_data.mat` carries the
MATLAB output of Step_1_data.m. Each Python series should match its
MATLAB counterpart at machine epsilon (the data construction is
algebraic, not iterative) — the trade tensors widen the tolerance to
rtol=1e-5 / atol=1e-3 to absorb cumulative round-off across 28
multiplicative steps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from qge.dynamic import build_quarterly_series

REPO_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Baseline_economy"
STEP1_MAT = REP_DIR / "Baseline_2000_2007_economy_actual_data.mat"


@pytest.fixture(scope="module")
def step1_fixture():
    if not STEP1_MAT.exists():
        pytest.skip(f"Step 1 MATLAB fixture not found: {STEP1_MAT}")
    return loadmat(str(STEP1_MAT))


@pytest.fixture(scope="module")
def step1_python(raw, baseline):
    if not REP_DIR.exists():
        pytest.skip(f"CDP replication kit not present: {REP_DIR}")
    return build_quarterly_series(REP_DIR, baseline, raw.gamma, raw.B)


@pytest.mark.parametrize(
    "attr, rtol, atol",
    [
        ("Din_baseline",     1e-5,  1e-7),
        ("series_xbilat",    1e-5,  1e-3),
        ("series_wageshat",  1e-5,  1e-7),
        ("series_Ljn0hat",   1e-5,  1e-7),
        ("series_mu",        1e-12, 0),
        ("L0_initial",       1e-12, 0),
    ],
    ids=["Din_baseline", "xbilat", "wageshat", "Ljn0hat", "mu", "L0_initial"],
)
def test_quarterly_series_matches_matlab(
    step1_python, step1_fixture, attr, rtol, atol
):
    actual = getattr(step1_python, attr)
    expected = np.squeeze(step1_fixture[attr])
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
