"""Verify Step 2 (Phase 2b) dynamic-baseline solve matches MATLAB.

Reference: ``Baseline_2000_2007_economy_actual.mat`` is the MATLAB
output of Step_2_Baseline_00_07.m. Each Python quarter-by-quarter solve
should reproduce the three saved series:

* ``New_Din_baseline``     (J*N, N, 29) — bilateral trade shares
* ``New_series_xbilat``    (J*N, N, 29) — bilateral trade flows
* ``New_series_wageshat``  (J, N, 29)   — wage changes

Tolerances are looser than the static Phase 1 (rtol=1e-4): the
28-quarter sequence accumulates per-quarter solver tolerance ~1e-7.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from qge.models.dynamic_baseline import compute_dynamic_baseline_2000_2007

REPO_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Baseline_economy"
STEP2_MAT = REP_DIR / "Baseline_2000_2007_economy_actual.mat"


@pytest.fixture(scope="module")
def step2_fixture():
    if not STEP2_MAT.exists():
        pytest.skip(f"Step 2 MATLAB fixture not found: {STEP2_MAT}")
    return loadmat(str(STEP2_MAT))


@pytest.fixture(scope="module")
def step2_python(raw, baseline):
    if not REP_DIR.exists():
        pytest.skip(f"CDP replication kit not present: {REP_DIR}")
    return compute_dynamic_baseline_2000_2007(
        raw=raw, baseline=baseline, rep_dir=REP_DIR,
    )


@pytest.mark.parametrize(
    "attr, rtol, atol",
    [
        ("New_Din_baseline",    1e-4, 1e-7),
        ("New_series_xbilat",   1e-4, 1e-3),
        ("New_series_wageshat", 1e-4, 1e-7),
    ],
    ids=["Din", "xbilat", "wageshat"],
)
def test_dynamic_baseline_matches_matlab(
    step2_python, step2_fixture, attr, rtol, atol
):
    actual = getattr(step2_python, attr)
    expected = np.squeeze(step2_fixture[attr])
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
