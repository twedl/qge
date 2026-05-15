"""Verify Step 4 (Phase 2d) stitched baseline matches MATLAB.

Reference: ``Baseline_economy.mat`` is the MATLAB output of
Step_4_Baseline.m. Step 4 is pure array stitching of Phase 2a/2b/2c
outputs — no new computation — so the dominant cost of this test is
running Phase 2c (the 200-period forward sim). The test is therefore
@pytest.mark.slow.

The MATLAB workspace is v7.3 (HDF5-backed) so we use h5py rather than
scipy.io.loadmat. HDF5 stores arrays in transposed axis order relative
to MATLAB — we restore the (J*N, N, time)-style layout on read.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from qge.dynamic import build_quarterly_series
from qge.models.baseline_economy import stitch_baseline_economy
from qge.models.dynamic_baseline import compute_dynamic_baseline_2000_2007
from qge.models.forward_simulation import compute_baseline_forward_2007

REPO_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Baseline_economy"
STEP4_MAT = REP_DIR / "Baseline_economy.mat"
STEP3_MAT = REP_DIR / "Baseline_2007.mat"


@pytest.fixture(scope="module")
def step4_fixture():
    if not STEP4_MAT.exists():
        pytest.skip(f"Step 4 MATLAB fixture not found: {STEP4_MAT}")
    out = {}
    with h5py.File(str(STEP4_MAT), "r") as f:
        for k in ("series_xbilat", "series_pi", "series_wages",
                  "series_Ljnhat", "series_mu", "L0_initial"):
            # HDF5 stores arrays transposed relative to MATLAB; restore
            # the trailing time axis by transposing all axes.
            out[k] = np.asarray(f[k]).T
    return out


@pytest.fixture(scope="module")
def step4_python(raw, baseline):
    if not REP_DIR.exists():
        pytest.skip(f"CDP replication kit not present: {REP_DIR}")
    Yt_seed = _load_Hvectnoshock()
    quarterly = build_quarterly_series(REP_DIR, baseline, raw.gamma, raw.B)
    dynamic_2007 = compute_dynamic_baseline_2000_2007(
        raw=raw, baseline=baseline, quarterly=quarterly,
    )
    forward = compute_baseline_forward_2007(
        Yt_seed=Yt_seed, raw=raw, baseline=baseline,
        dynamic_2000_2007=dynamic_2007, quarterly=quarterly,
        max_outer_iter=2,
    )
    return stitch_baseline_economy(quarterly, dynamic_2007, forward)


def _load_Hvectnoshock() -> np.ndarray:
    from scipy.io import loadmat
    return loadmat(str(STEP3_MAT))["Hvectnoshock"]


@pytest.mark.slow
@pytest.mark.parametrize(
    "attr, rtol, atol",
    [
        ("series_xbilat",   2e-2,  1e-3),
        ("series_pi",       5e-3,  1e-3),
        ("series_wages",    5e-3,  1e-3),
        ("series_Ljnhat",   5e-3,  1e-3),
        ("series_mu",       5e-2,  1e-3),
        ("L0_initial",      1e-12, 0),
    ],
    ids=["xbilat", "pi", "wages", "Ljnhat", "mu", "L0_initial"],
)
def test_baseline_economy_matches_matlab(
    step4_python, step4_fixture, attr, rtol, atol
):
    actual = getattr(step4_python, attr)
    expected = np.squeeze(step4_fixture[attr])
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_stitch_shapes(quarterly, raw):
    """Fast unit test: stitch shapes are correct without running Phase 2c.

    Constructs a synthetic ForwardSimulation with zeros to verify the
    array dimensions line up. Doesn't validate numerics — that's the
    slow integration test's job.
    """
    from qge.models.baseline_economy import (
        TOTAL_MU_TRANSITIONS, TOTAL_QUARTERS, stitch_baseline_economy,
    )
    from qge.models.dynamic_baseline import DynamicBaseline2000_2007
    from qge.models.base_year import EquilibriumResult
    from qge.models.forward_simulation import ForwardSimulation

    J, N, R, RJ1 = raw.J, raw.N, raw.R, raw.R * (raw.J + 1)
    dynamic = DynamicBaseline2000_2007(
        New_Din_baseline=np.zeros((J * N, N, 29)),
        New_series_xbilat=np.zeros((J * N, N, 29)),
        New_series_wageshat=np.zeros((J, N, 29)),
        final_equilibrium=EquilibriumResult(  # type: ignore[call-arg]
            om=np.zeros((J, N)), wf0=np.zeros((J, N)), rf0=np.zeros((J, N)),
            VARjnp=np.zeros((J, N)), VALjnp=np.zeros((J, N)),
            Phat=np.zeros(N), phat=np.zeros((J, N)),
            Dinp=np.zeros((J * N, N)), Xp=np.zeros((J, N)),
            Snp=np.zeros(N), xbilatp=np.zeros((J * N, N)), iterations=0,
        ),
    )
    forward = ForwardSimulation(
        Hvectnoshock=np.zeros((RJ1, 200)),
        pi_baseline=np.zeros((J * N, N, 200)),
        xbilat_out=np.zeros((J * N, N, 200)),
        wages0=np.zeros((J, N, 200)),
        Ljn_hat0=np.zeros((J, N, 200)),
        mu=np.zeros((RJ1, RJ1, 200)),
        outer_iters=1,
    )
    out = stitch_baseline_economy(quarterly, dynamic, forward)
    assert out.series_xbilat.shape == (J * N, N, TOTAL_QUARTERS)
    assert out.series_pi.shape == (J * N, N, TOTAL_QUARTERS)
    assert out.series_wages.shape == (J, N, TOTAL_QUARTERS)
    assert out.series_Ljnhat.shape == (J, R, TOTAL_QUARTERS)
    assert out.series_mu.shape == (RJ1, RJ1, TOTAL_MU_TRANSITIONS)
    assert out.L0_initial.shape == (RJ1,)
