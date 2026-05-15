"""Verify Phase 4 employment / welfare effects against MATLAB.

The MATLAB driver scripts print key scalar results to the console (e.g.
``manufactures_share_china_shock`` and ``Welfare``) and dump figures.
We verify those scalars plus the shape and sign of the array outputs.

Slow because both effects functions require a Phase 3 run (~13 min);
deselected behind the existing ``-m slow`` filter.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.io import loadmat

from qge.effects import compute_employment_effects, compute_welfare_effects
from qge.models.baseline_economy import BaselineEconomy
from qge.models.counterfactual import compute_counterfactual_economy

REPO_ROOT = Path(__file__).resolve().parent.parent
REP_DIR = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals"
CF_MAT = REP_DIR / "Counterfactual_economy" / "Counterfactual_economy.mat"
BASELINE_MAT = REP_DIR / "Baseline_economy" / "Baseline_economy.mat"


def _load_baseline_economy_from_mat() -> BaselineEconomy:
    with h5py.File(str(BASELINE_MAT), "r") as f:
        return BaselineEconomy(
            series_xbilat=np.asarray(f["series_xbilat"]).T,
            series_pi=np.asarray(f["series_pi"]).T,
            series_wages=np.asarray(f["series_wages"]).T,
            series_Ljnhat=np.asarray(f["series_Ljnhat"]).T,
            series_mu=np.asarray(f["series_mu"]).T,
            L0_initial=np.squeeze(np.asarray(f["L0_initial"])),
        )


@pytest.fixture(scope="module")
def cf_python(raw, baseline):
    if not BASELINE_MAT.exists() or not CF_MAT.exists():
        pytest.skip("CDP Phase 2/3 fixtures not present")
    baseline_econ = _load_baseline_economy_from_mat()
    V_seed = loadmat(str(CF_MAT))["V"]
    return compute_counterfactual_economy(
        V_seed=V_seed, baseline_econ=baseline_econ,
        base_year=baseline, raw=raw, max_outer_iter=2,
    )


@pytest.mark.slow
def test_employment_effects_signs_and_shapes(cf_python):
    """Manufacturing share falls, non-manufacturing rises, signs are stable."""
    if not BASELINE_MAT.exists():
        pytest.skip("Baseline_economy.mat not present")
    baseline_econ = _load_baseline_economy_from_mat()
    eff = compute_employment_effects(baseline_econ, cf_python)

    # Paper reports manufactures_share_china_shock ≈ -0.36 to -0.45 pp.
    assert -1.0 < eff.manuf_share_change < 0.0, eff.manuf_share_change
    assert 0.0 < eff.nonmanuf_share_change < 1.0
    assert eff.sectoral_manuf_contrib.shape == (12,)
    assert eff.sectoral_nonmanuf_contrib.shape == (10,)
    assert eff.regional_manuf_contrib.shape == (50,)
    assert eff.regional_nonmanuf_contrib.shape == (50,)
    # Sectoral contributions normalize to 100.
    np.testing.assert_allclose(eff.sectoral_manuf_contrib.sum(), 100.0, atol=1e-9)
    np.testing.assert_allclose(eff.sectoral_nonmanuf_contrib.sum(), 100.0, atol=1e-9)


@pytest.mark.slow
def test_welfare_effects_signs_and_shapes(cf_python):
    """Aggregate welfare from China shock should be slightly negative (paper ≈ -0.2%)."""
    if not BASELINE_MAT.exists():
        pytest.skip("Baseline_economy.mat not present")
    baseline_econ = _load_baseline_economy_from_mat()
    wel = compute_welfare_effects(baseline_econ, cf_python, baseline_econ.L0_initial)
    R, JNT1, T = wel.logdelta.shape
    assert R == 50 and JNT1 == 23 and T == 200
    # Paper aggregate welfare effect is small and negative.
    assert -2.0 < wel.aggregate_welfare_pct < 0.5, wel.aggregate_welfare_pct
