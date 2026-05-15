"""Shared fixtures: loads CDP Base_year.mat if present, otherwise skips."""

from __future__ import annotations

from pathlib import Path

import pytest
from scipy.io import loadmat

from qge.io import load_inputs
from qge.models.base_year import compute_baseline

REPO_ROOT = Path(__file__).resolve().parent.parent
MAT_FIXTURE = REPO_ROOT / "CDP replication files" / "Base_Year" / "Base_year.mat"


@pytest.fixture(scope="session")
def raw():
    return load_inputs()


@pytest.fixture(scope="session")
def baseline(raw):
    """One Base_Year solve shared across every test module."""
    return compute_baseline(raw=raw, tol=1e-7, vfactor=-0.05)


@pytest.fixture(scope="session")
def quarterly(raw, baseline):
    """Phase 2a quarterly series shared across test modules."""
    from qge.dynamic import build_quarterly_series
    rep_dir = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Baseline_economy"
    if not rep_dir.exists():
        pytest.skip(f"CDP replication kit not present: {rep_dir}")
    return build_quarterly_series(rep_dir, baseline, raw.gamma, raw.B)


@pytest.fixture(scope="session")
def baseline_economy_mat():
    """Load the saved Phase 2d Baseline_economy.mat (HDF5/v7.3).

    Used by Phase 3 / Phase 4 slow tests to skip rerunning Phase 2c.
    HDF5 stores arrays transposed relative to MATLAB; .T restores the
    (J*N, N, time)-style layout.
    """
    import h5py
    import numpy as np

    from qge.models.baseline_economy import BaselineEconomy

    path = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Baseline_economy" / "Baseline_economy.mat"
    if not path.exists():
        pytest.skip(f"Baseline_economy.mat not present: {path}")
    with h5py.File(str(path), "r") as f:
        return BaselineEconomy(
            series_xbilat=np.asarray(f["series_xbilat"]).T,
            series_pi=np.asarray(f["series_pi"]).T,
            series_wages=np.asarray(f["series_wages"]).T,
            series_Ljnhat=np.asarray(f["series_Ljnhat"]).T,
            series_mu=np.asarray(f["series_mu"]).T,
            L0_initial=np.squeeze(np.asarray(f["L0_initial"])),
        )


@pytest.fixture(scope="session")
def counterfactual(raw, baseline, baseline_economy_mat):
    """Phase 3 counterfactual shared across test modules — ~3-15 min."""
    from scipy.io import loadmat

    from qge.models.counterfactual import compute_counterfactual_economy

    cf_mat = REPO_ROOT / "CDP replication files" / "time_varying_fundamentals" / "Counterfactual_economy" / "Counterfactual_economy.mat"
    if not cf_mat.exists():
        pytest.skip(f"Counterfactual_economy.mat not present: {cf_mat}")
    V_seed = loadmat(str(cf_mat))["V"]
    return compute_counterfactual_economy(
        V_seed=V_seed, baseline_econ=baseline_economy_mat,
        base_year=baseline, raw=raw, max_outer_iter=2,
    )


@pytest.fixture(scope="session")
def matlab_baseline():
    """Load the MATLAB Base_year.mat workspace. Skip if not on disk."""
    if not MAT_FIXTURE.exists():
        pytest.skip(f"MATLAB fixture not found at {MAT_FIXTURE}")
    return loadmat(str(MAT_FIXTURE))
