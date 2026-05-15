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
def matlab_baseline():
    """Load the MATLAB Base_year.mat workspace. Skip if not on disk."""
    if not MAT_FIXTURE.exists():
        pytest.skip(f"MATLAB fixture not found at {MAT_FIXTURE}")
    return loadmat(str(MAT_FIXTURE))
