"""Shared fixtures: loads CDP Base_year.mat if present, otherwise skips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from qge.io import load_inputs

REPO_ROOT = Path(__file__).resolve().parent.parent
MAT_FIXTURE = REPO_ROOT / "CDP replication files" / "Base_Year" / "Base_year.mat"


@pytest.fixture(scope="session")
def raw():
    return load_inputs()


@pytest.fixture(scope="session")
def matlab_baseline():
    """Load the MATLAB Base_year.mat workspace. Skip if not on disk."""
    if not MAT_FIXTURE.exists():
        pytest.skip(f"MATLAB fixture not found at {MAT_FIXTURE}")
    return loadmat(str(MAT_FIXTURE))
