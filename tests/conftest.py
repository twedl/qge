"""Pytest fixtures: pre-load the raw inputs and the Benchmark golden state."""

from __future__ import annotations

import pytest

from qge.io import MATLAB_ROOT, load_base_year, load_raw_inputs


def pytest_collection_modifyitems(config, items):
    """Skip everything if the MATLAB reference tree is missing."""
    if MATLAB_ROOT.exists():
        return
    skip = pytest.mark.skip(
        reason=f"MATLAB reference tree not found at {MATLAB_ROOT}; "
        "fetch the CPRHS replication files to run these tests."
    )
    for item in items:
        item.add_marker(skip)


@pytest.fixture(scope="session")
def raw():
    """Raw model inputs (xbilat, L_j_n, IO, gamma, B, alphas, io)."""
    return load_raw_inputs()


@pytest.fixture(scope="session")
def benchmark_golden():
    """Pre-computed Benchmark baseline equilibrium from MATLAB."""
    return load_base_year("Benchmark")
