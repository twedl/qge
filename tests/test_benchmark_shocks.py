"""Golden-master tests for the Benchmark counterfactuals.

For each picked shock, compare to the corresponding
`Shock_region{n}_Benchmark.mat` / `Shock_sector{n}_Benchmark.mat` workspace.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from qge.io import MATLAB_ROOT
from qge.models.benchmark import compute_regional_shock, compute_sectoral_shock


REGIONAL_DIR = MATLAB_ROOT / "Benchmark_Model" / "regional shocks" / "shocks"
SECTORAL_DIR = MATLAB_ROOT / "Benchmark_Model" / "sectoral shocks" / "shocks"


def _load_shock_mat(path: Path) -> dict:
    raw = loadmat(path)
    return {k: v for k, v in raw.items() if not k.startswith("__")}


def _flat(x):
    return np.asarray(x).ravel()


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


# Pick three representative shocks: Alabama (region 0), California (region 4),
# Computers and Electronics (sector 10).
@pytest.fixture(scope="module")
def alabama_shock():
    return compute_regional_shock(region=0)


@pytest.fixture(scope="module")
def california_shock():
    return compute_regional_shock(region=4)


@pytest.fixture(scope="module")
def computers_shock():
    return compute_sectoral_shock(sector=10)


@pytest.fixture(scope="module")
def alabama_golden():
    return _load_shock_mat(REGIONAL_DIR / "Shock_region1_Benchmark.mat")


@pytest.fixture(scope="module")
def california_golden():
    return _load_shock_mat(REGIONAL_DIR / "Shock_region5_Benchmark.mat")


@pytest.fixture(scope="module")
def computers_golden():
    return _load_shock_mat(SECTORAL_DIR / "Shock_sector11_Benchmark.mat")


def _check_match(result, gold, *, rtol_strict=1e-6, atol_strict=1e-8):
    """Assert a bundle of post-shock outputs match the MATLAB golden state."""
    # Scalar aggregates.
    np.testing.assert_allclose(result.TFP_hat, _scalar(gold["TFP_hat"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(result.GDP_hat, _scalar(gold["GDP_hat"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(result.V_hat, _scalar(gold["V_hat"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(result.Y, _scalar(gold["Y"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(result.VA0, _scalar(gold["VA0"]),
                                rtol=rtol_strict, atol=atol_strict)
    # Vectors.
    np.testing.assert_allclose(_flat(result.L_hat), _flat(gold["L_hat"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(_flat(result.Yn), _flat(gold["Yn"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(_flat(result.VAn0), _flat(gold["VAn0"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(_flat(result.GDPn), _flat(gold["GDPn"]),
                                rtol=rtol_strict, atol=atol_strict)
    np.testing.assert_allclose(_flat(result.TFPn), _flat(gold["TFPn_hat"]),
                                rtol=rtol_strict, atol=atol_strict)


def test_regional_shock_alabama(alabama_shock, alabama_golden):
    _check_match(alabama_shock, alabama_golden)


def test_regional_shock_california(california_shock, california_golden):
    _check_match(california_shock, california_golden)


def test_sectoral_shock_computers(computers_shock, computers_golden):
    # Sectoral runs at MATLAB's looser tol=1e-8, so we relax slightly.
    _check_match(computers_shock, computers_golden, rtol_strict=1e-5, atol_strict=1e-6)
