"""Tests for the aggregate elasticity formulas.

Verifies the Aggregate_elasticities_*.m formulas line up:
(a) computing on the MATLAB-saved shock values reproduces the expected number;
(b) running the Python pipeline (shock → elasticity) lands on that same value.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import loadmat

from qge.elasticities import (
    ElasticityRow,
    regional_elasticities,
    sectoral_elasticities,
)
from qge.io import DATA_DIR, MATLAB_ROOT
from qge.models.benchmark import compute_regional_shock, compute_sectoral_shock

REGIONAL_DIR = MATLAB_ROOT / "Benchmark_Model" / "regional shocks" / "shocks"
SECTORAL_DIR = MATLAB_ROOT / "Benchmark_Model" / "sectoral shocks" / "shocks"


def _shock_namespace(mat: dict) -> SimpleNamespace:
    """Duck-typed BenchmarkShockResult from a MATLAB shock workspace."""
    return SimpleNamespace(
        TFP_hat=float(mat["TFP_hat"].ravel()[0]),
        GDP_hat=float(mat["GDP_hat"].ravel()[0]),
        V_hat=float(mat["V_hat"].ravel()[0]),
        Y=float(mat["Y"].ravel()[0]),
        VA0=float(mat["VA0"].ravel()[0]),
        Yn=mat["Yn"].ravel() if "Yn" in mat else np.array([]),
        Yj=mat["Yj"].ravel() if "Yj" in mat else np.array([]),
        VAn0=mat["VAn0"].ravel() if "VAn0" in mat else np.array([]),
        VAj0=mat["VAj0"].ravel() if "VAj0" in mat else np.array([]),
    )


def test_regional_formula_matches_matlab_recipe():
    """The function evaluates the formula in Aggregate_elasticities_regional_shocks.m."""
    region = 4  # California, MATLAB region 5
    mat = loadmat(REGIONAL_DIR / "Shock_region5_Benchmark.mat")
    shock = _shock_namespace(mat)
    Ln = mat["Ln"].ravel()

    elast = regional_elasticities(shock, region=region, Ln=Ln)

    # Literal port of the MATLAB three lines for this region.
    expected = ElasticityRow(
        TFP=10.0 * (shock.TFP_hat - 1) / (shock.Yn[region] / shock.Y),
        GDP=10.0 * (shock.GDP_hat - 1) / (shock.VAn0[region] / shock.VA0),
        welfare=10.0 * (shock.V_hat - 1) / Ln[region],
    )
    assert elast == expected


def test_sectoral_formula_matches_matlab_recipe():
    """The function evaluates the formula in Aggregate_elasticities_sectoral_shocks.m."""
    sector = 10  # Computers & Electronics, MATLAB sector 11
    mat = loadmat(SECTORAL_DIR / "Shock_sector11_Benchmark.mat")
    shock = _shock_namespace(mat)
    Ljn = loadmat(DATA_DIR / "Base_Year_Benchmark.mat")["Ljn_RS"]

    elast = sectoral_elasticities(shock, sector=sector, Ljn=Ljn)

    expected = ElasticityRow(
        TFP=10.0 * (shock.TFP_hat - 1) / (shock.Yj[sector] / shock.Y),
        GDP=10.0 * (shock.GDP_hat - 1) / (shock.VAj0[sector] / shock.VA0),
        welfare=10.0 * (shock.V_hat - 1) / Ljn[sector, :].sum(),
    )
    assert elast == expected


def test_python_regional_elasticity_matches_matlab_end_to_end():
    """Python shock → Python elasticity should equal MATLAB shock → MATLAB elasticity."""
    region = 4
    py_shock = compute_regional_shock(region=region)
    Ln = loadmat(DATA_DIR / "Base_Year_Benchmark.mat")["Ln_RS"].ravel()
    py_elast = regional_elasticities(py_shock, region=region, Ln=Ln)

    mat = loadmat(REGIONAL_DIR / "Shock_region5_Benchmark.mat")
    expected_TFP = 10.0 * (float(mat["TFP_hat"].ravel()[0]) - 1) / (
        mat["Yn"].ravel()[region] / float(mat["Y"].ravel()[0])
    )
    expected_GDP = 10.0 * (float(mat["GDP_hat"].ravel()[0]) - 1) / (
        mat["VAn0"].ravel()[region] / float(mat["VA0"].ravel()[0])
    )
    expected_welfare = 10.0 * (float(mat["V_hat"].ravel()[0]) - 1) / mat["Ln"].ravel()[region]

    assert py_elast.TFP == pytest.approx(expected_TFP, rel=1e-10)
    assert py_elast.GDP == pytest.approx(expected_GDP, rel=1e-10)
    assert py_elast.welfare == pytest.approx(expected_welfare, rel=1e-10)
