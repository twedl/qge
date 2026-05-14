"""Tests for the label-aware DataFrame reporting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qge.applications import shock_california_computers
from qge.io import load_inputs
from qge.models.benchmark import (
    compute_baseline,
    compute_regional_shock,
    compute_sectoral_shock,
    regional_sweep,
    sectoral_sweep,
)


@pytest.fixture(scope="module")
def raw():
    return load_inputs()


@pytest.fixture(scope="module")
def baseline(raw):
    return compute_baseline(raw=raw)


@pytest.fixture(scope="module")
def california_shock(raw):
    return compute_regional_shock(region=4, raw=raw)


def test_baseline_regional_summary(baseline, raw):
    df = baseline.regional_summary()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "region"
    assert list(df.index) == list(raw.regions)
    assert set(df.columns) == {"Ln", "VAL", "VAR", "Chin", "LnIn", "Sn", "Bn"}
    assert df.loc["California", "Ln"] > 0


def test_baseline_employment_shares(baseline, raw):
    df = baseline.employment_shares()
    assert df.shape == (raw.J, raw.N)
    assert df.index.name == "sector"
    assert df.columns.name == "region"
    np.testing.assert_allclose(df.to_numpy().sum(), 1.0, rtol=1e-10)


def test_baseline_bilateral_trade(baseline, raw):
    df = baseline.bilateral_trade()
    assert list(df.columns) == ["sector", "destination", "source", "value"]
    assert len(df) == raw.J * raw.N * raw.N
    # Spot check: row for sector 0, dest 0, source 0
    cali_to_cali = df[
        (df["sector"] == raw.sectors[0])
        & (df["destination"] == "California")
        & (df["source"] == "California")
    ]
    assert cali_to_cali["value"].iloc[0] > 0


def test_shock_regional_summary(california_shock, raw):
    df = california_shock.regional_summary()
    assert df.index.name == "region"
    assert list(df.index) == list(raw.regions)
    assert set(df.columns) >= {"L_hat", "P_index_hat", "TFPn_hat", "GDPn_hat",
                                "Yn", "VAn0", "om"}
    # California — the shocked region — should have L_hat above 1
    # (productivity boom attracts labor).
    assert df.loc["California", "L_hat"] > 1
    # And TFPn rises for California.
    assert df.loc["California", "TFPn_hat"] > 1


def test_shock_sectoral_summary(california_shock, raw):
    df = california_shock.sectoral_summary()
    assert df.index.name == "sector"
    assert list(df.index) == list(raw.sectors)
    assert set(df.columns) == {"TFPj_hat", "GDPj_hat", "Yj", "VAj0"}


def test_application_carries_labels(raw):
    """Application shock results expose the same DataFrame helpers."""
    result = shock_california_computers(raw=raw)
    df = result.regional_summary()
    assert "California" in df.index
    assert df.loc["California", "L_hat"] > 0


@pytest.fixture(scope="module")
def alabama_shock(raw):
    return compute_regional_shock(region=0, raw=raw)


def test_regional_sweep_as_dataframe(raw, alabama_shock, california_shock):
    """Mini regional sweep — two regions — has the right DataFrame shape."""
    from qge.elasticities import regional_elasticities
    from qge.models.benchmark import RegionalSweepResult

    Ln = raw.L_j_n.sum(0) / raw.L_j_n.sum()
    shocks = [alabama_shock, california_shock]
    elast = [
        regional_elasticities(s, region=i, Ln=Ln)
        for i, s in zip((0, 4), shocks)
    ]
    sweep = RegionalSweepResult(
        shocks=shocks, elasticities=elast,
        regions=(raw.regions[0], raw.regions[4]),
    )
    df = sweep.as_dataframe()
    assert df.index.name == "region"
    assert list(df.index) == ["Alabama", "California"]
    assert set(df.columns) == {
        "TFP_elasticity", "GDP_elasticity", "welfare_elasticity", "iterations",
    }
    assert df.loc["California", "iterations"] > 0
