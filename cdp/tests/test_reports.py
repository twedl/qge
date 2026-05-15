"""DataFrame reporting layer — label propagation and shape sanity.

These tests don't need MATLAB fixtures or solver runs; they verify the
wrapping functions produce correctly-shaped DataFrames with the right
labels for synthetic input arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qge.labels import (
    COUNTRIES, LABOR_MARKETS, REGIONS, SECTORS, US_STATES, quarter_labels,
)
from qge.reports import (
    employment_series, labor_at, manufacturing_share_series,
    nonemployment_share_series, trade_flows_at, trade_share_series,
    wage_series, wages_at, welfare_logdelta_at, welfare_summary,
    employment_effects_dataframes,
)

J, N, R, T = 22, 87, 50, 200


def test_label_counts():
    assert len(SECTORS) == 22
    assert len(LABOR_MARKETS) == 23
    assert len(US_STATES) == 50
    assert len(COUNTRIES) == 37
    assert len(REGIONS) == 87


def test_quarter_labels():
    q = quarter_labels(2000, 8)
    assert q == ("2000Q1", "2000Q2", "2000Q3", "2000Q4",
                 "2001Q1", "2001Q2", "2001Q3", "2001Q4")


def test_wages_at_shape_and_labels():
    arr = np.random.rand(J, N, T)
    df = wages_at(arr, t=5)
    assert df.shape == (J, N)
    assert list(df.index) == list(SECTORS)
    assert list(df.columns) == list(REGIONS)
    np.testing.assert_array_equal(df.values, arr[..., 5])


def test_labor_at_shape_and_labels():
    arr = np.random.rand(J + 1, R, T)
    df = labor_at(arr, t=10)
    assert df.shape == (J + 1, R)
    assert list(df.index) == list(LABOR_MARKETS)
    assert list(df.columns) == list(US_STATES)


def test_trade_flows_at_long_form():
    arr = np.random.rand(J * N, N, T)
    df = trade_flows_at(arr, t=0)
    assert list(df.columns) == ["sector", "destination", "source", "value"]
    assert len(df) == J * N * N


def test_wage_series_per_region():
    arr = np.random.rand(J, N, T)
    df = wage_series(arr, region="California")
    assert df.shape == (J, T)
    assert list(df.index) == list(SECTORS)
    assert list(df.columns)[:3] == ["2000Q1", "2000Q2", "2000Q3"]


def test_wage_series_all_regions():
    arr = np.random.rand(J, N, T)
    df = wage_series(arr, region=None)
    assert df.shape == (J * N, T)


def test_trade_share_series_for_pair():
    arr = np.random.rand(J * N, N, T)
    df = trade_share_series(arr, sector="Machinery", destination="Ohio")
    assert df.shape == (N, T)
    assert list(df.index) == list(REGIONS)


def test_employment_series_long_form():
    arr = np.random.rand(J + 1, R, T)
    df = employment_series(arr)
    assert list(df.columns) == ["labor_market", "state", "quarter", "value"]
    assert len(df) == (J + 1) * R * T


def test_manufacturing_share_series_signature():
    arr = np.random.rand(J + 1, R, T)
    s = manufacturing_share_series(arr)
    assert isinstance(s, pd.Series)
    assert len(s) == T
    assert all(0 <= v <= 1 for v in s.dropna())


def test_nonemployment_share_series_signature():
    arr = np.random.rand(J + 1, R, T)
    s = nonemployment_share_series(arr)
    assert len(s) == T


def test_employment_effects_dataframes():
    from qge.effects import EmploymentEffects
    eff = EmploymentEffects(
        manuf_share_change=-0.4,
        nonmanuf_share_change=0.3,
        construction_share_change=0.05,
        trade_share_change=0.04,
        services_share_change=0.21,
        nonemployment_share_change=0.1,
        sectoral_manuf_contrib=np.linspace(5, 20, 12),
        sectoral_nonmanuf_contrib=np.linspace(2, 15, 10),
        regional_manuf_contrib=np.linspace(-3, 7, 50),
        regional_nonmanuf_contrib=np.linspace(-2, 6, 50),
    )
    bundle = employment_effects_dataframes(eff)
    assert set(bundle) == {
        "aggregate", "sectoral_manuf", "sectoral_nonmanuf",
        "regional_manuf", "regional_nonmanuf",
    }
    assert bundle["aggregate"]["manufacturing"] == pytest.approx(-0.4)
    assert list(bundle["sectoral_manuf"].index) == list(SECTORS[:12])
    assert list(bundle["regional_manuf"].index) == list(US_STATES)


def test_welfare_logdelta_at():
    from qge.effects import WelfareEffects
    logdelta = np.random.rand(R, J + 1, T) * 0.01
    wel = WelfareEffects(logdelta=logdelta, aggregate_welfare_pct=-0.2)
    df = welfare_logdelta_at(wel, t=1)
    assert df.shape == (R, J + 1)
    assert list(df.index) == list(US_STATES)
    assert list(df.columns) == list(LABOR_MARKETS)


def test_welfare_summary():
    from qge.effects import WelfareEffects
    wel = WelfareEffects(
        logdelta=np.random.randn(R, J + 1, T) * 0.001,
        aggregate_welfare_pct=-0.15,
    )
    s = welfare_summary(wel)
    assert "aggregate (%)" in s.index
    assert s["aggregate (%)"] == -0.15
