"""Sanity tests for the geographic-barriers counterfactual.

The MATLAB script (trade_costs.m) only prints its results and does not save
a workspace, so there is no published .mat fixture to compare against.
These tests verify the function runs to convergence and produces aggregate
gains in the qualitative direction the paper reports (positive — reducing
trade barriers raises real income).
"""

from __future__ import annotations

import numpy as np
import pytest

from qge.geographic import compute_geographic_barriers


@pytest.fixture(scope="module")
def distance_result():
    return compute_geographic_barriers(scenario="distance")


@pytest.fixture(scope="module")
def other_result():
    return compute_geographic_barriers(scenario="other_barriers")


def test_distance_gains_positive(distance_result):
    """Eliminating geographic distance should raise aggregate TFP/GDP/welfare."""
    assert distance_result.TFP_hat > 1
    assert distance_result.GDP_hat > 1
    assert distance_result.V_hat > 1
    # Spot-checked against trade_costs.m output: TFP ≈ 1.51, GDP ≈ 2.26.
    assert np.isfinite(distance_result.TFP_hat)
    assert distance_result.TFP_hat < 2
    assert distance_result.GDP_hat < 3


def test_other_barriers_gains_positive(other_result):
    assert other_result.TFP_hat > 1
    assert other_result.GDP_hat > 1
    assert other_result.V_hat > 1
    # Spot-checked: TFP ≈ 1.036, GDP ≈ 1.105.
    assert other_result.TFP_hat < 1.2
    assert other_result.GDP_hat < 1.3


def test_distance_dominates_other_barriers(distance_result, other_result):
    """Eliminating distance is a larger shock than eliminating "other" barriers
    (distance dominates trade-cost reductions in the CPRHS decomposition)."""
    assert distance_result.TFP_hat > other_result.TFP_hat
    assert distance_result.GDP_hat > other_result.GDP_hat


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match="unknown scenario"):
        compute_geographic_barriers(scenario="nonexistent")
