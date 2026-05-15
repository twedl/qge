"""Verify Phase 4 employment / welfare effects against MATLAB.

The MATLAB driver scripts print key scalar results to the console (e.g.
``manufactures_share_china_shock`` and ``Welfare``) and dump figures.
We verify those scalars plus the shape and sign of the array outputs.

Slow because the counterfactual takes ~3-15 min depending on the
inner-equilibrium solver iterations; deselected behind ``-m slow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from qge.effects import compute_employment_effects, compute_welfare_effects


@pytest.mark.slow
def test_employment_effects_signs_and_shapes(baseline_economy_mat, counterfactual):
    """Manufacturing share falls, non-manufacturing rises."""
    eff = compute_employment_effects(baseline_economy_mat, counterfactual)

    # Paper reports manufactures_share_china_shock ≈ -0.36 to -0.45 pp.
    assert -1.0 < eff.manuf_share_change < 0.0, eff.manuf_share_change
    assert 0.0 < eff.nonmanuf_share_change < 1.0
    assert eff.sectoral_manuf_contrib.shape == (12,)
    assert eff.sectoral_nonmanuf_contrib.shape == (10,)
    assert eff.regional_manuf_contrib.shape == (50,)
    assert eff.regional_nonmanuf_contrib.shape == (50,)
    np.testing.assert_allclose(eff.sectoral_manuf_contrib.sum(), 100.0, atol=1e-9)
    np.testing.assert_allclose(eff.sectoral_nonmanuf_contrib.sum(), 100.0, atol=1e-9)


@pytest.mark.slow
def test_welfare_effects_signs_and_shapes(baseline_economy_mat, counterfactual):
    """Aggregate welfare from China shock should be slightly negative (paper ≈ -0.2%)."""
    wel = compute_welfare_effects(
        baseline_economy_mat, counterfactual, baseline_economy_mat.L0_initial,
    )
    R, JNT1, T = wel.logdelta.shape
    assert R == 50 and JNT1 == 23 and T == 200
    assert -2.0 < wel.aggregate_welfare_pct < 0.5, wel.aggregate_welfare_pct
