"""Smoke tests for the baseline equilibrium."""

import numpy as np

from qge.io import load_inputs
from qge.initialize import initialize


def test_baseline_invariants():
    raw = load_inputs()
    b = initialize(raw)

    # Total income normalized to 1
    assert abs(b.I.sum() - 1.0) < 1e-9, f"sum(I)={b.I.sum()}"

    # Trade imbalances sum to zero (no net flows out of system)
    assert abs(b.Sn.sum()) < 1e-9, f"sum(Sn)={b.Sn.sum()}"

    # Labor income matches wL + Sn = I (accounting identity)
    diff = (b.wL - b.Sn) - b.I
    assert np.max(np.abs(diff)) < 1e-8, f"wL - Sn != I, max diff={np.max(np.abs(diff))}"

    # Tn (per-capita centered) sums to 0
    assert abs(b.Tn.sum()) < 1e-12, f"sum(Tn)={b.Tn.sum()}"

    # Sn_EQ sums to zero too
    assert abs(b.Sn_EQ.sum()) < 1e-12, f"sum(Sn_EQ)={b.Sn_EQ.sum()}"

    # SnBar - Sn = Tn (provinces only)
    sn_diff = (b.SnBar - b.Sn).ravel()[: b.N - 1]
    assert np.max(np.abs(sn_diff - b.Tn.ravel())) < 1e-9


def test_baseline_solver_identity():
    """At what=ones, dL=ones the solver residual should be ~ zero."""
    from qge.solver import residual

    raw = load_inputs()
    b = initialize(raw)
    V = np.ones(2 * b.N - 1)
    res = residual(V, b)
    assert np.max(np.abs(res)) < 1e-6, f"|residual|∞={np.max(np.abs(res))}"
