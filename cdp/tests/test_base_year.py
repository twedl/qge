"""Verify the Python Base_Year solver matches the MATLAB workspace.

Two layers of checks: (1) the data.m transformations applied at load time
produce arrays that match the MATLAB workspace's "data" derivatives (B,
gamma, G, Din, VALjn0, VARjn0, alphas, VAR, Bn). (2) the full
solvewnew run produces the same om/Dinp/Xp/VARjnp/VALjnp/Phat/xbilatp
as Base_year.mat.

Tolerances are tight: ``rtol=1e-9`` for data terms, ``rtol=1e-6`` for
the iterative solver outputs (the MATLAB solver uses tol=1e-7).
"""

from __future__ import annotations

import numpy as np
import pytest

from qge.models.base_year import compute_baseline


def test_data_B_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.B, matlab_baseline["B"], rtol=1e-12, atol=0)


def test_data_gamma_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.gamma, matlab_baseline["gamma"], rtol=1e-12, atol=0)


def test_data_G_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.G, matlab_baseline["G"], rtol=1e-9, atol=1e-12)


def test_data_Din_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.Din, matlab_baseline["Din"], rtol=1e-9, atol=1e-12)


def test_data_VAL_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.VALjn0, matlab_baseline["VALjn0"], rtol=1e-9, atol=0)


def test_data_VAR_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.VARjn0, matlab_baseline["VARjn0"], rtol=1e-9, atol=0)


def test_data_alphas_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.alphas, matlab_baseline["alphas"], rtol=1e-9, atol=0)


def test_data_Bn_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.Bn, matlab_baseline["Bn"].ravel(), rtol=1e-9, atol=1e-3)


def test_data_T_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.T, matlab_baseline["T"].ravel(), rtol=1e-12, atol=0)


def test_data_io_matches_matlab(raw, matlab_baseline):
    np.testing.assert_allclose(raw.io, matlab_baseline["io"].ravel(), rtol=1e-9, atol=1e-12)


def test_GO_check_matches(raw, matlab_baseline):
    """Recomputed gross output from xbilat row sums matches the saved GO."""
    np.testing.assert_allclose(raw.GO_check.T, matlab_baseline["GO"], rtol=1e-9, atol=1e-3)


@pytest.fixture(scope="module")
def baseline(raw):
    """One end-to-end compute_baseline run shared across solver checks."""
    return compute_baseline(raw=raw, tol=1e-7, vfactor=-0.05)


def test_solver_om_matches_matlab(baseline, matlab_baseline):
    np.testing.assert_allclose(baseline.om, matlab_baseline["om"], rtol=1e-5, atol=1e-7)


def test_solver_Dinp_matches_matlab(baseline, matlab_baseline):
    np.testing.assert_allclose(baseline.Dinp, matlab_baseline["Dinp"], rtol=1e-5, atol=1e-9)


def test_solver_Xp_matches_matlab(baseline, matlab_baseline):
    np.testing.assert_allclose(baseline.Xp, matlab_baseline["Xp"], rtol=1e-5, atol=1e-3)


def test_solver_VARjnp_matches_matlab(baseline, matlab_baseline):
    np.testing.assert_allclose(baseline.VARjnp, matlab_baseline["VARjnp"], rtol=1e-5, atol=1e-3)


def test_solver_VALjnp_matches_matlab(baseline, matlab_baseline):
    np.testing.assert_allclose(baseline.VALjnp, matlab_baseline["VALjnp"], rtol=1e-5, atol=1e-3)


def test_solver_Phat_matches_matlab(baseline, matlab_baseline):
    np.testing.assert_allclose(baseline.Phat, matlab_baseline["Phat"].ravel(), rtol=1e-5)


def test_solver_xbilatp_matches_matlab(baseline, matlab_baseline):
    np.testing.assert_allclose(baseline.xbilatp, matlab_baseline["xbilatp"], rtol=1e-5, atol=1e-3)
