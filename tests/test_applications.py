"""Golden-master tests for the four Section 6 applications.

Compare each Python application result to the corresponding MATLAB workspace
in Applications/shocks applications/shocks/.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import loadmat

from qge.applications import (
    shock_california_computers,
    shock_fire_nyc,
    shock_katrina,
    shock_north_dakota,
)
from qge.io import MATLAB_ROOT

SHOCKS_DIR = MATLAB_ROOT / "Applications" / "shocks applications" / "shocks"


APPLICATIONS = [
    ("california_computers", shock_california_computers, "Shock_California_Computers.mat"),
    ("north_dakota",         shock_north_dakota,         "Shock_NorthDakota.mat"),
    ("fire_nyc",             shock_fire_nyc,             "Shock_FIRE_NewYork.mat"),
    ("katrina",              shock_katrina,              "Shock_Katrina.mat"),
]


def _scalar(x):
    return float(np.asarray(x).ravel()[0])


def _flat(x):
    return np.asarray(x).ravel()


@pytest.mark.parametrize(
    "name, shock_fn, golden_filename",
    APPLICATIONS,
    ids=[a[0] for a in APPLICATIONS],
)
def test_application_matches_matlab(name, shock_fn, golden_filename):
    """Each application result matches the saved MATLAB workspace."""
    result = shock_fn()
    gold = {
        k: v
        for k, v in loadmat(SHOCKS_DIR / golden_filename).items()
        if not k.startswith("__")
    }
    rtol, atol = 1e-5, 1e-6
    np.testing.assert_allclose(result.TFP_hat, _scalar(gold["TFP_hat"]),
                                rtol=rtol, atol=atol)
    np.testing.assert_allclose(result.GDP_hat, _scalar(gold["GDP_hat"]),
                                rtol=rtol, atol=atol)
    np.testing.assert_allclose(result.V_hat, _scalar(gold["V_hat"]),
                                rtol=rtol, atol=atol)
    np.testing.assert_allclose(_flat(result.L_hat), _flat(gold["L_hat"]),
                                rtol=rtol, atol=atol)
    np.testing.assert_allclose(_flat(result.GDPn), _flat(gold["GDPn"]),
                                rtol=rtol, atol=atol)
    np.testing.assert_allclose(_flat(result.TFPn), _flat(gold["TFPn_hat"]),
                                rtol=rtol, atol=atol)
