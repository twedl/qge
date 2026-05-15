"""Golden-master tests for the NS / NR / NRNS baselines.

Compare each variant to the corresponding saved MATLAB workspace.
"""

from __future__ import annotations

import numpy as np
import pytest

from qge.io import load_base_year
from qge.models.variants import (
    compute_baseline_nr,
    compute_baseline_nrns,
    compute_baseline_ns,
)


def _flat(x):
    return np.asarray(x).ravel()


@pytest.fixture(scope="module")
def ns_baseline():
    return compute_baseline_ns()


@pytest.fixture(scope="module")
def nr_baseline():
    from qge.io import load_inputs
    raw = load_inputs()
    return compute_baseline_nr(raw=raw, tradable=list(raw.sectors[:15]))


@pytest.fixture(scope="module")
def nrns_baseline():
    return compute_baseline_nrns()


@pytest.fixture(scope="module")
def ns_golden():
    return load_base_year("NS")


@pytest.fixture(scope="module")
def nr_golden():
    return load_base_year("NR")


@pytest.fixture(scope="module")
def nrns_golden():
    return load_base_year("NRNS")


def _check_baseline(result, gold, *, suffix: str, rtol=1e-6, atol=1e-8):
    """Compare BenchmarkResult fields against the saved variant workspace.

    MATLAB save-name suffix differs by variant: NS → _RNS, NR → _NRS,
    NRNS → _NRNS.
    """
    np.testing.assert_allclose(
        _flat(result.Ln), _flat(gold[f"Ln{suffix}"]),
        rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        result.xbilat, gold[f"xbilat{suffix}"],
        rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        _flat(result.VAL), _flat(gold[f"VAL{suffix}"]),
        rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        _flat(result.VAR), _flat(gold[f"VAR{suffix}"]),
        rtol=rtol, atol=atol,
    )


def test_ns_baseline(ns_baseline, ns_golden):
    _check_baseline(ns_baseline, ns_golden, suffix="_RNS")


def test_nr_baseline(nr_baseline, nr_golden):
    _check_baseline(nr_baseline, nr_golden, suffix="_NRS")


def test_nrns_baseline(nrns_baseline, nrns_golden):
    _check_baseline(nrns_baseline, nrns_golden, suffix="_NRNS")
