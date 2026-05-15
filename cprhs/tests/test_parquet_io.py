"""Round-trip test: parquet load equals .mat load element-wise.

Proves the .mat → parquet converter is lossless and the parquet loader
reconstructs the arrays in the same canonical layout.
"""

from __future__ import annotations

import numpy as np
import pytest

from qge.io import (
    DEFAULT_CALIBRATION,
    load_inputs,
    load_raw_inputs_from_mat,
)


@pytest.fixture(scope="module")
def from_parquet():
    return load_inputs(DEFAULT_CALIBRATION)


@pytest.fixture(scope="module")
def from_mat():
    return load_raw_inputs_from_mat()


def test_labels_match(from_parquet, from_mat):
    assert from_parquet.sectors == from_mat.sectors
    assert from_parquet.regions == from_mat.regions


@pytest.mark.parametrize(
    "field",
    ["T", "xbilat", "L_j_n", "IO", "gamma", "B", "alphas", "io"],
)
def test_array_field_matches(from_parquet, from_mat, field):
    """Parquet round-trip preserves every array bit-for-bit."""
    a = getattr(from_parquet, field)
    b = getattr(from_mat, field)
    assert a.shape == b.shape, f"{field}: shape mismatch {a.shape} vs {b.shape}"
    np.testing.assert_array_equal(a, b)
