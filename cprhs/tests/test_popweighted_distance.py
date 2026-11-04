"""Head-Mayer effective distance in scripts/build_popweighted_distance.py.

Checked against closed forms for uniformly spread population on a disc of
radius R: mean pairwise distance 128R/(45π), harmonic mean 3πR/16.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_popweighted_distance import effective_distances  # noqa: E402

R = 100.0  # km
N = 6000


@pytest.fixture(scope="module")
def disc():
    """Uniform 'DAs' on a disc, split east/west and merged back to one region."""
    rng = np.random.default_rng(0)
    r = R * np.sqrt(rng.random(N))
    t = 2 * np.pi * rng.random(N)
    da = pd.DataFrame({
        "lat": 50 + r * np.sin(t) / 111.2,
        "lon": -100 + r * np.cos(t) / (111.2 * math.cos(math.radians(50))),
        "pop": 1.0,
        "area": math.pi * R**2 / N,
        "region": np.where(np.cos(t) > 0, "east", "west"),
    })
    out = effective_distances(da, {"east": "disc", "west": "disc"})
    return out.set_index(["destination", "source", "measure"]).value


def test_arithmetic_matches_closed_form(disc):
    assert disc[("disc", "disc", "arith")] == pytest.approx(128 * R / (45 * math.pi), rel=0.01)


def test_harmonic_matches_closed_form(disc):
    assert disc[("disc", "disc", "harm")] == pytest.approx(3 * math.pi * R / 16, rel=0.03)


def test_power_means_are_ordered(disc):
    vals = [disc[("disc", "disc", m)] for m in ("arith", "rho_0.5", "geom", "rho_-0.5", "harm")]
    assert vals == sorted(vals, reverse=True)


def test_symmetric(disc):
    for m in ("arith", "geom", "harm"):
        assert disc[("east", "west", m)] == pytest.approx(disc[("west", "east", m)])
