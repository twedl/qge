"""Head-Ries index and distance stripping in scripts/dty_replication.py.

Gravity flows X_ni = S_i M_n τ_ni^(−θ) with τ_ni = d_ni^δ · b (n ≠ i) and
τ_nn = d_nn^δ: the Head-Ries index recovers θ ln τ̄ = θδ ln d̄_ni + θ ln b, and
the FE regression strips the distance term, leaving θ ln b on every pair.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from dty_replication import head_ries, non_distance  # noqa: E402

THETA, DELTA, BORDER = 5.0, 0.8, 1.15


@pytest.fixture(scope="module")
def gravity():
    rng = np.random.default_rng(0)
    regs = list("abcdef")
    xy = dict(zip(regs, rng.uniform(0, 3000, size=(len(regs), 2))))
    d_own = dict(zip(regs, rng.uniform(30, 200, len(regs))))
    S, M = dict(zip(regs, rng.lognormal(size=len(regs)))), dict(zip(regs, rng.lognormal(size=len(regs))))
    dist = {(n, i): d_own[n] if n == i else float(np.hypot(*(xy[n] - xy[i]))) for n in regs for i in regs}
    tau = {(n, i): dist[n, i] ** DELTA * (1.0 if n == i else BORDER) for n, i in dist}
    flows = pd.DataFrame([("j", "goods", i, n, S[i] * M[n] * tau[n, i] ** -THETA) for n, i in dist],
                         columns=["ind", "kind", "origin", "dest", "X"])
    return flows, dist


def test_head_ries_recovers_log_tau_bar(gravity):
    flows, dist = gravity
    hr = head_ries(flows)
    assert len(hr) == 30
    lnd = np.log([dist[n, i] / np.sqrt(dist[n, n] * dist[i, i]) for n, i in zip(hr.n, hr.i)])
    np.testing.assert_allclose(hr.L / THETA, DELTA * lnd + np.log(BORDER), rtol=1e-12)


def test_non_distance_leaves_border_cost(gravity):
    flows, dist = gravity
    hr = head_ries(flows)
    lnd = pd.Series(np.log([dist[n, i] / np.sqrt(dist[n, n] * dist[i, i]) for n, i in zip(hr.n, hr.i)]),
                    index=hr.index)
    ngL = non_distance(hr, lnd, pd.Series("g", index=hr.index))
    np.testing.assert_allclose(ngL / THETA, np.log(BORDER), atol=1e-10)
