"""Table 6 — Gains from lower internal trade costs (% change).

Three sub-experiments using the τ̂ shocks shipped in maindata.mat:
1. Asymmetric Costs   — remove trade-cost asymmetries  (tauhat_asym)
2. Non-Distance Costs — remove non-distance costs       (tauhat_dist)
3. All Internal Costs — remove all internal trade costs (tauhat_all)

Reported metric: Uhat − 1 (effect of LOWERING trade costs; positive = gain).

Direct port of Table6.m. Run from tw/:
    uv run python scripts/table6.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qge.initialize import initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402

DISPLAY_ORDER_NAMES = ("BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL")


def _force_diagonal_one(tauhat: np.ndarray, N: int, J: int) -> np.ndarray:
    """tauhat = tauhat + kron(I_N, ones(J,1)) * (1 - tauhat) — ensures
    diagonal blocks (within-region) are 1."""
    mask = np.kron(np.eye(N), np.ones((J, 1)))    # 1 on (n*J+j, n) cells
    return tauhat + mask * (1.0 - tauhat)


def _display(label: str, result, b, region_index) -> None:
    out = result.outputs
    N = b.N
    ri = (out.Uhat.ravel()[: N - 1] - 1) * 100
    rw = (out.realGDPhat.ravel()[: N - 1] - 1) * 100
    empl = (out.lambda2.ravel() / b.lambda_.ravel() - 1) * 100
    print(f"\n=== {label} ===\n")
    print(f"  {'Prov':<5} {'R. Inc.':>10} {'R. Wage':>10} {'Empl':>10}")
    print(f"  {'-' * 38}")
    for name in DISPLAY_ORDER_NAMES:
        i = region_index[name]
        print(f"  {name:<5} {ri[i]:+10.1f} {rw[i]:+10.1f} {empl[i]:+10.1f}")
    print(f"  {'-' * 38}")
    can_ri = 100 * (out.aggregate_Uhat - 1)
    can_rgdp = 100 * (out.aggregate_rGDPhat - 1)
    print(f"  {'CAN':<5} {can_ri:+10.1f} {can_rgdp:+10.1f} {'-':>10}")


def main() -> None:
    raw = load_inputs()
    region_index = {r: i for i, r in enumerate(raw.regions)}
    N, J = len(raw.regions), len(raw.sectors)

    # ── 1. Remove asymmetric trade costs ────────────────────────────────
    b = initialize(raw)
    tauhat = raw.tauhat_asym.copy()
    tauhat[:, N - 1] = 1.0                       # WLD column: no change
    tauhat[(N - 1) * J: N * J, :] = 1.0           # WLD rows: no change
    tauhat = _force_diagonal_one(tauhat, N, J)
    r1 = solve(b, tauhat=tauhat, verbose=True)
    _display("Removing asymmetric trade costs", r1, b, region_index)

    # ── 2. Remove non-distance trade costs ──────────────────────────────
    b = initialize(raw)
    tauhat = raw.tauhat_dist.copy()
    tauhat[:, N - 1] = 1.0
    tauhat[(N - 1) * J: N * J, :] = 1.0
    tauhat = _force_diagonal_one(tauhat, N, J)
    r2 = solve(b, tauhat=tauhat, verbose=True)
    _display("Removing non-distance trade costs", r2, b, region_index)

    # ── 3. Remove all internal trade costs ──────────────────────────────
    b = initialize(raw)
    tauhat = np.ones((N * J, N))
    tauhat[: (N - 1) * J, : N - 1] = raw.tauhat_all[: (N - 1) * J, : N - 1]
    tauhat = _force_diagonal_one(tauhat, N, J)
    r3 = solve(b, tauhat=tauhat, verbose=True)
    _display("Removing all internal trade costs", r3, b, region_index)


if __name__ == "__main__":
    main()
