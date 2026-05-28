"""Alberta separation grid: 18 scenarios (2 fiscal × 3 labour × 3 trade).

Levers:
  fiscal ∈ {on, off}
    off — T_AB set to zero, redistributed per worker to the other 9 provinces
          (transfer_exit_sn_shift in counterfactual_ab_scenarios.py)
  labour ∈ {normal, medium, frozen}
    normal — κ = 1.5 baseline for all 10 provinces
    medium — κ_AB = 0.75 (half elasticity), other 9 provinces stay at 1.5
    frozen — frozen_idx = (AB,): AB labour pinned at baseline λ
  trade ∈ {normal, +10%, WLD-walls}
    +10% — tauhat[AB↔p, all sectors] = 1.10 for all 9 RoC provinces; WLD untouched
    WLD-walls — AB↔RoC non-distance friction raised to AB↔WLD level
          (build_tauhat_ab_walls from counterfactual_ab_walls.py)

Total 2×3×3 = 18 scenarios. Output: real-income table for each.

Run from tw/:
    TW_CALIBRATION=canada_2022 uv run python scripts/counterfactual_ab_grid.py
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qge.initialize import initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402
from scripts.counterfactual_ab_walls import (  # noqa: E402
    _force_diagonal_one,
    build_tauhat_ab_walls,
)
from scripts.counterfactual_ab_scenarios import transfer_exit_sn_shift  # noqa: E402


DISPLAY_ORDER = ("BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL")

LABOUR_LEVELS = ("normal", "medium", "frozen")
TRADE_LEVELS = ("normal", "+10%", "WLD-walls")
FISCAL_LEVELS = ("on", "off")


def build_tauhat_uniform_10pct(N: int, J: int, ab_idx: int, wld_idx: int) -> np.ndarray:
    """tauhat[AB↔p, all j] = 1.10 for p in RoC; WLD pairs untouched."""
    tauhat = np.ones((N * J, N))
    # AB → p (source = AB, dest = p)
    for p in range(N):
        if p == ab_idx or p == wld_idx:
            continue
        # row block ab*J : ab*J+J  applies to source=AB, all sectors j
        tauhat[ab_idx * J: (ab_idx + 1) * J, p] = 1.10
        # p → AB (source = p, dest = AB)
        tauhat[p * J: (p + 1) * J, ab_idx] = 1.10
    return _force_diagonal_one(tauhat, N, J)


def run_one(raw, fiscal: str, labour: str, trade: str, ab_idx: int, wld_idx: int):
    """Solve one scenario. Returns dict mapping province → Uhat-1 (in %)."""
    b = initialize(raw)
    N, J = b.N, b.J
    sectors = list(raw.sectors)
    regions = list(raw.regions)

    # ─── Trade lever ───
    tauhat = None
    if trade == "normal":
        tauhat = None
    elif trade == "+10%":
        tauhat = build_tauhat_uniform_10pct(N, J, ab_idx, wld_idx)
    elif trade == "WLD-walls":
        pi_3d = np.zeros((N, N, J))
        for i in range(N):
            for j in range(J):
                for n in range(N):
                    pi_3d[n, i, j] = raw.PI[i * J + j, n]
        tauhat, _ = build_tauhat_ab_walls(pi_3d, raw.theta.ravel(),
                                           regions, sectors)
        tauhat = _force_diagonal_one(tauhat, N, J)
    else:
        raise ValueError(f"unknown trade={trade!r}")

    # ─── Fiscal lever ───
    if fiscal == "off":
        b.Sn_exogenous = b.Sn_exogenous + transfer_exit_sn_shift(b, ab_idx)
    elif fiscal != "on":
        raise ValueError(f"unknown fiscal={fiscal!r}")

    # ─── Labour lever ───
    frozen = ()
    kappa_override = None
    if labour == "normal":
        pass
    elif labour == "medium":
        kappa_override = np.full((N - 1, 1), b.kappa)
        kappa_override[ab_idx] = 0.5 * b.kappa  # κ_AB = 0.75 when baseline=1.5
    elif labour == "frozen":
        frozen = (ab_idx,)
    else:
        raise ValueError(f"unknown labour={labour!r}")

    res = solve(b, tauhat=tauhat, frozen_idx=frozen,
                kappa_override=kappa_override, verbose=False)

    region_index = {r: i for i, r in enumerate(regions)}
    ri = (res.outputs.Uhat.ravel() - 1) * 100
    out = {p: float(ri[region_index[p]]) for p in DISPLAY_ORDER}
    out["CAN"] = 100 * float(res.outputs.aggregate_Uhat - 1)
    return out


def main() -> None:
    raw = load_inputs()
    ab_idx = list(raw.regions).index("AB")
    wld_idx = list(raw.regions).index("WLD")

    print(f"Running 18-scenario grid (2 fiscal × 3 labour × 3 trade)")
    print(f"Calibration: {Path.cwd().name} via TW_CALIBRATION={list(raw.regions)[ab_idx]}/{list(raw.regions)[wld_idx]}")
    print()

    results = {}
    for fiscal, labour, trade in product(FISCAL_LEVELS, LABOUR_LEVELS, TRADE_LEVELS):
        key = (fiscal, labour, trade)
        results[key] = run_one(raw, fiscal, labour, trade, ab_idx, wld_idx)

    # ── Print as 6 mini-tables: one per (fiscal, labour) combo, columns trade ──
    for fiscal in FISCAL_LEVELS:
        for labour in LABOUR_LEVELS:
            print(f"\n=== Fiscal: {fiscal:<3}  Labour: {labour:<6}  (real income, % Uhat-1) ===")
            print(f"  {'Prov':<5} " + ''.join(f"{t:>13}" for t in TRADE_LEVELS))
            print("  " + '-' * 47)
            for p in DISPLAY_ORDER + ("CAN",):
                row = f"  {p:<5} "
                for trade in TRADE_LEVELS:
                    row += f"{results[(fiscal, labour, trade)][p]:>+12.2f}%"
                print(row)


if __name__ == "__main__":
    main()
