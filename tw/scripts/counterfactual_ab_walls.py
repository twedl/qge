"""Alberta counterfactual: raise AB↔RoC non-distance trade costs to AB↔WLD level.

Models "Alberta becomes as foreign as the rest of the world to/from the other
nine provinces", with own-geography preserved.

Method:
  1. Recompute baseline Head-Ries τ̄ and distance-only piece d̄^δ from the
     canada_2022 calibration (same machinery as build_trade_costs.py).
  2. For each sector j, compute the "non-distance ratio" r^j = τ̄_{AB,WLD}^j /
     d̄_{AB,WLD}^{δ^j}. This is the symmetric residual frictions component for
     AB↔WLD trade.
  3. Build a target τ̄_{AB,p}^j_new = r^j × d̄_{AB,p}^{δ^j} for each
     province p ≠ AB. (Own geography kept, WLD non-distance level imported.)
  4. Construct the tauhat shock tensor: only AB↔p pairs (both directions),
     floor at 1.0 so this is strictly a cost INCREASE.
  5. Solve the model with this shock; report % change in real income,
     real wage, and employment per province.

Run from tw/:
    TW_CALIBRATION=canada_2022 uv run python scripts/counterfactual_ab_walls.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qge.initialize import initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402
from scripts.build_trade_costs import (  # noqa: E402
    build_distance_matrix,
    distance_regression,
    head_ries_inversion,
)


DISPLAY_ORDER = ("BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL")


def build_tauhat_ab_walls(
    pi_3d: np.ndarray,
    theta: np.ndarray,
    regions: list[str],
    sectors: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Construct the (N*J, N) MATLAB-layout shock.

    Returns the shock tensor + a per-sector / per-pair diagnostic frame.
    """
    N, J = len(regions), len(sectors)
    ab = regions.index("AB")
    wld = regions.index("WLD")

    # Step 1: distance matrix and Head-Ries τ̄
    _, d_bar = build_distance_matrix(regions)
    tau_bar = head_ries_inversion(pi_3d, theta)             # (N, N, J)
    # Step 2: per-sector δ via the AT-2016 distance regression on τ̄
    delta, _ = distance_regression(tau_bar, d_bar, regions, sectors)

    # Step 3: build per-sector target non-distance ratio r^j from AB↔WLD
    # AB↔WLD non-distance ratio = τ̄_{AB,WLD}^j / d̄_{AB,WLD}^{δ^j}.
    # Head-Ries τ̄ is symmetric so τ̄_{AB,WLD} = τ̄_{WLD,AB}.
    r_target = np.full(J, np.nan)
    for j in range(J):
        tb_AB_WLD = tau_bar[ab, wld, j]
        if np.isnan(tb_AB_WLD) or np.isnan(delta[j]):
            continue
        r_target[j] = tb_AB_WLD / (d_bar[ab, wld] ** delta[j])

    # Step 4: per-pair shock
    tauhat = np.ones((N * J, N))
    rows = []
    for p, prov in enumerate(regions):
        if prov == "AB" or prov == "WLD":
            continue
        for j, sec in enumerate(sectors):
            tb_AB_p = tau_bar[ab, p, j]
            if np.isnan(tb_AB_p) or np.isnan(r_target[j]):
                continue
            tau_new = r_target[j] * (d_bar[ab, p] ** delta[j])
            factor = max(1.0, tau_new / tb_AB_p)
            # AB → p shipping (source = AB, dest = p): row ab*J + j, col p
            tauhat[ab * J + j, p] = factor
            # p → AB shipping (source = p, dest = AB): row p*J + j, col ab
            tauhat[p * J + j, ab] = factor
            rows.append({
                "sector": sec,
                "partner": prov,
                "tau_baseline": tb_AB_p,
                "tau_target": tau_new,
                "factor": factor,
                "delta": delta[j],
            })
    diag = pd.DataFrame(rows)
    return tauhat, diag


def _force_diagonal_one(tauhat: np.ndarray, N: int, J: int) -> np.ndarray:
    """Belt-and-suspenders: ensure within-region cells are 1."""
    mask = np.kron(np.eye(N), np.ones((J, 1)))
    return tauhat + mask * (1.0 - tauhat)


def main() -> None:
    raw = load_inputs()
    sectors = list(raw.sectors)
    regions = list(raw.regions)
    N, J = len(regions), len(sectors)

    # Re-derive pi[n, i, j] (destination, source, sector) from PI MATLAB layout.
    # raw.PI is (N*J, N); row k = i*J + j (source i, sector j); col n (dest).
    pi_3d = np.zeros((N, N, J))
    for i in range(N):
        for j in range(J):
            for n in range(N):
                pi_3d[n, i, j] = raw.PI[i * J + j, n]
    theta = raw.theta.ravel()

    tauhat, diag = build_tauhat_ab_walls(pi_3d, theta, regions, sectors)
    tauhat = _force_diagonal_one(tauhat, N, J)

    # ── Diagnostics ───────────────────────────────────────────────────────
    print("=== Tauhat construction ===\n")
    print(f"Affected pairs: {len(diag)} (sector × partner combinations)")
    print(f"Floored at 1 (no actual increase): "
          f"{(diag['factor'] == 1.0).sum()} / {len(diag)}\n")

    print("Per-partner mean factor (geo-mean across sectors where factor > 1):")
    sub = diag[diag['factor'] > 1.0]
    if len(sub):
        agg = sub.groupby("partner")["factor"].apply(
            lambda s: float(np.exp(np.log(s).mean()))
        ).sort_values()
        for p, f in agg.items():
            print(f"  AB↔{p}: ×{f:.3f}  ({(f-1)*100:+.1f}%)")
    print()

    # ── Solve ─────────────────────────────────────────────────────────────
    b = initialize(raw)
    res = solve(b, tauhat=tauhat, verbose=True)

    out = res.outputs
    ri = (out.Uhat.ravel()[: N - 1] - 1) * 100
    rw = (out.realGDPhat.ravel()[: N - 1] - 1) * 100
    empl = (out.lambda2.ravel() / b.lambda_.ravel() - 1) * 100

    print("\n=== Counterfactual: AB↔RoC non-distance costs raised to AB↔WLD level ===\n")
    print(f"  {'Prov':<5} {'R. Inc.':>10} {'R. Wage':>10} {'Empl':>10}")
    print(f"  {'-' * 38}")
    region_index = {r: i for i, r in enumerate(regions)}
    for name in DISPLAY_ORDER:
        i = region_index[name]
        print(f"  {name:<5} {ri[i]:+10.2f} {rw[i]:+10.2f} {empl[i]:+10.2f}")
    print(f"  {'-' * 38}")
    can_ri = 100 * (out.aggregate_Uhat - 1)
    can_rgdp = 100 * (out.aggregate_rGDPhat - 1)
    print(f"  {'CAN':<5} {can_ri:+10.2f} {can_rgdp:+10.2f} {'-':>10}")


if __name__ == "__main__":
    main()
