"""Alberta separation counterfactuals — four scenarios side-by-side.

  Walls only      AB↔RoC non-distance frictions raised to AB↔WLD level.
                  (Reference from counterfactual_ab_walls.py.)
  Scenario 3      Fiscal exit only. AB stops contributing to/receiving from
                  the federal transfer system; the missing flow is absorbed
                  uniformly per worker by the other 9 provinces.
  Scenario 2      Scenario 3 + labor freeze (AB labor mobility like WLD —
                  no in/out migration). Trade costs unchanged.
  Scenario 1      Scenario 2 + walls. The full "AB becomes a country" run.

Run from tw/:
    TW_CALIBRATION=canada_2022 uv run python scripts/counterfactual_ab_scenarios.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qge.initialize import initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402
from scripts.counterfactual_ab_walls import (  # noqa: E402
    _force_diagonal_one,
    build_tauhat_ab_walls,
)


DISPLAY_ORDER = ("BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL")


def transfer_exit_sn_shift(b, ab_idx: int) -> np.ndarray:
    """Per-province Sn shift to remove AB's transfer with per-worker closure.

    Returns delta_T_full (1, N) — add to b.Sn_exogenous to apply.
    """
    Tn = b.Tn.ravel().copy()
    lam = b.lambda_.ravel()
    T_AB = Tn[ab_idx]
    other_idx = [i for i in range(len(Tn)) if i != ab_idx]
    sum_other_lam = lam[other_idx].sum()

    T_new = Tn.copy()
    T_new[ab_idx] = 0.0
    for i in other_idx:
        T_new[i] = Tn[i] + T_AB * lam[i] / sum_other_lam
    assert abs(T_new.sum()) < 1e-12

    delta_T_provs = (Tn - T_new).reshape(1, -1)
    return np.concatenate([delta_T_provs, np.array([[0.0]])], axis=1)


def run_scenario(raw, label, *, walls=False, fiscal_exit=False,
                 freeze_ab_labor=False, ab_idx: int):
    """Run one counterfactual; return solver result + baseline."""
    b = initialize(raw)
    N, J = b.N, b.J
    sectors = list(raw.sectors)
    regions = list(raw.regions)

    tauhat = None
    if walls:
        # Rebuild pi[n,i,j] from PI MATLAB layout
        pi_3d = np.zeros((N, N, J))
        for i in range(N):
            for j in range(J):
                for n in range(N):
                    pi_3d[n, i, j] = raw.PI[i * J + j, n]
        tauhat, _ = build_tauhat_ab_walls(pi_3d, raw.theta.ravel(),
                                           regions, sectors)
        tauhat = _force_diagonal_one(tauhat, N, J)

    if fiscal_exit:
        b.Sn_exogenous = b.Sn_exogenous + transfer_exit_sn_shift(b, ab_idx)

    frozen = (ab_idx,) if freeze_ab_labor else ()

    res = solve(b, tauhat=tauhat, frozen_idx=frozen, verbose=False)
    if not res.converged:
        print(f"  WARNING: {label} did not converge: {res.message}")
    return res, b


def main() -> None:
    raw = load_inputs()
    region_index = {r: i for i, r in enumerate(raw.regions)}
    provs = [r for r in raw.regions if r != "WLD"]
    ab_idx = provs.index("AB")

    scenarios = [
        ("Walls only",  dict(walls=True,  fiscal_exit=False, freeze_ab_labor=False)),
        ("S3: fiscal",  dict(walls=False, fiscal_exit=True,  freeze_ab_labor=False)),
        ("S2: fis+frz", dict(walls=False, fiscal_exit=True,  freeze_ab_labor=True)),
        ("S1: full",    dict(walls=True,  fiscal_exit=True,  freeze_ab_labor=True)),
    ]

    results = []
    for label, kw in scenarios:
        res, b = run_scenario(raw, label, ab_idx=ab_idx, **kw)
        results.append((label, res, b))
        ok = "✓" if res.converged else "✗"
        ri = (res.outputs.Uhat.ravel()[1] - 1) * 100  # AB real income (idx 1 = AB)
        print(f"  {ok}  {label:<14}  AB Uhat-1 = {ri:+.2f}%   "
              f"|res|∞={np.abs(res.outputs.what1 - res.what).max():.1e}")

    # ── Real-income table ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REAL INCOME % change (Uhat - 1)")
    print("=" * 72)
    print(f"  {'Prov':<5} " + "".join(f"{lbl:>14}" for lbl, _, _ in results))
    print("  " + "-" * (5 + 14 * len(results)))
    for name in DISPLAY_ORDER:
        i = region_index[name]
        row = f"  {name:<5} "
        for _, res, _ in results:
            v = (res.outputs.Uhat.ravel()[i] - 1) * 100
            row += f"{v:+14.2f}"
        print(row)
    print("  " + "-" * (5 + 14 * len(results)))
    row = f"  {'CAN':<5} "
    for _, res, _ in results:
        v = (res.outputs.aggregate_Uhat - 1) * 100
        row += f"{v:+14.2f}"
    print(row)

    # ── Real-wage table ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REAL WAGE % change (realGDPhat - 1)")
    print("=" * 72)
    print(f"  {'Prov':<5} " + "".join(f"{lbl:>14}" for lbl, _, _ in results))
    print("  " + "-" * (5 + 14 * len(results)))
    for name in DISPLAY_ORDER:
        i = region_index[name]
        row = f"  {name:<5} "
        for _, res, _ in results:
            v = (res.outputs.realGDPhat.ravel()[i] - 1) * 100
            row += f"{v:+14.2f}"
        print(row)
    print("  " + "-" * (5 + 14 * len(results)))
    row = f"  {'CAN':<5} "
    for _, res, _ in results:
        v = (res.outputs.aggregate_rGDPhat - 1) * 100
        row += f"{v:+14.2f}"
    print(row)

    # ── Employment table ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EMPLOYMENT % change (lambda2/lambda - 1)")
    print("=" * 72)
    print(f"  {'Prov':<5} " + "".join(f"{lbl:>14}" for lbl, _, _ in results))
    print("  " + "-" * (5 + 14 * len(results)))
    for name in DISPLAY_ORDER:
        i = region_index[name]
        row = f"  {name:<5} "
        for _, res, b in results:
            v = (res.outputs.lambda2.ravel()[i] / b.lambda_.ravel()[i] - 1) * 100
            row += f"{v:+14.2f}"
        print(row)

    # Sanity check: AB employment is frozen at 0 in S2/S1
    ab_empl_s2 = (results[2][1].outputs.lambda2.ravel()[ab_idx]
                  / results[2][2].lambda_.ravel()[ab_idx] - 1) * 100
    ab_empl_s1 = (results[3][1].outputs.lambda2.ravel()[ab_idx]
                  / results[3][2].lambda_.ravel()[ab_idx] - 1) * 100
    print(f"\n  AB employment check (should be ~0 in S2/S1): "
          f"S2={ab_empl_s2:+.6f}%  S1={ab_empl_s1:+.6f}%")


if __name__ == "__main__":
    main()
