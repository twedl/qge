"""Table 7 — Unilateral reduction of import and export costs by province (%).

For each of the 10 provinces, run two simulations:
1. 10% lower IMPORT costs for that province  (column for province = 0.9)
2. 10% lower EXPORT costs for that province  (rows for province as source = 0.9)

For each, report Real Income, Real Wages, and the change in "transfer
multiplier" 1 − S_n/wL_n (i.e., I/wL = total absorption / labor income).

Direct port of Table7.m.
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
    mask = np.kron(np.eye(N), np.ones((J, 1)))
    return tauhat + mask * (1.0 - tauhat)


def _run_one(b, tauhat, target_idx: int) -> tuple[float, float, float]:
    """Solve and return (Uhat-1, realGDPhat-1, transfer-multiplier-change)
    for the targeted province, all in percent."""
    r = solve(b, tauhat=tauhat)
    out = r.outputs
    ri = (out.Uhat.ravel()[target_idx] - 1) * 100
    rw = (out.realGDPhat.ravel()[target_idx] - 1) * 100
    # transfers = ((1 - Sn/wL_new) / (1 - Sn0/wL)) - 1
    new_ratio = 1 - out.Sn[0, target_idx] / out.wL_new[0, target_idx]
    base_ratio = 1 - b.Sn0[0, target_idx] / b.wL[0, target_idx]
    tr = (new_ratio / base_ratio - 1) * 100
    return ri, rw, tr


def _print_table(rows: list[tuple[str, float, float, float]], label: str) -> None:
    print(f"\n=== {label} ===\n")
    print(f"  {'Prov':<5} {'R. Inc.':>10} {'R. Wage':>10} {'Trnsfrs':>10}")
    print(f"  {'-' * 38}")
    for name, ri, rw, tr in rows:
        print(f"  {name:<5} {ri:+10.1f} {rw:+10.1f} {tr:+10.1f}")


def main() -> None:
    raw = load_inputs()
    region_index = {r: i for i, r in enumerate(raw.regions)}
    N, J = len(raw.regions), len(raw.sectors)

    # ── 10% lower IMPORT costs (one province at a time) ──────────────────
    import_rows = []
    for name in DISPLAY_ORDER_NAMES:
        i = region_index[name]
        b = initialize(raw)
        tauhat = np.ones((N * J, N))
        tauhat[:, i] = 0.9                    # all imports into province i cost 10% less
        tauhat = _force_diagonal_one(tauhat, N, J)
        ri, rw, tr = _run_one(b, tauhat, target_idx=i)
        import_rows.append((name, ri, rw, tr))
    _print_table(import_rows, "Gains from 10% lower IMPORT costs")

    # ── 10% lower EXPORT costs (one province at a time) ──────────────────
    export_rows = []
    for name in DISPLAY_ORDER_NAMES:
        i = region_index[name]
        b = initialize(raw)
        tauhat = np.ones((N * J, N))
        tauhat[i * J: (i + 1) * J, :] = 0.9   # all exports FROM province i cost 10% less
        tauhat = _force_diagonal_one(tauhat, N, J)
        ri, rw, tr = _run_one(b, tauhat, target_idx=i)
        export_rows.append((name, ri, rw, tr))
    _print_table(export_rows, "Gains from 10% lower EXPORT costs")


if __name__ == "__main__":
    main()
