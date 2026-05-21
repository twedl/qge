"""Table 4 — Effect of interprovincial transfers (% change).

Three sub-experiments, all reporting (Real income, Real wages, Employment) by
province plus aggregate Canada row:

1. All fiscal transfers  — f=0, Sn_exogenous = SnBar (Sn + Tn)
2. Equalization only     — Sn = Sn0 + Sn_EQ (one-off override)
3. Income-sensitive only — f=0, Sn_exogenous unchanged (default baseline)

Direct port of Table4.m. Province display order matches the paper:
BC, AB, SK, MB, ON, QC, NB, NS, PE, NL.

Run from tw/:
    uv run python scripts/table4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qge.initialize import initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import SolveResult, solve  # noqa: E402

DISPLAY_ORDER_NAMES = (
    "BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL",
)


def _display(result: SolveResult, label: str, display_idx: list[int],
              display_names: tuple[str, ...]) -> None:
    out = result.outputs
    # 1/Uhat - 1 = effect of transfers on real income.
    # 1/realGDPhat - 1 = effect of transfers on real wages.
    # lambda / lambda2 - 1 = effect of transfers on employment.
    N = result.what.shape[1]
    ri = (1.0 / out.Uhat.ravel()[: N - 1] - 1) * 100
    rw = (1.0 / out.realGDPhat.ravel()[: N - 1] - 1) * 100
    lambda_ratio = (raw_lambda.ravel() / out.lambda2.ravel() - 1) * 100

    print(f"\n=== {label} (Real income, Real wages, Employment in %) ===\n")
    print(f"  {'Prov':<5s} {'R. Inc.':>10s} {'R. Wage':>10s} {'Empl':>10s}")
    print(f"  {'-' * 38}")
    for idx, name in zip(display_idx, display_names):
        print(f"  {name:<5s} {ri[idx]:+10.1f} {rw[idx]:+10.1f} {lambda_ratio[idx]:+10.1f}")
    print(f"  {'-' * 38}")
    can_ri = 100 * (1.0 / out.aggregate_Uhat - 1)
    can_rgdp = 100 * (1.0 / out.aggregate_rGDPhat - 1)
    print(f"  {'CAN':<5s} {can_ri:+10.2f} {can_rgdp:+10.2f} {'-':>10s}")


def main() -> None:
    raw = load_inputs()
    region_index = {r: i for i, r in enumerate(raw.regions)}
    display_idx = [region_index[n] for n in DISPLAY_ORDER_NAMES]
    global raw_lambda
    raw_lambda = raw.lambda_

    # ── Sub-experiment 1: All fiscal transfers ────────────────────────────
    b = initialize(raw)
    b.Sn_exogenous = b.SnBar.copy()
    b.f = 0.0
    result1 = solve(b, verbose=True)
    _display(result1, "All fiscal transfers", display_idx, DISPLAY_ORDER_NAMES)

    # ── Sub-experiment 2: Equalization only ───────────────────────────────
    # Sn override = Sn0 + Sn_EQ (the published equalization-only impulse).
    b = initialize(raw)
    Sn_override = b.Sn0 + b.Sn_EQ
    result2 = solve(b, Sn_override=Sn_override, verbose=True)
    _display(result2, "Equalization only", display_idx, DISPLAY_ORDER_NAMES)

    # ── Sub-experiment 3: Income-sensitive transfers only ─────────────────
    b = initialize(raw)
    b.f = 0.0  # Sn_exogenous untouched ⇒ removes only income-sensitive part
    result3 = solve(b, verbose=True)
    _display(result3, "Income-sensitive transfers", display_idx, DISPLAY_ORDER_NAMES)


if __name__ == "__main__":
    main()
