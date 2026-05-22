"""Table 8 — Bilateral trade liberalization (% change).

Three scenarios, each a 10% reduction in trade costs between selected
provinces. Reported metric: Real Income (Uhat − 1) and Employment
(lambda2/lambda − 1), with two panels each:
  (1) baseline calibration with fiscal transfers
  (2) `eliminate_transfers` baseline (S_n = S̄_n, no fiscal transfers)

Scenarios:
  - BC–AB–SK   (New West Partnership)
  - ON–QC      (Ontario–Quebec Trade and Cooperation Agreement)
  - NB–NS–PE   (Maritime provinces)

Direct port of Table8.m.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qge.initialize import eliminate_transfers, initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402

DISPLAY_ORDER_NAMES = ("BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL")

SCENARIOS = (
    ("BC–AB–SK (New West)", ("BC", "AB", "SK")),
    ("ON–QC",               ("ON", "QC")),
    ("NB–NS–PE (Maritime)", ("NB", "NS", "PE")),
)


def _force_diagonal_one(tauhat: np.ndarray, N: int, J: int) -> np.ndarray:
    mask = np.kron(np.eye(N), np.ones((J, 1)))
    return tauhat + mask * (1.0 - tauhat)


def _build_tauhat(member_idx: list[int], N: int, J: int) -> np.ndarray:
    """tauhat that reduces all pairs (n, i) within the member set by 10%."""
    tauhat = np.ones((N * J, N))
    for i in member_idx:                       # i = source
        for n in member_idx:                   # n = destination
            if n == i:
                continue
            tauhat[i * J: (i + 1) * J, n] = 0.9
    return _force_diagonal_one(tauhat, N, J)


def _print_scenario(label: str, result, b, region_index: dict[str, int]) -> None:
    out = result.outputs
    N = b.N
    ri = (out.Uhat.ravel()[: N - 1] - 1) * 100
    empl = (out.lambda2.ravel() / b.lambda_.ravel() - 1) * 100
    print(f"\n  --- {label} ---\n")
    print(f"    {'Prov':<5} {'R. Inc.':>10} {'Empl':>10}")
    for name in DISPLAY_ORDER_NAMES:
        i = region_index[name]
        print(f"    {name:<5} {ri[i]:+10.2f} {empl[i]:+10.2f}")


def main() -> None:
    raw = load_inputs()
    region_index = {r: i for i, r in enumerate(raw.regions)}
    N, J = len(raw.regions), len(raw.sectors)

    print("=" * 64)
    print("  PANEL A — Baseline calibration (with fiscal transfers)")
    print("=" * 64)
    for label, members in SCENARIOS:
        b = initialize(raw)
        tauhat = _build_tauhat([region_index[m] for m in members], N, J)
        r = solve(b, tauhat=tauhat)
        _print_scenario(label, r, b, region_index)

    print("\n" + "=" * 64)
    print("  PANEL B — `eliminate_transfers` baseline (no fiscal transfers)")
    print("=" * 64)
    for label, members in SCENARIOS:
        b = initialize(raw)
        b = eliminate_transfers(b)
        tauhat = _build_tauhat([region_index[m] for m in members], N, J)
        r = solve(b, tauhat=tauhat)
        _print_scenario(label, r, b, region_index)


if __name__ == "__main__":
    main()
