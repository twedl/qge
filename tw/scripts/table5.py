"""Table 5 — Gains from observed trade relative to autarky (% change).

Direct port of Table5.m. Constructs the autarky counterfactual by setting
all between-region trade costs τ̂ to Inf (keeping within-region τ̂ = 1).
Set Sn_exogenous = 0 (no trade ⇒ no imbalance) and f = 0 (no income-sensitive
transfers to redistribute either). The reported metric is 1/Uhat − 1, i.e.
the percent gain FROM AUTARKY to observed trade.

Run from tw/:
    uv run python scripts/table5.py
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


def main() -> None:
    raw = load_inputs()
    b = initialize(raw)
    N, J = b.N, b.J

    # Autarky: between-region τ̂ = ∞, within-region τ̂ = 1
    tauhat = np.kron(np.eye(N), np.ones((J, 1)))
    tauhat[tauhat == 0] = np.inf

    # Override fiscal & exogenous Sn to zero (autarky has no trade flows)
    b.Sn_exogenous = np.zeros((1, N))
    b.f = 0.0

    result = solve(b, tauhat=tauhat, verbose=True)
    out = result.outputs

    # Reported metric: 1/Uhat - 1 = gain FROM AUTARKY TO observed trade
    ri = (1.0 / out.Uhat.ravel()[: N - 1] - 1) * 100
    rw = (1.0 / out.realGDPhat.ravel()[: N - 1] - 1) * 100
    empl = (b.lambda_.ravel() / out.lambda2.ravel() - 1) * 100

    region_index = {r: i for i, r in enumerate(raw.regions)}
    print("\n=== Table 5: Gains from observed trade relative to autarky (%) ===\n")
    print(f"  {'Prov':<5} {'R. Inc.':>10} {'R. Wage':>10} {'Empl':>10}")
    print(f"  {'-' * 38}")
    for name in DISPLAY_ORDER_NAMES:
        i = region_index[name]
        print(f"  {name:<5} {ri[i]:+10.1f} {rw[i]:+10.1f} {empl[i]:+10.1f}")
    print(f"  {'-' * 38}")
    can_ri = 100 * (1.0 / out.aggregate_Uhat - 1)
    can_rgdp = 100 * (1.0 / out.aggregate_rGDPhat - 1)
    print(f"  {'CAN':<5} {can_ri:+10.1f} {can_rgdp:+10.1f} {'-':>10}")


if __name__ == "__main__":
    main()
