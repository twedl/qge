"""Figure 2 — Effect of varying ζ (degree of fiscal integration) on aggregate
welfare and aggregate real GDP.

Two panels:
  (a) Sn_exogenous = SnBar (panel a) — start from "no transfers" with
      S_n = S̄_n (observed exogenous trade imbalances including original
      transfer effect).
  (b) default Sn_exogenous (panel b) — start from baseline with the
      observed Sn_exogenous decomposition.

For each panel, rebase the initial equilibrium to the f=0 counterfactual,
then loop ζ from 0 to 0.6 in steps of 0.01, solving each time and
recording aggregate_rGDPhat and aggregate_gain.

The optimal ζ that maximizes aggregate welfare (aggregate_gain) is the
headline result. WP reports ζ* ≈ -0.24 (panel a) and ζ* ≈ -0.35 (panel b).

Output goes to stdout + a CSV at data/outputs/figure2_<panel>.csv for
plotting later. Direct port of Figure2.m.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qge.equations import model_equations  # noqa: E402
from qge.initialize import BaselineEquilibrium, initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402


def _rebase_to_no_transfers(b: BaselineEquilibrium, use_SnBar: bool
                             ) -> BaselineEquilibrium:
    """Rebase the baseline by running the f=0 counterfactual once and
    overwriting the equilibrium quantities to the new no-transfers steady
    state, as in Figure2.m lines 14-25."""
    if use_SnBar:
        b.Sn_exogenous = b.SnBar.copy()
    b.f = 0.0
    result = solve(b)
    out = result.outputs

    # Promote the counterfactual to the new baseline.
    N = b.N
    b.R = out.Rnew
    b.PI = out.PInew
    b.I = out.Inew
    b.wL = out.wL_new
    b.A = out.Anew
    b.AX = out.AXnew
    b.Fmat = out.Fmatnew
    b.Sn = out.Sn
    b.Sn0 = out.Sn.copy()
    b.lambda_ = out.lambda2
    b.mu = out.mu_new

    omega_un = b.lambda_ * b.mu
    b.omega = omega_un / omega_un.sum()
    S_provs = out.Sn[0, : N - 1]
    wL_provs = out.wL_new[0, : N - 1]
    income_per_labor_provs = 1.0 - S_provs / wL_provs
    o2_un = (b.lambda_.ravel() * b.mu.ravel() * income_per_labor_provs)
    b.omega2 = (o2_un / o2_un.sum()).reshape(N - 1, 1)
    return b


def _sweep_f(b: BaselineEquilibrium, f_grid: np.ndarray) -> pd.DataFrame:
    rows = []
    for f in f_grid:
        b.f = float(f)
        r = solve(b)
        out = r.outputs
        rows.append({"f": float(f),
                      "agg_rGDPhat_pct": 100 * (out.aggregate_rGDPhat - 1),
                      "agg_gain_pct":    100 * (out.aggregate_gain - 1),
                      "agg_Uhat_pct":    100 * (out.aggregate_Uhat - 1)})
    return pd.DataFrame(rows)


def main() -> None:
    raw = load_inputs()
    out_dir = Path(__file__).resolve().parent.parent / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    f_grid = np.arange(0.0, 0.61, 0.01)

    for panel, use_SnBar in (("a", True), ("b", False)):
        print(f"\n=== Figure 2 panel ({panel}) ===")
        b = initialize(raw)
        b = _rebase_to_no_transfers(b, use_SnBar=use_SnBar)

        df = _sweep_f(b, f_grid)
        df.to_csv(out_dir / f"figure2_panel_{panel}.csv", index=False)

        # Find the welfare-maximizing ζ (note ζ = −f in the paper's sign
        # convention; we report both)
        idx_gain = df["agg_gain_pct"].idxmax()
        idx_rgdp = df["agg_rGDPhat_pct"].idxmax()
        print(f"  ζ* (welfare maximizer)    = {-df.loc[idx_gain, 'f']:+.3f}  "
              f"with agg welfare gain {df.loc[idx_gain, 'agg_gain_pct']:+.3f}%")
        print(f"  ζ* (real-GDP maximizer)   = {-df.loc[idx_rgdp, 'f']:+.3f}  "
              f"with agg rGDP gain {df.loc[idx_rgdp, 'agg_rGDPhat_pct']:+.3f}%")
        print(f"  CSV → {out_dir.relative_to(out_dir.parent.parent)}/figure2_panel_{panel}.csv")


if __name__ == "__main__":
    main()
