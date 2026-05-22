"""Figure 3 — Within-province effect of fiscal transfers, by sector.

For each of two provinces (Alberta — net contributor; Prince Edward Island —
net recipient), plot per-sector output change vs the closed-economy
upstreamness measure from initialize. Counterfactual: remove all fiscal
transfers (Sn_exogenous = SnBar, f = 0).

For each sector j and the chosen province p:
    y = 100 · log(Rnj[j, p] / (Rnjnew[j, p] · Phat[j, p]))

i.e., the deflated change in sector output (inverted so that "removing
transfers" gain shows positive when sector p benefits from transfers).

Output: data CSV with (sector, isic_code, upstreamness, AB_change,
PE_change). No matplotlib here — keep scientific output independent of
plotting. Plot the result however you prefer downstream.

Direct port of Figure3.m.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from qge.initialize import initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402


def main() -> None:
    raw = load_inputs()
    region_index = {r: i for i, r in enumerate(raw.regions)}
    idx_AB = region_index["AB"]
    idx_PE = region_index["PE"]

    # Counterfactual: remove all fiscal transfers
    b = initialize(raw)
    b.Sn_exogenous = b.SnBar.copy()
    b.f = 0.0
    result = solve(b, verbose=True)
    out = result.outputs

    # Baseline sector output Rnj (J, N) and counterfactual Rnjnew (J, N)
    J = b.J
    Rnj = b.R.reshape(b.N, J).T          # (J, N)
    Rnjnew = out.Rnjnew                   # (J, N)
    Phat = out.Phat                       # (J, N)

    # Deflated inverse change: 100 * log(Rnj / Rnjnew / Phat) for AB
    # 100 * log((Rnj / Rnjnew) / Phat) for PE — same math, different parenthesization in Figure3.m
    # Both reduce to 100 * log(Rnj / (Rnjnew * Phat))
    AB_change = 100 * np.log(Rnj[:, idx_AB] / (Rnjnew[:, idx_AB] * Phat[:, idx_AB]))
    PE_change = 100 * np.log(Rnj[:, idx_PE] / (Rnjnew[:, idx_PE] * Phat[:, idx_PE]))

    # Load sector ISIC codes
    sector_labels = pd.read_parquet(
        Path(__file__).resolve().parent.parent / "data" / "inputs"
        / "canada_2010" / "sector_labels.parquet"
    )

    out_df = pd.DataFrame({
        "sector":         list(raw.sectors),
        "isic_code":      sector_labels["isic_code"].tolist(),
        "upstreamness":   b.upstream_fally.ravel(),
        "AB_pct_change":  AB_change,
        "PE_pct_change":  PE_change,
    })

    out_dir = Path(__file__).resolve().parent.parent / "data" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figure3.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path.relative_to(out_dir.parent.parent)}\n")

    print(out_df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
