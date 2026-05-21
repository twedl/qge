"""Convert maindata.mat → long-form parquets under tw/data/inputs/canada_2010/.

One-time conversion of the Tombe & Winter (2021) replication package's MATLAB
data file. Run once after extracting the Dataverse zip:

    mkdir -p /tmp/tombe_winter_2021 && \\
      unzip -o ~/Downloads/doi-10.5683-sp3-shdohy.zip -d /tmp/tombe_winter_2021/

Then from tw/:

    uv run python scripts/convert_maindata.py

## Indexing conventions

MATLAB stores arrays in column-major order. Several maindata.mat variables are
flattened across (region, sector). The conventions verified by inspection:

- `PI` (N·J, N): `PI[i*J + j, n] = π_{n,i}^j` — share of destination n's
  expenditure on sector j sourced from i. Row index k = i*J + j. Row sums
  over i (i.e., across sources for fixed (n, j)) equal 1.
- `gamma` (J, N·J): `gamma[k, n*J + j] = γ^{k,j}_n` — share of sector j (in
  region n)'s intermediate spending that goes to sector k.
- `phi` (1, N·J): `phi[n*J + j] = φ^j_n` — labor share of sector j in n.
- `theta` (J, 1): sector-specific trade elasticity θ^j (from Albrecht-Tombe
  2016).
- `tauhat_*` (N·J, N): counterfactual trade-cost shocks. Same indexing as PI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

MAT_FILE = Path("/tmp/tombe_winter_2021/DATA/maindata.mat")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2010"


def _long_3d(arr: np.ndarray, dims: tuple[str, ...], labels: tuple[tuple[str, ...], ...]
              ) -> pd.DataFrame:
    """Materialize an N-D array as a long-form DataFrame with named labels."""
    if arr.shape != tuple(len(lbl) for lbl in labels):
        raise ValueError(f"shape {arr.shape} doesn't match label sizes "
                          f"{tuple(len(l) for l in labels)}")
    idx = pd.MultiIndex.from_product(labels, names=list(dims))
    return pd.DataFrame({"value": arr.ravel()}, index=idx).reset_index()


def _reshape_pi(arr: np.ndarray, N: int, J: int) -> np.ndarray:
    """Reshape (N·J, N) — row k = i*J + j — into (sector, destination, source).

    MATLAB:   PI[i*J + j, n] = π_{n,i}^j
    Returns:  out[j, n, i]   = π_{n,i}^j   (sector, destination, source)
    """
    # arr[k, n] where k = i*J + j  →  arr.reshape(N, J, N)[i, j, n]
    src_sec_dest = arr.reshape(N, J, N)            # axes: (source i, sector j, dest n)
    return np.transpose(src_sec_dest, (1, 2, 0))   # axes: (sector j, dest n, source i)


def _reshape_phi(arr: np.ndarray, N: int, J: int) -> np.ndarray:
    """`phi[n*J + j]` flat → (sector, region) array of shape (J, N)."""
    arr = arr.ravel()
    return arr.reshape(N, J).T  # axes: (sector j, region n)


def _reshape_gamma(arr: np.ndarray, N: int, J: int) -> np.ndarray:
    """`gamma[k, n*J + j]` shape (J, N·J) → (input_sector, dest_sector, region)
    array of shape (J, J, N)."""
    # arr[k, n*J + j] = γ^{k,j}_n
    return arr.reshape(J, N, J).transpose(0, 2, 1)  # (k input, j dest, n region)


def main() -> None:
    if not MAT_FILE.exists():
        raise FileNotFoundError(
            f"missing {MAT_FILE}; extract via:\n"
            f"  mkdir -p /tmp/tombe_winter_2021 && unzip -o "
            f"~/Downloads/doi-10.5683-sp3-shdohy.zip -d /tmp/tombe_winter_2021/"
        )

    m = loadmat(MAT_FILE)
    N = int(m["N"].ravel()[0])
    J = int(m["J"].ravel()[0])
    provinces = tuple(str(x[0]) for x in m["province"].ravel())
    sectors = tuple(str(x[0]) for x in m["industry"].ravel())
    codes = tuple(str(x[0]) for x in m["industry_codes"].ravel())
    provs10 = provinces[:N - 1]                            # 10 provinces, no WLD
    assert len(provinces) == N and len(sectors) == J

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing parquets to {OUTPUT_DIR} (N={N}, J={J}) …\n")

    # ── bilateral trade shares ────────────────────────────────────────────
    pi_3d = _reshape_pi(m["PI"], N, J)
    _long_3d(pi_3d, ("sector", "destination", "source"),
             (sectors, provinces, provinces)
             ).to_parquet(OUTPUT_DIR / "bilateral_trade.parquet", index=False)
    print(f"  bilateral_trade.parquet              ({pi_3d.size:,} entries)")

    # ── national IO matrix ────────────────────────────────────────────────
    # IOmatrix[k, j] — sector k as input into sector j (per CPRHS convention)
    _long_3d(m["IOmatrix"], ("source_sector", "dest_sector"),
             (sectors, sectors)
             ).to_parquet(OUTPUT_DIR / "io_matrix.parquet", index=False)
    print(f"  io_matrix.parquet                    ({m['IOmatrix'].size:,} entries)")

    # ── regional intermediate input shares ────────────────────────────────
    gamma_3d = _reshape_gamma(m["gamma"], N, J)
    _long_3d(gamma_3d, ("input_sector", "dest_sector", "region"),
             (sectors, sectors, provinces)
             ).to_parquet(OUTPUT_DIR / "intermediate_share.parquet", index=False)
    print(f"  intermediate_share.parquet           ({gamma_3d.size:,} entries)")

    # ── labor share by (sector, region) ───────────────────────────────────
    phi_2d = _reshape_phi(m["phi"], N, J)
    _long_3d(phi_2d, ("sector", "region"), (sectors, provinces)
             ).to_parquet(OUTPUT_DIR / "labor_share.parquet", index=False)
    print(f"  labor_share.parquet                  ({phi_2d.size:,} entries)")

    # ── final demand share α^j_n ──────────────────────────────────────────
    _long_3d(m["alpha"], ("sector", "region"), (sectors, provinces)
             ).to_parquet(OUTPUT_DIR / "final_demand_share.parquet", index=False)
    print(f"  final_demand_share.parquet           ({m['alpha'].size:,} entries)")

    # ── sectoral dispersion T^j = 1/θ^j (cprhs convention) ────────────────
    theta_vec = m["theta"].ravel()
    pd.DataFrame({"sector": sectors, "value": 1.0 / theta_vec}).to_parquet(
        OUTPUT_DIR / "sectoral_dispersion.parquet", index=False
    )
    print(f"  sectoral_dispersion.parquet          ({J} entries, stored as 1/θ)")

    # ── employment shares λ_n (provinces only) ─────────────────────────────
    pd.DataFrame({"region": provs10, "value": m["lambda"].ravel()}
                 ).to_parquet(OUTPUT_DIR / "employment.parquet", index=False)
    print(f"  employment.parquet                   ({N - 1} provinces)")

    # ── real income index μ_n (provinces only) ─────────────────────────────
    pd.DataFrame({"region": provs10, "value": m["mu"].ravel()}
                 ).to_parquet(OUTPUT_DIR / "income_index.parquet", index=False)
    print(f"  income_index.parquet                 ({N - 1} provinces)")

    # ── trade surplus / GDP (all 11 regions) ───────────────────────────────
    pd.DataFrame({"region": provinces, "value": m["surplus_gdp"].ravel()}
                 ).to_parquet(OUTPUT_DIR / "trade_surplus.parquet", index=False)
    print(f"  trade_surplus.parquet                ({N} regions)")

    # ── absolute baseline income I_n and trade surplus S_n (in the
    # MATLAB-normalized units where sum(I) = 1). These are the converged
    # values from initialize_model.m's inner fixed point; we save them so
    # the Python solver can match the baseline exactly without rerunning the
    # iteration. ────────────────────────────────────────────────────────────
    pd.DataFrame({"region": provinces, "value": m["I"].ravel()}
                 ).to_parquet(OUTPUT_DIR / "income_baseline.parquet", index=False)
    print(f"  income_baseline.parquet              ({N} regions)")

    pd.DataFrame({"region": provinces, "value": m["Sn"].ravel()}
                 ).to_parquet(OUTPUT_DIR / "Sn_baseline.parquet", index=False)
    print(f"  Sn_baseline.parquet                  ({N} regions)")

    # ── exogenous fiscal transfers / GDP (provinces only) ──────────────────
    pd.DataFrame({"region": provs10,
                   "value": m["transfers_share_GDP"].ravel()}
                 ).to_parquet(OUTPUT_DIR / "transfers_share_gdp.parquet", index=False)
    print(f"  transfers_share_gdp.parquet          ({N - 1} provinces)")

    # ── counterfactual trade-cost shocks (from AT-2016) ───────────────────
    for name in ("tauhat_all", "tauhat_asym", "tauhat_dist"):
        shock = _reshape_pi(m[name], N, J)
        _long_3d(shock, ("sector", "destination", "source"),
                 (sectors, provinces, provinces)
                 ).to_parquet(OUTPUT_DIR / f"{name}.parquet", index=False)
        print(f"  {name}.parquet                  ({shock.size:,} entries)")

    # ── sector + region labels (for round-trip cross-checks) ───────────────
    pd.DataFrame({"sector": sectors, "isic_code": codes}
                 ).to_parquet(OUTPUT_DIR / "sector_labels.parquet", index=False)
    print(f"  sector_labels.parquet                ({J} sectors, with ISIC codes)")

    print("\nDone.")


if __name__ == "__main__":
    main()
