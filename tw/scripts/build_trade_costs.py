"""Build TW 2021 trade-cost shocks (tauhat_all/asym/dist) from canada_2022 data.

Methodology follows Albrecht & Tombe (2016), "Internal trade, productivity and
interconnected industries", CJE 49(1) — the paper TW 2021 directly imports
trade-cost shocks from. The four steps map to AT-2016 Section 2.2-2.3 + online
Appendix B:

  1. Head-Ries inversion (per sector j):
       τ̄_{ni}^j = ((π_{nn}^j π_{ii}^j) / (π_{ni}^j π_{in}^j))^(1/(2θ^j))

  2. Asymmetry identification (Waugh-style gravity regression):
       ln(π_{ni}^j / π_{nn}^j) = γ^j ln(d̄_{ni}) + ι_n^j + η_i^j + ε
     → ln(t_i^j / t_ref^j) = -(ι_i + η_i) / θ^j  (exporter-specific cost,
       identified up to a reference category)

  3. Distance decomposition of the symmetric component:
       ln(τ̄_{ni}^j) = δ^j ln(d̄_{ni}) + ι_n^j + η_i^j + ε
     → distance-attributable τ̄^{dist}_{ni}^j = d̄_{ni}^{δ^j}

  4. Counterfactual shocks (τ_{ni}^j = τ̄_{ni}^j · √(t_i^j / t_n^j) is the
     directional bilateral cost):
       - tauhat_all[i→n]  = 1 / τ_{ni}^j                (full removal)
       - tauhat_asym[i→n] = min(1, t_n^j / t_i^j)        (drop expensive
                                                          direction down)
       - tauhat_dist[i→n] = d̄_{ni}^{δ^j} / τ̄_{ni}^j     (keep distance,
                                                          drop non-distance
                                                          residual)

Distance d̄_{ni} = d_{ni} / √(d_{nn} d_{ii}) — bilateral over within-province.
d_{nn} via Mayer-Zignago: (2/3) √(area_n / π).

Outputs (canada_2022/):
  - tauhat_all.parquet
  - tauhat_asym.parquet
  - tauhat_dist.parquet
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# Province population-weighted centroids (matches cprhs/scripts/akt_replication.py).
PROVINCE_LATLON: dict[str, tuple[float, float]] = {
    "AB":  (51.700, -114.000),  # Calgary-Edmonton mean
    "BC":  (49.250, -123.100),  # Vancouver
    "MB":  (49.895,  -97.138),  # Winnipeg
    "NB":  (45.948,  -66.673),  # Fredericton/Moncton
    "NL":  (47.617,  -52.703),  # St. John's region
    "NS":  (44.873,  -63.255),  # Halifax region
    "ON":  (43.700,  -79.400),  # Toronto region
    "PE":  (46.288,  -63.130),  # Charlottetown
    "QC":  (45.700,  -73.300),  # Montreal corridor
    "SK":  (50.450, -104.617),  # Regina-Saskatoon mean
    "WLD": (39.500,  -98.350),  # US population-weighted centroid as ROW proxy
}

# Provincial area in km² (StatCan).
PROVINCE_AREA_KM2: dict[str, float] = {
    "AB":    661_848,
    "BC":    944_735,
    "MB":    647_797,
    "NB":     72_908,
    "NL":    405_212,
    "NS":     55_284,
    "ON":  1_076_395,
    "PE":      5_660,
    "QC":  1_542_056,
    "SK":    651_036,
    "WLD": 50_000_000,  # rough ROW landmass; pairs involving WLD are masked
                        # by TW Tables 5-8 so the precise value doesn't matter
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_distance_matrix(regions: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (d_raw, d_bar) where d_bar[n,i] = d_ni / √(d_nn d_ii)."""
    N = len(regions)
    d = np.zeros((N, N))
    for n, rn in enumerate(regions):
        lat_n, lon_n = PROVINCE_LATLON[rn]
        for i, ri in enumerate(regions):
            if n == i:
                d[n, i] = (2.0 / 3.0) * math.sqrt(PROVINCE_AREA_KM2[rn] / math.pi)
            else:
                lat_i, lon_i = PROVINCE_LATLON[ri]
                d[n, i] = haversine_km(lat_n, lon_n, lat_i, lon_i)
    d_bar = d / np.sqrt(np.outer(np.diag(d), np.diag(d)))
    return d, d_bar


def head_ries_inversion(pi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """τ̄_{ni}^j = ((π_{nn} π_{ii}) / (π_{ni} π_{in}))^(1/(2θ_j)).

    Args:
      pi:    shape (N, N, J), pi[n, i, j] = π_{n,i}^j (dest n's share spent on source i)
      theta: shape (J,)
    Returns:
      tau_bar: shape (N, N, J), NaN where any input π ≤ 0.
    """
    N, _, J = pi.shape
    tau_bar = np.full((N, N, J), np.nan)
    for j in range(J):
        theta_j = theta[j]
        for n in range(N):
            for i in range(N):
                if n == i:
                    tau_bar[n, i, j] = 1.0
                    continue
                p_ni = pi[n, i, j]
                p_in = pi[i, n, j]
                p_nn = pi[n, n, j]
                p_ii = pi[i, i, j]
                if min(p_ni, p_in, p_nn, p_ii) <= 0:
                    continue
                tau_bar[n, i, j] = ((p_nn * p_ii) / (p_ni * p_in)) ** (1.0 / (2.0 * theta_j))
    return tau_bar


def _gravity_regression(
    y: np.ndarray, logd: np.ndarray, n_idx: np.ndarray, i_idx: np.ndarray, N: int,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Run y_{ni} = const + γ logd_{ni} + ι_n + η_i + ε for one sector.

    Returns (const, gamma, imp_FE, exp_FE, r2, ss_res).
    """
    n_obs = len(y)
    # Design matrix columns: [1, logd, imp_1, ..., imp_{N-1}, exp_1, ..., exp_{N-1}]
    # imp_r dummy = 1 if importer == r, for r = 1..N-1 (reference category r=0 dropped).
    X = np.zeros((n_obs, 2 + 2 * (N - 1)))
    X[:, 0] = 1.0
    X[:, 1] = logd
    for r in range(1, N):
        X[n_idx == r, 1 + r] = 1.0          # imp_r column at index 1+r
        X[i_idx == r, 1 + (N - 1) + r] = 1.0  # exp_r column
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    const = float(beta[0])
    gamma = float(beta[1])
    # imp_r and exp_r FE values for r=1..N-1; reference (r=0) is implicit 0.
    imp_fe = np.concatenate(([0.0], beta[2:1 + N]))
    exp_fe = np.concatenate(([0.0], beta[1 + N:]))
    return const, gamma, imp_fe, exp_fe, r2


def waugh_regression(
    pi: np.ndarray, d_bar: np.ndarray, theta: np.ndarray,
    regions: list[str], sectors: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Identify exporter-specific cost t_i^j by sector.

    Reference category is regions[0]; t_ref^j = 1 by construction.

    Returns (t, coefs) where t has shape (N, J) and coefs is a per-sector
    diagnostic frame.
    """
    N, _, J = pi.shape
    t = np.ones((N, J))
    rows = []
    for j in range(J):
        theta_j = theta[j]
        obs = []
        for n in range(N):
            for i in range(N):
                if n == i:
                    continue
                if pi[n, i, j] <= 0 or pi[n, n, j] <= 0:
                    continue
                obs.append((n, i, math.log(pi[n, i, j] / pi[n, n, j]), math.log(d_bar[n, i])))
        if len(obs) < 2 * N:
            rows.append((sectors[j], np.nan, np.nan, len(obs)))
            continue
        obs_arr = np.array(obs)
        n_idx = obs_arr[:, 0].astype(int)
        i_idx = obs_arr[:, 1].astype(int)
        y = obs_arr[:, 2]
        logd = obs_arr[:, 3]
        _, gamma, imp_fe, exp_fe, r2 = _gravity_regression(y, logd, n_idx, i_idx, N)
        # AT-2016 appendix: ln(t_r/t_ref) = -(ι_r + η_r) / θ
        log_t = -(imp_fe + exp_fe) / theta_j
        t[:, j] = np.exp(log_t)
        rows.append((sectors[j], gamma, r2, len(obs)))
    coefs = pd.DataFrame(rows, columns=["sector", "gamma", "r2", "n_obs"])
    return t, coefs


def distance_regression(
    tau_bar: np.ndarray, d_bar: np.ndarray, regions: list[str], sectors: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-sector regression ln(τ̄_{ni}) = δ^j ln(d̄_{ni}) + ι_n + η_i + ε.

    Returns (delta, coefs) — δ^j per sector + diagnostic frame.
    """
    N, _, J = tau_bar.shape
    delta = np.full(J, np.nan)
    rows = []
    for j in range(J):
        obs = []
        for n in range(N):
            for i in range(N):
                if n == i:
                    continue
                tb = tau_bar[n, i, j]
                if np.isnan(tb) or tb <= 0:
                    continue
                obs.append((n, i, math.log(tb), math.log(d_bar[n, i])))
        if len(obs) < 2 * N:
            rows.append((sectors[j], np.nan, np.nan, len(obs)))
            continue
        obs_arr = np.array(obs)
        n_idx = obs_arr[:, 0].astype(int)
        i_idx = obs_arr[:, 1].astype(int)
        y = obs_arr[:, 2]
        logd = obs_arr[:, 3]
        _, gamma, _, _, r2 = _gravity_regression(y, logd, n_idx, i_idx, N)
        delta[j] = gamma
        rows.append((sectors[j], gamma, r2, len(obs)))
    coefs = pd.DataFrame(rows, columns=["sector", "delta", "r2", "n_obs"])
    return delta, coefs


def build_shocks(
    tau_bar: np.ndarray, t: np.ndarray, delta: np.ndarray, d_bar: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply AT-2016 counterfactuals to produce three shock tensors.

    Indexing convention (matches canada_2010 long-form parquet):
      arr[i_source, n_dest, j_sector] = shock for shipment from i to n in sector j.

    Defaults to 1 (no shock) wherever any input is NaN.
    """
    N, _, J = tau_bar.shape
    tauhat_all = np.ones((N, N, J))
    tauhat_asym = np.ones((N, N, J))
    tauhat_dist = np.ones((N, N, J))
    for i in range(N):
        for n in range(N):
            if i == n:
                continue
            for j in range(J):
                tb = tau_bar[n, i, j]
                if np.isnan(tb):
                    continue
                t_i, t_n = t[i, j], t[n, j]
                tau = tb * math.sqrt(t_i / t_n)
                # Cap at 1: τ < 1 (sub-frictional bilateral cost) is an
                # artefact of regression noise; shock should never raise cost.
                tauhat_all[i, n, j] = min(1.0, 1.0 / tau)
                tauhat_asym[i, n, j] = min(1.0, t_n / t_i)
                if not np.isnan(delta[j]):
                    # Cap at 1: removing non-distance can only reduce or leave cost
                    # unchanged (matches canada_2010 maindata.mat convention).
                    tauhat_dist[i, n, j] = min(1.0, (d_bar[n, i] ** delta[j]) / tb)
    return {
        "tauhat_all":  tauhat_all,
        "tauhat_asym": tauhat_asym,
        "tauhat_dist": tauhat_dist,
    }


def reshape_to_long(
    arr: np.ndarray, sectors: list[str], regions: list[str],
) -> pd.DataFrame:
    """arr[i, n, j] → long-form (sector, destination, source, value)."""
    N = len(regions); J = len(sectors)
    rows = []
    for i, src in enumerate(regions):
        for n, dst in enumerate(regions):
            for j, sec in enumerate(sectors):
                rows.append((sec, dst, src, arr[i, n, j]))
    return pd.DataFrame(rows, columns=["sector", "destination", "source", "value"])


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2022"

    # ── Load inputs ───────────────────────────────────────────────────────
    bt = pd.read_parquet(root / "bilateral_trade.parquet")
    sd = pd.read_parquet(root / "sectoral_dispersion.parquet")
    sl = pd.read_parquet(root / "sector_labels.parquet")

    sectors = sl["sector"].tolist()
    regions = sorted(bt["destination"].unique())
    # Reorder regions to match canada_2010 (provinces alphabetical, WLD last).
    provinces = [r for r in regions if r != "WLD"]
    regions = sorted(provinces) + (["WLD"] if "WLD" in regions else [])
    N, J = len(regions), len(sectors)

    theta = np.array([1.0 / float(sd.loc[sd.sector == s, "value"].iloc[0]) for s in sectors])

    # Reshape bilateral_trade → pi[n, i, j]
    reg_idx = {r: k for k, r in enumerate(regions)}
    sec_idx = {s: k for k, s in enumerate(sectors)}
    pi = np.zeros((N, N, J))
    for r in bt.itertuples(index=False):
        pi[reg_idx[r.destination], reg_idx[r.source], sec_idx[r.sector]] = r.value

    print(f"Loaded {N} regions, {J} sectors")
    print(f"  pi rowsums (should be ~1 over source): "
          f"mean={pi.sum(axis=1).mean():.4f}, "
          f"min={pi.sum(axis=1).min():.4f}, max={pi.sum(axis=1).max():.4f}")
    print()

    # ── Step 1: distance matrix ───────────────────────────────────────────
    d_raw, d_bar = build_distance_matrix(regions)
    interprov = [(n, i) for n in range(N - 1) for i in range(N - 1) if n != i]
    d_bar_offdiag = np.array([d_bar[n, i] for n, i in interprov])
    print(f"Distance d̄_ni (interprovincial only): "
          f"min={d_bar_offdiag.min():.2f}, max={d_bar_offdiag.max():.2f}, "
          f"mean={d_bar_offdiag.mean():.2f}")

    # ── Step 2: Head-Ries inversion ───────────────────────────────────────
    tau_bar = head_ries_inversion(pi, theta)
    valid = ~np.isnan(tau_bar) & (np.indices(tau_bar.shape)[0] != np.indices(tau_bar.shape)[1])
    print(f"Head-Ries τ̄: {valid.sum()} valid off-diagonal cells / {(N*(N-1))*J} possible")
    interprov_valid = valid[:N-1, :N-1, :]
    if interprov_valid.any():
        print(f"  interprov τ̄ range: [{tau_bar[:N-1,:N-1,:][interprov_valid].min():.3f}, "
              f"{tau_bar[:N-1,:N-1,:][interprov_valid].max():.3f}], "
              f"mean={tau_bar[:N-1,:N-1,:][interprov_valid].mean():.3f}")
    print()

    # ── Step 3: Waugh regression for asymmetry ────────────────────────────
    t_ij, waugh_coefs = waugh_regression(pi, d_bar, theta, regions, sectors)
    print("Waugh regression (per sector):")
    print(f"  {'sector':<25} {'gamma':>8} {'r2':>6} {'n_obs':>6}")
    for _, r in waugh_coefs.iterrows():
        g = f"{r.gamma:>8.3f}" if not np.isnan(r.gamma) else "      —"
        r2 = f"{r.r2:>6.3f}" if not np.isnan(r.r2) else "    —"
        print(f"  {r.sector[:25]:<25} {g} {r2} {r.n_obs:>6}")
    print()
    # Exporter-cost summary
    print("Exporter-specific costs t_i (avg across sectors, ratio to reference):")
    t_avg = np.exp(np.nanmean(np.log(t_ij), axis=1))
    for k, r in enumerate(regions):
        print(f"  {r}: t̄={t_avg[k]:.3f}")
    print()

    # ── Step 4: distance regression on τ̄ ─────────────────────────────────
    delta, dist_coefs = distance_regression(tau_bar, d_bar, regions, sectors)
    print("Distance regression (per sector):")
    print(f"  {'sector':<25} {'delta':>8} {'r2':>6} {'n_obs':>6}")
    for _, r in dist_coefs.iterrows():
        d = f"{r.delta:>8.3f}" if not np.isnan(r.delta) else "      —"
        r2 = f"{r.r2:>6.3f}" if not np.isnan(r.r2) else "    —"
        print(f"  {r.sector[:25]:<25} {d} {r2} {r.n_obs:>6}")
    print(f"  AT-2016 reports δ typically 0.20 across sectors")
    print()

    # ── Step 5: build shocks ──────────────────────────────────────────────
    shocks = build_shocks(tau_bar, t_ij, delta, d_bar)
    for name, arr in shocks.items():
        df = reshape_to_long(arr, sectors, regions)
        off = df[df.source != df.destination]
        out_path = root / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"Wrote {out_path.name}: {len(df)} rows; "
              f"off-diag mean={off['value'].mean():.4f}, "
              f"min={off['value'].min():.4f}, max={off['value'].max():.4f}")


if __name__ == "__main__":
    main()
