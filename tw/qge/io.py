"""Load the Tombe & Winter (2021) calibration from long-form parquets.

The parquets in ``data/inputs/canada_2010/`` are written by
``scripts/convert_maindata.py`` from the .mat file. This module reads them
back into NumPy arrays sized to match the original MATLAB layout — chosen
deliberately to make line-by-line comparison with the original
``model_equations.m`` straightforward during the port.

Conventions (all 0-indexed in Python; 1-indexed in MATLAB):

- ``N`` = 11 regions (10 provinces + 'WLD'); ``J`` = 22 sectors.
- ``PI`` shape ``(N*J, N)``: row k = i*J + j (source i, sector j); column n =
  destination. ``PI[i*J + j, n] = π_{n,i}^j``. Row sums over i (i.e.,
  ``PI[j::J, n].sum()``) equal 1.
- ``gamma`` shape ``(J, N*J)``: ``gamma[k, n*J + j] = γ^{k,j}_n`` (share of
  sector j-in-region-n's intermediate spending on sector k).
- ``phi`` shape ``(1, N*J)``: ``phi[0, n*J + j] = φ^j_n`` (labor share).
- ``alpha`` shape ``(J, N)``: ``alpha[j, n] = α^j_n`` (final-demand share).
- ``theta`` shape ``(J, 1)``: trade elasticity (column vector for MATLAB
  broadcast compatibility).
- ``lambda_``, ``mu``, ``transfers_share_GDP`` shape ``(N-1, 1)``: provinces
  only.
- ``surplus_gdp`` shape ``(N, 1)``: all 11 regions.
- ``tauhat_*`` shape ``(N*J, N)``: same layout as ``PI``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_CALIBRATION = (
    Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2010"
)


@dataclass(frozen=True)
class RawInputs:
    """Primitives for the Tombe & Winter (2021) calibration.

    Layouts are MATLAB-compatible (column-major flattening of (region,sector)
    flat dimensions) — see module docstring.
    """

    sectors: tuple[str, ...]
    regions: tuple[str, ...]

    PI: np.ndarray              # (N*J, N) bilateral trade shares
    IOmatrix: np.ndarray        # (J, J) national IO
    gamma: np.ndarray           # (J, N*J) regional intermediate shares
    phi: np.ndarray             # (1, N*J) labor shares
    alpha: np.ndarray           # (J, N) final demand shares
    theta: np.ndarray           # (J, 1) trade elasticity
    lambda_: np.ndarray         # (N-1, 1) employment shares (provinces)
    mu: np.ndarray              # (N-1, 1) real income index (provinces)
    surplus_gdp: np.ndarray     # (N, 1) trade surplus / GDP (all regions)
    transfers_share_GDP: np.ndarray  # (N-1, 1) exogenous transfers / GDP
    I_baseline: np.ndarray      # (1, N) converged income in MATLAB normalization
    Sn_baseline: np.ndarray     # (1, N) converged trade surplus, same units as I
    tauhat_all: np.ndarray      # (N*J, N) counterfactual cost shock
    tauhat_asym: np.ndarray     # (N*J, N)
    tauhat_dist: np.ndarray     # (N*J, N)

    @property
    def J(self) -> int:
        return len(self.sectors)

    @property
    def N(self) -> int:
        return len(self.regions)


def _long_to_array(df: pd.DataFrame, dim_cols: Sequence[str],
                   dim_labels: Sequence[Sequence[str]]) -> np.ndarray:
    """Reshape a long-form DataFrame into a dense N-D array following the
    order of ``dim_labels``. Raises on missing or unknown labels."""
    shape = tuple(len(lbl) for lbl in dim_labels)
    out = np.full(shape, np.nan, dtype=float)
    codes = []
    for col, labels in zip(dim_cols, dim_labels):
        idx_map = {v: i for i, v in enumerate(labels)}
        c = df[col].map(idx_map)
        if c.isna().any():
            unknown = sorted(df.loc[c.isna(), col].unique().tolist())
            raise ValueError(f"unknown {col!r} labels: {unknown[:5]}")
        codes.append(c.to_numpy().astype(np.int64))
    out[tuple(codes)] = df["value"].to_numpy()
    if np.isnan(out).any():
        raise ValueError(f"missing cells when reshaping {dim_cols} into {shape}")
    return out


def _pi_to_matlab(pi_3d: np.ndarray, N: int, J: int) -> np.ndarray:
    """(sector, dest, source) → (N*J, N) MATLAB layout where row=i*J+j."""
    # pi_3d[j, n, i] = π_{n,i}^j ; want out[i*J + j, n] = π_{n,i}^j
    src_sec_dest = np.transpose(pi_3d, (2, 0, 1))   # (i, j, n)
    return src_sec_dest.reshape(N * J, N)


def _gamma_to_matlab(g_3d: np.ndarray, N: int, J: int) -> np.ndarray:
    """(input_k, dest_j, region_n) → (J, N*J) where col = n*J + j."""
    # g_3d[k, j, n] = γ^{k,j}_n ; want out[k, n*J + j] = γ^{k,j}_n
    return np.transpose(g_3d, (0, 2, 1)).reshape(J, N * J)


def _phi_to_matlab(phi_2d: np.ndarray, N: int, J: int) -> np.ndarray:
    """(sector, region) → (1, N*J) where col = n*J + j."""
    return phi_2d.T.ravel().reshape(1, N * J)


def load_inputs(directory: Path | str = DEFAULT_CALIBRATION) -> RawInputs:
    """Load all primitives from parquets into a ``RawInputs`` dataclass."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"missing {directory}; run scripts/convert_maindata.py first."
        )

    sector_labels = pd.read_parquet(directory / "sector_labels.parquet")
    sectors = tuple(sector_labels["sector"].tolist())

    # Region labels: derived from a column we know contains all 11 regions.
    trade_surplus = pd.read_parquet(directory / "trade_surplus.parquet")
    regions = tuple(trade_surplus["region"].tolist())
    J = len(sectors)
    N = len(regions)

    def _load(name: str, dims: tuple[str, ...],
              labels: tuple[Sequence[str], ...]) -> np.ndarray:
        return _long_to_array(pd.read_parquet(directory / name), dims, labels)

    pi_3d = _load("bilateral_trade.parquet",
                  ("sector", "destination", "source"),
                  (sectors, regions, regions))
    PI = _pi_to_matlab(pi_3d, N, J)

    IOmatrix = _load("io_matrix.parquet",
                     ("source_sector", "dest_sector"),
                     (sectors, sectors))

    gamma_3d = _load("intermediate_share.parquet",
                     ("input_sector", "dest_sector", "region"),
                     (sectors, sectors, regions))
    gamma = _gamma_to_matlab(gamma_3d, N, J)

    phi_2d = _load("labor_share.parquet",
                   ("sector", "region"),
                   (sectors, regions))
    phi = _phi_to_matlab(phi_2d, N, J)

    alpha = _load("final_demand_share.parquet",
                  ("sector", "region"),
                  (sectors, regions))

    sd_df = pd.read_parquet(directory / "sectoral_dispersion.parquet")
    sd_df = sd_df.set_index("sector").loc[list(sectors)]
    theta = (1.0 / sd_df["value"].to_numpy()).reshape(J, 1)

    provs = regions[:N - 1]                         # 10 provinces
    lambda_df = pd.read_parquet(directory / "employment.parquet").set_index("region").loc[list(provs)]
    lambda_ = lambda_df["value"].to_numpy().reshape(N - 1, 1)

    mu_df = pd.read_parquet(directory / "income_index.parquet").set_index("region").loc[list(provs)]
    mu = mu_df["value"].to_numpy().reshape(N - 1, 1)

    surplus_gdp = trade_surplus.set_index("region").loc[list(regions), "value"
                  ].to_numpy().reshape(N, 1)

    transfers_df = pd.read_parquet(directory / "transfers_share_gdp.parquet"
                                    ).set_index("region").loc[list(provs)]
    transfers_share_GDP = transfers_df["value"].to_numpy().reshape(N - 1, 1)

    I_df = pd.read_parquet(directory / "income_baseline.parquet"
                           ).set_index("region").loc[list(regions)]
    I_baseline = I_df["value"].to_numpy().reshape(1, N)
    Sn_df = pd.read_parquet(directory / "Sn_baseline.parquet"
                            ).set_index("region").loc[list(regions)]
    Sn_baseline = Sn_df["value"].to_numpy().reshape(1, N)

    tauhats = {}
    for name in ("tauhat_all", "tauhat_asym", "tauhat_dist"):
        shock_3d = _load(f"{name}.parquet",
                         ("sector", "destination", "source"),
                         (sectors, regions, regions))
        tauhats[name] = _pi_to_matlab(shock_3d, N, J)

    return RawInputs(
        sectors=sectors, regions=regions,
        PI=PI, IOmatrix=IOmatrix, gamma=gamma, phi=phi, alpha=alpha,
        theta=theta, lambda_=lambda_, mu=mu,
        surplus_gdp=surplus_gdp, transfers_share_GDP=transfers_share_GDP,
        I_baseline=I_baseline, Sn_baseline=Sn_baseline,
        tauhat_all=tauhats["tauhat_all"],
        tauhat_asym=tauhats["tauhat_asym"],
        tauhat_dist=tauhats["tauhat_dist"],
    )
