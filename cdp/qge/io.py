"""Input loading and data.m transformations for the CDP Base_Year.

The parquet files emitted by ``scripts/convert_cdp_txt.py`` are close to
the raw .txt sources. ``load_inputs`` here reads them, applies the
transformations from the MATLAB ``data.m`` (build the (J, N) γ, B matrices,
the per-region IO block matrix G, derive Din / VAL / VAR / α / B_n / ι /
S from xbilat and the value-added shares), and returns a ``RawInputs``
dataclass with everything the solver needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

INPUTS_ROOT = Path(__file__).resolve().parent.parent / "data" / "inputs"


@dataclass(frozen=True)
class RawInputs:
    """All initial-year inputs to the CDP Base_Year solver.

    Shapes follow the MATLAB conventions:
    * ``J`` = number of sectors (22 for CDP year-2000)
    * ``N`` = number of regions (50 US states + 37 countries = 87)
    * ``R`` = number of US states (50)
    * ``C`` = ``N − R``

    Arrays:
    * ``xbilat``     ``(J*N, N)``  bilateral trade flows, stacked by sector;
                                    rows = (sector, destination), cols = source
    * ``Din``        ``(J*N, N)``  bilateral trade shares = xbilat / row sums
    * ``gamma``      ``(J, N)``    value-added share of gross output
    * ``B``          ``(J, N)``    structures share in value added, replicated
                                    along sectors
    * ``G``          ``(J*N, J)``  IO coefficients scaled by (1 − γ);
                                    rows = (region, source_sector), cols = dest_sector
    * ``VALjn0``     ``(J, N)``    labor compensation by (sector, region)
    * ``VARjn0``     ``(J, N)``    payments to structures by (sector, region)
    * ``alphas``     ``(J, N)``    final-demand shares (CDP makes this region-invariant)
    * ``VAR``        ``(N,)``      regional payments to structures
    * ``Bn``         ``(N,)``      regional trade-balance shock
    * ``io``         ``(N,)``      portfolio share ι
    * ``Sn``         ``(N,)``      trade surplus (≡ 0 by construction at baseline)
    * ``T``          ``(J,)``      1/θ_j, sectoral dispersion
    * ``GO_check``   ``(J, N)``    gross output recomputed from xbilat row sums
    * ``sectors``    tuple of length J — sector labels
    * ``regions``    tuple of length N — region labels
    """

    xbilat: np.ndarray
    Din: np.ndarray
    gamma: np.ndarray
    B: np.ndarray
    G: np.ndarray
    VALjn0: np.ndarray
    VARjn0: np.ndarray
    alphas: np.ndarray
    VAR: np.ndarray
    Bn: np.ndarray
    io: np.ndarray
    Sn: np.ndarray
    T: np.ndarray
    GO_check: np.ndarray
    sectors: tuple[str, ...]
    regions: tuple[str, ...]

    @property
    def J(self) -> int:
        return len(self.sectors)

    @property
    def N(self) -> int:
        return len(self.regions)

    @property
    def R(self) -> int:
        return 50  # CDP fixed number of US states


# ---------------------------------------------------------------- loader


def _long_to_array(df: pd.DataFrame, dims: tuple[str, ...], labels: dict[str, tuple[str, ...]]) -> np.ndarray:
    """Long-form DataFrame → dense numpy array with axis order ``dims``."""
    idx_maps = {d: {label: i for i, label in enumerate(labels[d])} for d in dims}
    shape = tuple(len(labels[d]) for d in dims)
    arr = np.zeros(shape, dtype=np.float64)
    codes = [df[d].map(idx_maps[d]).to_numpy() for d in dims]
    arr[tuple(codes)] = df["value"].to_numpy(dtype=np.float64)
    return arr


def load_inputs(directory: Path = INPUTS_ROOT / "cdp_2000") -> RawInputs:
    """Read the six parquet files and run the data.m transformations."""
    directory = Path(directory)

    trade = pd.read_parquet(directory / "bilateral_trade.parquet")
    gamma_df = pd.read_parquet(directory / "value_added_share.parquet")
    B_df = pd.read_parquet(directory / "structures_share.parquet")
    io_df = pd.read_parquet(directory / "io_coefficients.parquet")
    T_df = pd.read_parquet(directory / "sectoral_dispersion.parquet")
    go_df = pd.read_parquet(directory / "gross_output.parquet")

    sectors = tuple(pd.unique(gamma_df["sector"]))
    regions = tuple(pd.unique(gamma_df["region"]))
    countries = tuple(pd.unique(io_df["country"]))
    J, N = len(sectors), len(regions)
    R = 50  # fixed for CDP

    # xbilat (J, N_dest, N_source) → flatten to (J*N, N)
    xbilat_3d = _long_to_array(
        trade, ("sector", "destination", "source"),
        {"sector": sectors, "destination": regions, "source": regions},
    )
    xbilat = xbilat_3d.reshape(J * N, N)

    # gamma (J, N)
    gamma = _long_to_array(
        gamma_df, ("sector", "region"),
        {"sector": sectors, "region": regions},
    )

    # B (J, N): replicate the (N,) structures share across J sector rows
    B_vec = _long_to_array(
        B_df.assign(_dummy="x"), ("region",),
        {"region": regions},
    )
    B = np.tile(B_vec, (J, 1))

    # IO blocks (38, J, J): one block per country (US + 37 foreign)
    io_blocks = _long_to_array(
        io_df, ("country", "source_sector", "dest_sector"),
        {"country": countries, "source_sector": sectors, "dest_sector": sectors},
    )
    # Broadcast: 50 US states share the US block (countries[0]).
    IO = np.empty((J * N, J))
    for n in range(N):
        block_idx = 0 if n < R else (n - R + 1)
        IO[n * J:(n + 1) * J, :] = io_blocks[block_idx]

    # G[n*J + i, k] = (1 - γ[k, n]) · IO[n*J + i, k]
    G = np.empty_like(IO)
    for n in range(N):
        G[n * J:(n + 1) * J, :] = (1 - gamma[:, n])[None, :] * IO[n * J:(n + 1) * J, :]

    # Non-tradable US-state diagonal fill is baked into the parquet
    # already (convert_cdp_txt.build_bilateral_trade). Verify GO consistency.
    GO_target = _long_to_array(
        go_df, ("sector", "region"),
        {"sector": sectors, "region": regions},
    )
    GO_check = np.zeros((J, N))
    for j in range(J):
        GO_check[j, :] = xbilat[j * N:(j + 1) * N, :].sum(axis=0)

    # Din
    row_sums = xbilat.sum(axis=1, keepdims=True)
    Din = np.where(row_sums > 0, xbilat / np.maximum(row_sums, 1e-30), 0.0)

    # Aggregate deficits Bn = sum_j (exports[j, n] − imports[j, n])
    Bn = np.zeros(N)
    for j in range(J):
        block = xbilat[j * N:(j + 1) * N, :]
        M = block.sum(axis=1)        # imports to each dest in sector j
        E = block.sum(axis=0)        # exports from each source in sector j
        Bn += E - M

    # X0[j, n] = sum_m xbilat[j, n, m] (sum over sources → expenditure at n)
    X0 = np.zeros((J, N))
    for j in range(J):
        X0[j, :] = xbilat[j * N:(j + 1) * N, :].sum(axis=1)

    # Total expenditure (sector, destination) flattened row-major. The
    # MATLAB reshape(X0', 1, J*N) is X0' read column-major = X0 read
    # row-major, so PQ_vec[j*N + n_dest] = X0[j, n_dest].
    PQ_vec = X0.flatten()
    Exjn0 = np.zeros((J, N))
    for n in range(N):
        DP0 = Din[:, n] * PQ_vec
        for j in range(J):
            Exjn0[j, n] = DP0[j * N:(j + 1) * N].sum()

    VALjn0 = gamma * (1 - B) * Exjn0
    VARjn0 = (B / (1 - B)) * VALjn0
    VAL = VALjn0.sum(axis=0)
    VAR = VARjn0.sum(axis=0)
    VA = VAR + VAL

    # Final-demand share α. CDP's data.m makes α region-invariant: the
    # sector-wise residual (X0 - intermediate uses) summed across regions
    # is divided by the total of (VA - Bn), then tiled across regions.
    aux2 = np.zeros((J, N))
    for n in range(N):
        G_n = G[n * J:(n + 1) * J, :]  # (J, J)
        # E[:, n] in data.m is the (J,) vector of column sums of xbilat for source n
        E_n = np.zeros(J)
        for j in range(J):
            E_n[j] = xbilat[j * N:(j + 1) * N, n].sum()
        aux2[:, n] = X0[:, n] - G_n @ E_n
    sector_total = aux2.sum(axis=1)               # (J,)
    total_va_minus_def = (VA - Bn).sum()
    alphas = np.tile((sector_total / total_va_minus_def)[:, None], (1, N))

    # Iotas — make Sn ≡ 0 at baseline by construction
    Chi = VAR.sum()
    io = (VAR - Bn) / Chi
    Sn = Bn - VAR + io * Chi   # = 0 ✓

    # Sectoral dispersion
    T = _long_to_array(
        T_df, ("sector",),
        {"sector": sectors},
    )

    return RawInputs(
        xbilat=xbilat, Din=Din, gamma=gamma, B=B, G=G,
        VALjn0=VALjn0, VARjn0=VARjn0, alphas=alphas,
        VAR=VAR, Bn=Bn, io=io, Sn=Sn, T=T,
        GO_check=GO_check,
        sectors=sectors, regions=regions,
    )
