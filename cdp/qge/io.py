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

# Fixed in this calibration: 50 US states sit at the front of the region
# vector, followed by 37 foreign countries. The IO blocks layout puts the
# US block at index 0 and country k at index k + 1 (one shared US block
# for all 50 states + 37 individual foreign blocks = 38 blocks total).
N_US_STATES = 50


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
        return N_US_STATES


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
    R = N_US_STATES

    xbilat_3d = _long_to_array(
        trade, ("sector", "destination", "source"),
        {"sector": sectors, "destination": regions, "source": regions},
    )
    xbilat = xbilat_3d.reshape(J * N, N)

    gamma = _long_to_array(
        gamma_df, ("sector", "region"),
        {"sector": sectors, "region": regions},
    )

    B_vec = _long_to_array(B_df, ("region",), {"region": regions})
    B = np.tile(B_vec, (J, 1))

    io_blocks = _long_to_array(
        io_df, ("country", "source_sector", "dest_sector"),
        {"country": countries, "source_sector": sectors, "dest_sector": sectors},
    )
    # Map each region to its IO block: US states share index 0; foreign
    # country k (region R + k) maps to block k + 1.
    block_idx = np.concatenate([np.zeros(R, dtype=int), np.arange(1, N - R + 1)])
    IO_3d = io_blocks[block_idx]                                        # (N, J, J)
    G_3d = (1 - gamma.T)[:, None, :] * IO_3d                            # (N, J, J)
    G = G_3d.reshape(N * J, J)

    GO_check = xbilat_3d.sum(axis=1)                                    # (J, N)

    row_sums = xbilat.sum(axis=1, keepdims=True)
    Din = np.where(row_sums > 0, xbilat / np.maximum(row_sums, 1e-30), 0.0)

    exports = xbilat_3d.sum(axis=1)                                     # (J, N) by source
    imports = xbilat_3d.sum(axis=2)                                     # (J, N) by destination
    Bn = (exports - imports).sum(axis=0)
    X0 = imports                                                         # alias for clarity

    # Exjn0[j, n] = sum_m Din[j*N + m, n] · X0[j, m]
    # (the data.m loop with PQ_vec[j*N + n_dest] = X0[j, n_dest])
    Din_3d = Din.reshape(J, N, N)
    Exjn0 = np.einsum("jmn,jm->jn", Din_3d, X0)

    VALjn0 = gamma * (1 - B) * Exjn0
    VARjn0 = (B / (1 - B)) * VALjn0
    VAL = VALjn0.sum(axis=0)
    VAR = VARjn0.sum(axis=0)
    VA = VAR + VAL

    # Final-demand share α. CDP's data.m makes α region-invariant: the
    # sector-wise residual (X0 - intermediate uses) summed across regions
    # is divided by the total of (VA - Bn), then tiled across regions.
    # G_3d[n, source, dest] · exports[dest, n] summed over dest → intermediate
    # use of source-sector input in region n.
    aux2 = X0 - np.einsum("nsd,dn->sn", G_3d, exports)
    sector_total = aux2.sum(axis=1)
    alphas = np.tile((sector_total / (VA - Bn).sum())[:, None], (1, N))

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
