"""Loaders for the raw CPRHS-style inputs.

Two loader paths:

* ``load_inputs(directory)`` reads a calibration from long-form parquet files
  under a directory like ``data/inputs/cprhs/``. This is the canonical format
  and the intended interface for new calibrations (e.g. Canadian data).

* ``load_raw_inputs_from_mat(...)`` reads the legacy MATLAB ``.mat`` files
  that ship with the CPRHS replication kit. Used by
  ``scripts/convert_cprhs.py`` to materialize the parquet form, and as a
  verification path during the port.

The ``RawInputs`` dataclass carries the seven model arrays plus the sector
and region labels and the sectoral-dispersion vector ``T``, so consumers
downstream can derive dimensions and report results with meaningful names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parent.parent
MATLAB_ROOT = REPO_ROOT / "CPRHS replication files"
DATA_DIR = MATLAB_ROOT / "Data_and_Baseline_economies"
INPUTS_ROOT = REPO_ROOT / "data" / "inputs"
DEFAULT_CALIBRATION = INPUTS_ROOT / "cprhs"


@dataclass(frozen=True)
class RawInputs:
    """All seven raw arrays plus labels and the sectoral-dispersion vector.

    Shapes use ``J = len(sectors)`` and ``N = len(regions)``, both derived
    from the data. Position in the arrays follows the order of ``sectors``
    and ``regions``.

    Attributes
    ----------
    sectors : tuple of length J
    regions : tuple of length N
    T       : (J,)         sectoral dispersion (1 / θ_j, Eaton-Kortum)
    xbilat  : (J*N, N)     bilateral trade flows, stacked by sector. Row
                            ``j*N + n`` = (sector j, destination n); column k
                            = source k.
    L_j_n   : (J, N)       employment by sector × region
    IO      : (J, J)       input-output coefficients (rows = source sector,
                            columns = destination sector)
    gamma   : (J, N)       value-added share in gross output
    B       : (J, N)       structures share in value added
    alphas  : (J, N)       final-demand shares (columns sum to 1)
    io      : (N,)         fraction of structure rents to global portfolio
    """

    sectors: tuple[str, ...]
    regions: tuple[str, ...]
    T: np.ndarray
    xbilat: np.ndarray
    L_j_n: np.ndarray
    IO: np.ndarray
    gamma: np.ndarray
    B: np.ndarray
    alphas: np.ndarray
    io: np.ndarray

    @property
    def J(self) -> int:
        return len(self.sectors)

    @property
    def N(self) -> int:
        return len(self.regions)


# Canonical CPRHS labels (from Readme.pdf in the replication kit).
_CPRHS_SECTORS: tuple[str, ...] = (
    "Food, Beverage, Tobacco",
    "Textile, Apparel, Leather",
    "Wood and Paper",
    "Printing",
    "Petroleum and Coal",
    "Chemicals",
    "Plastics and Rubber",
    "Nonmetallic Mineral",
    "Primary and Fabricated Metal",
    "Machinery",
    "Computer and Electronics",
    "Electrical Equipment",
    "Transportation Equipment",
    "Furniture",
    "Miscellaneous",
    "Construction",
    "Wholesale and Retail Trade",
    "Transport Services",
    "Information Services",
    "Finance and Insurance",
    "Real Estate",
    "Education",
    "Health Care",
    "Arts and Recreation",
    "Accom. and Food Services",
    "Other Services",
)

_CPRHS_REGIONS: tuple[str, ...] = (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
    "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
    "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia and DC", "Washington", "West Virginia", "Wisconsin", "Wyoming",
)

# Sectoral trade elasticities (1 / θ_j) — CPRHS_Benchmark.m lines 19-21.
_CPRHS_T: np.ndarray = np.array(
    [
        1 / 2.55, 1 / 5.56, 1 / 9.46, 1 / 9.07, 1 / 51.08, 1 / 4.75, 1 / 1.66,
        1 / 2.76, 1 / 6.78, 1 / 1.52, 1 / 12.79, 1 / 10.60, 1 / 1.01, 1 / 5.00,
        1 / 5.00,
    ]
    + [1 / 4.55] * 11
)


def _long_to_array(
    df: pd.DataFrame,
    dim_cols: Sequence[str],
    val_col: str,
    dim_labels: Sequence[Sequence[str]],
) -> np.ndarray:
    """Reshape a long-form table into an N-D numpy array via positional mapping.

    Each `dim_cols[i]` is expected to take values from `dim_labels[i]`. Unknown
    labels and missing label combinations both raise ValueError.
    """
    shape = tuple(len(lbl) for lbl in dim_labels)
    code_arrays = []
    for col, labels in zip(dim_cols, dim_labels):
        idx_map = {v: i for i, v in enumerate(labels)}
        codes = df[col].map(idx_map)
        if codes.isna().any():
            unknown = sorted(df.loc[codes.isna(), col].unique().tolist())
            raise ValueError(f"unknown {col!r} labels: {unknown[:5]}")
        code_arrays.append(codes.to_numpy().astype(np.int64))

    out = np.full(shape, np.nan, dtype=float)
    out[tuple(code_arrays)] = df[val_col].to_numpy()
    if np.isnan(out).any():
        raise ValueError(f"missing entries when reshaping {dim_cols} into {shape}")
    return out


def _array_to_long(
    arr: np.ndarray,
    dim_cols: Sequence[str],
    dim_labels: Sequence[Sequence[str]],
    val_col: str = "value",
) -> pd.DataFrame:
    """Convert an N-D array to long-form, one row per cell. Inverse of `_long_to_array`."""
    expected_shape = tuple(len(lbl) for lbl in dim_labels)
    if arr.shape != expected_shape:
        raise ValueError(f"arr shape {arr.shape} doesn't match labels {expected_shape}")
    idx = pd.MultiIndex.from_product(dim_labels, names=list(dim_cols))
    return pd.DataFrame({val_col: arr.ravel()}, index=idx).reset_index()


def load_inputs(directory: Path | str = DEFAULT_CALIBRATION) -> RawInputs:
    """Load a calibration from long-form parquet files.

    Expected files in ``directory``::

        bilateral_trade.parquet      (sector, destination, source, value)
        employment.parquet           (sector, region, value)
        io_matrix.parquet            (source_sector, dest_sector, value)
        value_added_share.parquet    (sector, region, value)   # γ
        structures_share.parquet     (sector, region, value)   # B
        final_demand_share.parquet   (sector, region, value)   # α
        portfolio_share.parquet      (region, value)           # ι
        sectoral_dispersion.parquet  (sector, value)           # 1/θ

    Sector and region labels are derived from ``employment.parquet`` (in
    first-appearance order). All other files must use the same label set.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"input directory not found: {directory}. "
            "Run scripts/convert_cprhs.py to materialize the CPRHS calibration."
        )

    employment = pd.read_parquet(directory / "employment.parquet")
    sectors = tuple(pd.unique(employment["sector"]))
    regions = tuple(pd.unique(employment["region"]))
    labels_for = {
        "sector": sectors, "source_sector": sectors, "dest_sector": sectors,
        "region": regions, "destination": regions, "source": regions,
    }

    def _load(name: str, *dim_cols: str) -> np.ndarray:
        df = employment if name == "employment.parquet" else pd.read_parquet(directory / name)
        labels = tuple(labels_for[c] for c in dim_cols)
        return _long_to_array(df, dim_cols, "value", labels)

    L_j_n  = _load("employment.parquet",          "sector", "region")
    gamma  = _load("value_added_share.parquet",   "sector", "region")
    B      = _load("structures_share.parquet",    "sector", "region")
    alphas = _load("final_demand_share.parquet",  "sector", "region")
    IO     = _load("io_matrix.parquet",           "source_sector", "dest_sector")
    xbilat = _load("bilateral_trade.parquet",     "sector", "destination", "source") \
                .reshape(len(sectors) * len(regions), len(regions))
    io     = _load("portfolio_share.parquet",     "region").ravel()
    T      = _load("sectoral_dispersion.parquet", "sector").ravel()

    raw = RawInputs(
        sectors=sectors, regions=regions, T=T,
        xbilat=xbilat, L_j_n=L_j_n, IO=IO,
        gamma=gamma, B=B, alphas=alphas, io=io,
    )
    _validate(raw)
    return raw


def _validate(raw: RawInputs) -> None:
    """Range and finite-ness checks. Raises ValueError on failure."""
    for name in ("xbilat", "L_j_n", "IO", "gamma", "B", "alphas", "io", "T"):
        arr = getattr(raw, name)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name!r} contains non-finite values")
    if (raw.xbilat < 0).any():
        raise ValueError("xbilat has negative entries")
    if (raw.L_j_n < 0).any():
        raise ValueError("L_j_n has negative entries")
    if (raw.IO < 0).any():
        raise ValueError("IO has negative entries")
    if not ((raw.gamma >= 0) & (raw.gamma <= 1)).all():
        raise ValueError("gamma out of [0, 1]")
    if not ((raw.B >= 0) & (raw.B < 1)).all():
        raise ValueError("B out of [0, 1)")
    if not ((raw.io >= 0) & (raw.io <= 1)).all():
        raise ValueError("io out of [0, 1]")
    if (raw.T <= 0).any():
        raise ValueError("T (sectoral dispersion) must be strictly positive")


def _load_var(path: Path, var: str) -> np.ndarray:
    mat = loadmat(path)
    if var not in mat:
        available = sorted(k for k in mat if not k.startswith("__"))
        raise KeyError(f"{var!r} not in {path.name}; available: {available}")
    return np.asarray(mat[var])


def load_raw_inputs_from_mat(data_dir: Path = DATA_DIR) -> RawInputs:
    """Read the CPRHS replication-kit `.mat` files and tag with canonical labels.

    Used by scripts/convert_cprhs.py and as a verification path.
    """
    xbilat = _load_var(data_dir / "xbilat.mat", "xbilat")
    # The matrix in L_j_n.mat is stored under `L_j_n_with_construction`
    # (CPRHS_Benchmark.m line 40 renames it on load).
    L_j_n = _load_var(data_dir / "L_j_n.mat", "L_j_n_with_construction")
    gamma = _load_var(data_dir / "gamma.mat", "gamma")
    B = _load_var(data_dir / "B.mat", "B")
    alphas = _load_var(data_dir / "alphas.mat", "alphas")
    io = _load_var(data_dir / "io.mat", "io").reshape(-1)
    IO = np.loadtxt(data_dir / "IO_table.txt")

    raw = RawInputs(
        sectors=_CPRHS_SECTORS,
        regions=_CPRHS_REGIONS,
        T=_CPRHS_T,
        xbilat=xbilat, L_j_n=L_j_n, IO=IO,
        gamma=gamma, B=B, alphas=alphas, io=io,
    )
    _validate(raw)
    return raw


def load_raw_inputs() -> RawInputs:
    """Default loader — reads the CPRHS calibration from parquet."""
    return load_inputs(DEFAULT_CALIBRATION)


def load_base_year(name: str) -> dict[str, np.ndarray]:
    """Load a precomputed baseline equilibrium as a dict of arrays.

    Parameters
    ----------
    name : one of {"Benchmark", "NR", "NS", "NRNS"}
    """
    path = DATA_DIR / f"Base_Year_{name}.mat"
    mat = loadmat(path)
    return {k: v for k, v in mat.items() if not k.startswith("__")}
