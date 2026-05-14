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
    B       : (N,)         structures share in value added (assumed
                            constant across sectors within a region)
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


def _load_long_array(
    path: Path,
    dim_cols: Sequence[str],
    dim_labels: Sequence[Sequence[str]],
) -> np.ndarray:
    """Read a long-form parquet and reshape into an N-D numpy array.

    Convenience wrapper over `pd.read_parquet` + `_long_to_array`. Each
    `dim_cols[i]` is expected to take values from `dim_labels[i]`.
    """
    return _long_to_array(pd.read_parquet(path), dim_cols, "value", dim_labels)


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
        structures_share.parquet     (region, value)           # B (per-state)
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
    B      = _load("structures_share.parquet",    "region").ravel()
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
    """Shape, range, and consistency checks. Raises ValueError on failure.

    Catches likely real-world data issues that would otherwise produce silent
    NaN / wrong-answer behavior inside the solver.
    """
    J, N = raw.J, raw.N
    expected_shapes = {
        "xbilat": (J * N, N),
        "L_j_n":  (J, N),
        "IO":     (J, J),
        "gamma":  (J, N),
        "B":      (N,),
        "alphas": (J, N),
        "io":     (N,),
        "T":      (J,),
    }
    for name, want in expected_shapes.items():
        arr = getattr(raw, name)
        if arr.shape != want:
            raise ValueError(f"{name!r} has shape {arr.shape}, expected {want}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name!r} contains non-finite values")
    # Sign / range constraints.
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
    # Consistency constraints — would silently break the solver.
    alpha_col_sums = raw.alphas.sum(axis=0)
    if not np.allclose(alpha_col_sums, 1.0, atol=1e-6):
        worst = int(np.argmax(np.abs(alpha_col_sums - 1)))
        raise ValueError(
            f"alphas column for region index {worst} sums to "
            f"{alpha_col_sums[worst]:.4f}; expected 1.0 (final-demand shares)"
        )
    if (raw.xbilat.sum(axis=1) == 0).any():
        zero_rows = int(np.flatnonzero(raw.xbilat.sum(axis=1) == 0)[0])
        raise ValueError(
            f"xbilat row {zero_rows} sums to 0 (no expenditure for that "
            f"sector × destination); the trade-share denominator would divide by zero"
        )
    if (raw.IO.sum(axis=0) == 0).any():
        zero_cols = int(np.flatnonzero(raw.IO.sum(axis=0) == 0)[0])
        raise ValueError(
            f"IO column {zero_cols} sums to 0 (destination sector has no "
            f"intermediate inputs); IO column-normalization would divide by zero"
        )


def _collapse_to_region_vector(
    arr: np.ndarray, *, name: str, atol: float = 1e-6,
) -> np.ndarray:
    """Reduce a (J, N) array assumed constant across sectors to (N,).

    Raises ValueError if rows differ by more than `atol` — the model assumes
    the quantity is sector-invariant within a region, and silently averaging
    sector-varying data would obscure a calibration bug.
    """
    if arr.ndim == 1:
        return arr
    if arr.shape[0] == 1:
        return arr[0, :]
    max_diff = np.max(np.abs(arr - arr[0:1, :]), axis=0)
    if (max_diff > atol).any():
        worst = int(np.argmax(max_diff))
        raise ValueError(
            f"{name!r} varies across sectors in region index {worst} "
            f"(max diff {max_diff[worst]:.3e}); the model assumes "
            f"sector-invariant — aggregate externally before loading."
        )
    return arr[0, :]


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
    # B in the .mat file is (J, N) with rows identical (CPRHS data has structures
    # share constant across sectors within a state). The model uses (N,).
    B = _collapse_to_region_vector(_load_var(data_dir / "B.mat", "B"), name="B")
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
