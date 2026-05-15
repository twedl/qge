"""Convert CDP replication-kit .txt files to canonical long-form parquet.

Source layout (MATLAB convention from Caliendo-Dvorkin-Parro 2019):

  xbilat.txt       (1914, 87)  J*N rows × N cols. Stacked by sector:
                                rows 0..86 are destination for sector 1,
                                87..173 for sector 2, etc. Columns are sources.
  gamma.txt        (87, 22)    Region × sector. Value-added share of gross output.
  IO_tables.txt    (836, 22)   (38 region blocks × 22 source sectors, 22 dest sectors).
                                First 22 rows = US IO, next 37 × 22 = foreign IO.
  GO.txt           (87, 22)    Region × sector. Gross output.
  B.txt            (50, 1)     US-state structures share in value added.
  B_row.txt        (37, 1)     Foreign-country structures share in value added.

Sector ordering (1..22): the 22 productive sectors. Sector 0 of the dynamic
model — non-employment — is not part of Base_Year.

Region ordering (1..87): 50 US states (alphabetical), then 37 countries in the
order spelled out in the CDP Readme.

Usage::

    uv run python scripts/convert_cdp_txt.py \\
        --src "CDP replication files/Base_Year/" \\
        --out data/inputs/cdp_2000/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- labels

SECTORS: tuple[str, ...] = (
    "Food, Beverage, Tobacco",
    "Textile, Apparel, Leather",
    "Wood, Paper, Printing",
    "Petroleum and Coal",
    "Chemical",
    "Plastics and Rubber",
    "Nonmetallic Mineral Products",
    "Primary Metal and Fabricated Metal",
    "Machinery",
    "Computer, Electronic, Electrical",
    "Transportation Equipment",
    "Furniture and Miscellaneous Manufacturing",
    "Wholesale and Retail Trade",
    # End of 13 tradables — sectors 1-13
    "Construction",
    "Transport Services",
    "Information Services",
    "Finance and Insurance",
    "Real Estate",
    "Education",
    "Health Care",
    "Accommodation and Food Services",
    "Other Services",
    # End of 9 non-tradables — sectors 14-22
)
N_TRADABLES = 13

US_STATES: tuple[str, ...] = (
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia and DC",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
)

COUNTRIES: tuple[str, ...] = (
    "Australia", "Austria", "Belgium", "Bulgaria", "Brazil", "Canada",
    "China", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland",
    "France", "Germany", "Greece", "Hungary", "India", "Indonesia",
    "Italy", "Ireland", "Japan", "Lithuania", "Mexico", "Netherlands",
    "Poland", "Portugal", "Romania", "Russia", "Spain", "Slovak Republic",
    "Slovenia", "South Korea", "Sweden", "Taiwan", "Turkey",
    "United Kingdom", "Rest of World",
)

REGIONS: tuple[str, ...] = US_STATES + COUNTRIES   # 87 total
IO_BLOCKS: tuple[str, ...] = ("United States",) + COUNTRIES  # 38 blocks

# Sectoral dispersion 1/θ. Order matches SECTORS, hard-coded in Base_year.m.
# First 12 sectors have measured θ; sectors 13-22 share θ = 4.55 (the
# non-tradable default).
INV_THETA: tuple[float, ...] = (
    1/2.55, 1/5.56, 1/9.27, 1/51.08, 1/4.75, 1/1.66, 1/2.76, 1/6.78,
    1/1.52, 1/11.70, 1/1.01, 1/5.00, 1/4.55,
    *([1/4.55] * 9),
)

J, R, C = len(SECTORS), len(US_STATES), len(COUNTRIES)
N = R + C  # 87


# ---------------------------------------------------------------- loaders


def _read_matrix(path: Path, expected_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.loadtxt(path)
    if arr.shape != expected_shape:
        raise ValueError(f"{path.name}: expected {expected_shape}, got {arr.shape}")
    return arr


def load_raw_txt(src: Path) -> dict[str, np.ndarray]:
    """Read the 6 raw text files and return them as numpy arrays."""
    return {
        "xbilat":   _read_matrix(src / "xbilat.txt",    (J * N, N)),
        "gamma":    _read_matrix(src / "gamma.txt",     (N, J)),
        "IO_data":  _read_matrix(src / "IO_tables.txt", ((C + 1) * J, J)),
        "GO":       _read_matrix(src / "GO.txt",        (N, J)),
        "B_usa":    _read_matrix(src / "B.txt",         (R,)),
        "B_row":    _read_matrix(src / "B_row.txt",     (C,)),
    }


# ---------------------------------------------------------------- builders


def _array_to_long(
    arr: np.ndarray, dim_labels: tuple[tuple[str, tuple[str, ...]], ...]
) -> pd.DataFrame:
    """N-D numpy array → long form (one row per cell). Vectorized via
    ``np.indices`` — for a (22, 87, 87) bilateral trade tensor this is
    ~100× faster than the obvious nditer + per-row-dict loop."""
    expected = tuple(len(lbls) for _, lbls in dim_labels)
    if arr.shape != expected:
        raise ValueError(f"shape mismatch: {arr.shape} vs {expected}")
    idx = np.indices(arr.shape).reshape(arr.ndim, -1)
    cols = {
        name: np.asarray(labels)[idx[i]]
        for i, (name, labels) in enumerate(dim_labels)
    }
    cols["value"] = arr.ravel()
    return pd.DataFrame(cols)


def build_bilateral_trade(xbilat: np.ndarray, GO: np.ndarray) -> pd.DataFrame:
    """Long-form xbilat with non-tradable US-state diagonals filled in.

    Mirrors the data.m treatment: the raw xbilat is incomplete for non-
    tradable sectors in US states (CDP records cross-state flows for non-
    tradables as zero by assumption and supplements the diagonal from GO).
    Here we precompute the completed matrix so the parquet stores a
    self-contained xbilat.
    """
    # xbilat[j*N + n_dest, n_source] = flow from n_source to n_dest in
    # sector j (README: "Columns are the source countries and rows are the
    # destination countries"). E[j, n] from data.m = total sales by source n
    # in sector j = sum across destinations of xbilat[j*N + dest, n].
    xbilat = xbilat.copy()
    xbilat_3d = xbilat.reshape(J, N, N)         # (sector, dest, source)
    E = xbilat_3d.sum(axis=1)                    # (sector, source)
    # The raw xbilat zeros the own-state diagonal for non-tradable US-state
    # sectors; fill from GO so the saved parquet is self-contained.
    DS = GO.T[N_TRADABLES:, :R] - E[N_TRADABLES:, :R]      # (JNT, R)
    states = np.arange(R)
    for j_nt, j in enumerate(range(N_TRADABLES, J)):
        xbilat_3d[j, states, states] = DS[j_nt, states]
    return _array_to_long(
        xbilat_3d,
        (("sector", SECTORS), ("destination", REGIONS), ("source", REGIONS)),
    )


def build_value_added_share(gamma: np.ndarray) -> pd.DataFrame:
    """γ_{j,n} = value-added share of gross output in sector j, region n."""
    # gamma.txt is (N, J); transpose to (J, N) for our convention.
    return _array_to_long(
        gamma.T,
        (("sector", SECTORS), ("region", REGIONS)),
    )


def build_gross_output(GO: np.ndarray) -> pd.DataFrame:
    return _array_to_long(
        GO.T,
        (("sector", SECTORS), ("region", REGIONS)),
    )


def build_structures_share(B_usa: np.ndarray, B_row: np.ndarray) -> pd.DataFrame:
    values = np.concatenate([B_usa, B_row])
    return pd.DataFrame({"region": list(REGIONS), "value": values})


def build_io_coefficients(IO_data: np.ndarray) -> pd.DataFrame:
    """IO coefficients per region block.

    IO_data is (38 blocks × 22 source sectors, 22 dest sectors). Block 0
    = United States (shared across all 50 states), blocks 1..37 = the 37
    foreign countries in COUNTRIES order. We store one block per (country,
    source_sector, dest_sector) triple — the loader broadcasts US to all 50
    states when building arrays.
    """
    n_blocks = C + 1
    if IO_data.shape != (n_blocks * J, J):
        raise ValueError(f"IO_data shape {IO_data.shape} ≠ ({n_blocks * J}, {J})")
    blocks = IO_data.reshape(n_blocks, J, J)
    return _array_to_long(
        blocks,
        (("country", IO_BLOCKS),
         ("source_sector", SECTORS),
         ("dest_sector", SECTORS)),
    )


def build_sectoral_dispersion() -> pd.DataFrame:
    return pd.DataFrame({"sector": list(SECTORS), "value": list(INV_THETA)})


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src", type=Path,
        default=Path("CDP replication files/Base_Year/"),
        help="Directory containing the .txt source files",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("data/inputs/cdp_2000/"),
    )
    args = parser.parse_args()

    print(f"Reading raw .txt files from {args.src}...")
    raw = load_raw_txt(args.src)
    print(f"  xbilat:    {raw['xbilat'].shape}")
    print(f"  gamma:     {raw['gamma'].shape}")
    print(f"  IO_data:   {raw['IO_data'].shape}")
    print(f"  GO:        {raw['GO'].shape}")
    print(f"  B_usa:     {raw['B_usa'].shape}")
    print(f"  B_row:     {raw['B_row'].shape}")

    args.out.mkdir(parents=True, exist_ok=True)

    print("\nBuilding parquet files...")
    print("  bilateral_trade.parquet...")
    trade = build_bilateral_trade(raw["xbilat"], raw["GO"])
    trade.to_parquet(args.out / "bilateral_trade.parquet", index=False)
    print(f"    {len(trade):>6d} rows")

    print("  value_added_share.parquet...")
    gamma = build_value_added_share(raw["gamma"])
    gamma.to_parquet(args.out / "value_added_share.parquet", index=False)
    print(f"    {len(gamma):>6d} rows")

    print("  gross_output.parquet...")
    go = build_gross_output(raw["GO"])
    go.to_parquet(args.out / "gross_output.parquet", index=False)
    print(f"    {len(go):>6d} rows")

    print("  structures_share.parquet...")
    B = build_structures_share(raw["B_usa"], raw["B_row"])
    B.to_parquet(args.out / "structures_share.parquet", index=False)
    print(f"    {len(B):>6d} rows")

    print("  io_coefficients.parquet...")
    io = build_io_coefficients(raw["IO_data"])
    io.to_parquet(args.out / "io_coefficients.parquet", index=False)
    print(f"    {len(io):>6d} rows")

    print("  sectoral_dispersion.parquet...")
    T = build_sectoral_dispersion()
    T.to_parquet(args.out / "sectoral_dispersion.parquet", index=False)
    print(f"    {len(T):>6d} rows")


if __name__ == "__main__":
    main()
