"""Convert the CPRHS MATLAB replication-kit inputs into long-form parquet.

Run once to materialize ``data/inputs/cprhs/`` from the gitignored
``CPRHS replication files/`` tree. The output is the canonical input format
consumed by ``qge.io.load_inputs``.

Usage::

    uv run python scripts/convert_cprhs.py
"""

from __future__ import annotations

from qge.io import INPUTS_ROOT, _array_to_long, load_raw_inputs_from_mat


_README_TEMPLATE = """# CPRHS 2017 reference calibration

This directory holds the seven raw inputs of the Caliendo, Parro, Rossi-Hansberg
and Sarte (2017) "Impact of Regional and Sectoral Productivity Changes in the
U.S. Economy" calibration, materialized from the paper's MATLAB replication
kit by `scripts/convert_cprhs.py`.

## Provenance

Source: `CPRHS replication files/Data_and_Baseline_economies/` (the .mat and
.txt files distributed with the paper). Labels come from `Readme.pdf` of the
replication kit. Sectoral dispersion (1/θ_j) is the vector hardcoded in
`CPRHS_Benchmark.m` lines 19-21.

## Schema

All files are long-form parquet with a `value` column. Categorical columns
hold human-readable sector and region names.

| file                          | columns                                  | rows  |
|-------------------------------|------------------------------------------|------:|
| `bilateral_trade.parquet`     | `sector, destination, source, value`     | 65000 |
| `employment.parquet`          | `sector, region, value`                  |  1300 |
| `io_matrix.parquet`           | `source_sector, dest_sector, value`      |   676 |
| `value_added_share.parquet`   | `sector, region, value` (γ)              |  1300 |
| `structures_share.parquet`    | `sector, region, value` (B)              |  1300 |
| `final_demand_share.parquet`  | `sector, region, value` (α)              |  1300 |
| `portfolio_share.parquet`     | `region, value` (ι)                      |    50 |
| `sectoral_dispersion.parquet` | `sector, value` (1/θ)                    |    26 |

Sectors and regions are taken from `employment.parquet` in first-appearance
order; all other files must use the same label set. Loaded via
`qge.io.load_inputs(directory)`.
"""


def main() -> None:
    raw = load_raw_inputs_from_mat()
    out_dir = INPUTS_ROOT / "cprhs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sectors, regions = raw.sectors, raw.regions
    sec_reg = ("sector", "region"), (sectors, regions)
    tables = {
        "employment.parquet":          _array_to_long(raw.L_j_n,  *sec_reg),
        "value_added_share.parquet":   _array_to_long(raw.gamma,  *sec_reg),
        "structures_share.parquet":    _array_to_long(raw.B,      *sec_reg),
        "final_demand_share.parquet":  _array_to_long(raw.alphas, *sec_reg),
        "io_matrix.parquet":           _array_to_long(
            raw.IO, ("source_sector", "dest_sector"), (sectors, sectors)),
        "bilateral_trade.parquet":     _array_to_long(
            raw.xbilat.reshape(raw.J, raw.N, raw.N),
            ("sector", "destination", "source"), (sectors, regions, regions)),
        "portfolio_share.parquet":     _array_to_long(raw.io, ("region",), (regions,)),
        "sectoral_dispersion.parquet": _array_to_long(raw.T,  ("sector",), (sectors,)),
    }
    for name, df in tables.items():
        path = out_dir / name
        df.to_parquet(path, index=False)
        print(f"  wrote {path}  ({len(df):>6d} rows, {path.stat().st_size/1024:6.1f} KiB)")

    readme = out_dir / "README.md"
    readme.write_text(_README_TEMPLATE)
    print(f"  wrote {readme}")


if __name__ == "__main__":
    main()
