"""Convert FiscalData.RData → parquet under tw/data/inputs/canada_2010/.

One-time conversion of the R-workspace shipped with the Tombe & Winter (2021)
replication package. Re-shapes the R data frames into long-form parquets so
downstream Python scripts don't need pyreadr at runtime.

Original source: StatCan table 36-10-0450 (federal revenue + spending by
province), as cited in Tombe & Winter 2021 ReadMe.pdf. Phase 3 of the port
will fetch this directly via the StatCan WDS API to drop the .RData
dependency.

Run from tw/:

    uv run python scripts/convert_fiscaldata.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadr

RDATA_FILE = Path("/tmp/tombe_winter_2021/DATA/FiscalData.RData")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "inputs" / "canada_2010"


def main() -> None:
    if not RDATA_FILE.exists():
        raise FileNotFoundError(
            f"missing {RDATA_FILE}; extract via:\n"
            f"  mkdir -p /tmp/tombe_winter_2021 && unzip -o "
            f"~/Downloads/doi-10.5683-sp3-shdohy.zip -d /tmp/tombe_winter_2021/"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = pyreadr.read_r(str(RDATA_FILE))
    print(f"Loaded {len(result)} R objects from {RDATA_FILE.name}:")
    for name, df in result.items():
        shape = getattr(df, "shape", None)
        print(f"  {name:25s}  shape={shape}")

    # Write each object as a parquet so downstream code can read it via pandas
    # without pyreadr. The object names are preserved.
    for name, df in result.items():
        if df is None:
            continue
        out_path = OUTPUT_DIR / f"fiscal_{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  → wrote {out_path.relative_to(OUTPUT_DIR.parent.parent.parent)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
