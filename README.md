# qge

Python implementations of quantitative general-equilibrium models from the Caliendo & Parro line of work. Each model lives in its own subdirectory with an independent `uv` environment — they share no code.

## Sub-projects

| dir | model | reference | data archive | status |
|---|---|---|---|---|
| [`cprhs/`](cprhs/) | Caliendo, Parro, Rossi-Hansberg, Sarte — *The Impact of Regional and Sectoral Productivity Changes in the U.S. Economy* | [author site](https://sites.google.com/site/lorenzocaliendo/research/CPRHS) | — | working; verified to MATLAB workspaces at machine epsilon; ships a CPRHS reference calibration and a Canadian calibration (`canada_2021`) plus an extended 17-region partner version |
| [`cdp/`](cdp/) | Caliendo, Dvorkin, Parro — dynamic labor-market trade model | [author site](https://sites.google.com/site/lorenzocaliendo/research/cdp) | — | Phase 1 + Phase 2 + Phase 3 done (static base year, full dynamic baseline 2000-forward, China-shock counterfactual); employment/welfare analysis next |
| [`tw/`](tw/) | Tombe & Winter — *Fiscal Integration with Internal Trade: Quantifying the Effects of Federal Transfers in Canada* | *Canadian Journal of Economics* 54(2), 522–556 ([doi:10.1111/caje.12491](https://doi.org/10.1111/caje.12491)) | Tombe & Winter (2022), Borealis, V1 ([doi:10.5683/SP3/SHDOHY](https://doi.org/10.5683/SP3/SHDOHY)) | complete; every published table and figure replicated end-to-end from the Dataverse archive; ships **`canada_2022`** alongside the paper's `canada_2010` — full refresh from StatCan 12-10-0088 (trade), 14-10-0327 (LFS), 15-211-X Detail 2022 (industrial structure, NAICS 2022), and 36-10-0450 (fiscal), with tauhat_* re-estimated via the AT-2016 pipeline |
| [`dty/`](dty/) | Díez, Tombe & Yang — *Barriers Within Borders: Internal Trade and Domestic Market Integration* (trade-cost measurement only) | IMF Working Paper WP/26/194 (2026) | — | non-distance internal trade costs from StatCan 2021 detail flows; matches the 9.3 % headline and the uniform-θ robustness levels at ρ = −0.5 population-weighted distance; bilateral goods pattern partly matched (see [`dty/REPLICATION.md`](dty/REPLICATION.md)); ships 2021 population-weighted DA distances |

## Working in a sub-project

```sh
cd cprhs
uv sync
uv run pytest
```

Each sub-project has its own `pyproject.toml`, `qge/` package, `data/`, `scripts/`, `tests/`, and README. `dty/` has no `qge/` package, only scripts. The Python package name `qge` is reused across sub-projects — they're never imported simultaneously, so this is fine.

## Repository-wide files

`CLAUDE.md` — coding guidelines that apply to all sub-projects.
`LICENSE` — MIT, applies to all code.
`.gitignore` — Python / venv / OS clutter, plus MATLAB source folders.
