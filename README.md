# qge

Python implementations of quantitative general-equilibrium models from the Caliendo & Parro line of work. Each model lives in its own subdirectory with an independent `uv` environment — they share no code.

## Sub-projects

| dir | model | reference | status |
|---|---|---|---|
| [`cprhs/`](cprhs/) | Caliendo, Parro, Rossi-Hansberg, Sarte — *The Impact of Regional and Sectoral Productivity Changes in the U.S. Economy* | [author site](https://sites.google.com/site/lorenzocaliendo/research/CPRHS) | working; verified to MATLAB workspaces at machine epsilon; ships a CPRHS reference calibration and a Canadian calibration (`canada_2021`) plus an extended 17-region partner version |
| [`cdp/`](cdp/) | Caliendo, Dvorkin, Parro — dynamic labor-market trade model | [author site](https://sites.google.com/site/lorenzocaliendo/research/cdp) | Phase 1 + Phase 2 + Phase 3 done (static base year, full dynamic baseline 2000-forward, China-shock counterfactual); employment/welfare analysis next |

## Working in a sub-project

```sh
cd cprhs
uv sync
uv run pytest
```

Each sub-project has its own `pyproject.toml`, `qge/` package, `data/`, `scripts/`, `tests/`, and README. The Python package name `qge` is reused across sub-projects — they're never imported simultaneously, so this is fine.

## Repository-wide files

`CLAUDE.md` — coding guidelines that apply to both projects.
`LICENSE` — MIT, applies to all code.
`.gitignore` — Python / venv / OS clutter, plus MATLAB source folder under `cprhs/`.
