"""Geographic-barriers counterfactual (CPRHS Section 5 / trade_costs.m).

Computes the TFP, GDP, and welfare effects of reducing trade costs across
US states. Two scenarios are shipped:

* ``"distance"`` — eliminates the geographic-distance component of trade costs.
* ``"other_barriers"`` — eliminates the non-distance ("other") trade barriers.

Each scenario is encoded as a `kappa_hat` matrix loaded from
``data/inputs/cprhs/geographic_barriers/``. Tradable sectors carry the
calibrated reductions; non-tradable sectors are left at 1 (no change).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from qge.io import DEFAULT_CALIBRATION, RawInputs, _load_long_array, load_inputs
from qge.models.benchmark import BenchmarkResult, BenchmarkShockResult, _run_shock

_GEOGRAPHIC_DIR = DEFAULT_CALIBRATION / "geographic_barriers"
_SCENARIO_FILES = {
    "distance":        "kappa_distance.parquet",
    "other_barriers":  "kappa_other_barriers.parquet",
}


def _load_kappa(scenario: str, raw: RawInputs) -> np.ndarray:
    """Load a (J*N, N) kappa_hat for the requested scenario."""
    if scenario not in _SCENARIO_FILES:
        raise ValueError(
            f"unknown scenario {scenario!r}; expected one of {list(_SCENARIO_FILES)}"
        )
    kappa_3d = _load_long_array(
        _GEOGRAPHIC_DIR / _SCENARIO_FILES[scenario],
        ("sector", "destination", "source"),
        (raw.sectors, raw.regions, raw.regions),
    )
    return kappa_3d.reshape(raw.J * raw.N, raw.N)


def compute_geographic_barriers(
    *,
    scenario: Literal["distance", "other_barriers"] = "distance",
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    tol: float = 1e-8,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkShockResult:
    """Trade-cost reduction counterfactual (trade_costs.m).

    Mirrors trade_costs.m: `kappa_hat` is loaded from the calibration's
    `geographic_barriers/` subfolder; lambda_hat stays at 1.
    """
    if raw is None:
        raw = load_inputs()
    kappa_hat = _load_kappa(scenario, raw)
    lambda_hat = np.ones((raw.J, raw.N))
    return _run_shock(
        lambda_hat, baseline=baseline, raw=raw, kappa_hat=kappa_hat,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )
