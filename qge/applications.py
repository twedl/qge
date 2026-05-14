"""Real-world counterfactuals from Section 6 of CPRHS (2017).

Each entry point wraps the Benchmark shock pipeline with a specific
lambda_hat (and, in the Katrina case, H_hat) mask:

    shock_california_computers   the 2002-2007 productivity boom in
                                  Computer & Electronics in California
    shock_north_dakota            the Bakken-era productivity boom in
                                  North Dakota
    shock_fire_nyc                contraction in Finance/Insurance and
                                  Real Estate in New York
    shock_katrina                 structural destruction in Louisiana
                                  (Hurricane Katrina)

Shock data files live in the calibration directory's ``applications/``
subfolder. To batch many calls, pass ``raw=load_inputs()`` so the parquet
load is shared.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from qge.io import DEFAULT_CALIBRATION, RawInputs, _load_long_array, load_inputs
from qge.models.benchmark import BenchmarkResult, BenchmarkShockResult, _run_shock

_APPLICATIONS_DIR = DEFAULT_CALIBRATION / "applications"


def _measured_tfp(raw: RawInputs) -> np.ndarray:
    """Measured-TFP-change matrix 2002-2007 (J, N) for the CPRHS calibration."""
    return _load_long_array(
        _APPLICATIONS_DIR / "measured_tfp_2002_2007.parquet",
        ("sector", "region"),
        (raw.sectors, raw.regions),
    )


def _north_dakota_lambda(raw: RawInputs) -> np.ndarray:
    """North Dakota productivity shock (J,) for the CPRHS calibration."""
    return _load_long_array(
        _APPLICATIONS_DIR / "north_dakota_lambda.parquet",
        ("sector",),
        (raw.sectors,),
    ).ravel()


def shock_california_computers(
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    tol: float = 1e-8,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkShockResult:
    """Annualized productivity boom in Computer & Electronics in California.

    Mirrors Shock_California_Computers.m: measured TFP changes 2002-2007 are
    converted to fundamental TFP via λ_app = λ_measured^(1/γ), then inverted
    and annualized (5-year shock → 1-year, exponent 0.2). The single shocked
    entry becomes λ_measured^(0.2/γ) at (Computer & Electronics, California).
    """
    if raw is None:
        raw = load_inputs()
    sector = raw.sectors.index("Computer and Electronics")
    region = raw.regions.index("California")

    measured = _measured_tfp(raw)
    lambda_hat = np.ones((raw.J, raw.N))
    lambda_hat[sector, region] = measured[sector, region] ** (
        0.2 / raw.gamma[sector, region]
    )
    return _run_shock(
        lambda_hat, baseline=baseline, raw=raw,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )


def shock_north_dakota(
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    tol: float = 1e-8,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkShockResult:
    """Productivity boom across all sectors in North Dakota.

    Mirrors Shock_NorthDakota.m: lambda_hat is set to the supplied per-sector
    vector for North Dakota and 1 elsewhere.
    """
    if raw is None:
        raw = load_inputs()
    region = raw.regions.index("North Dakota")
    lambda_hat = np.ones((raw.J, raw.N))
    lambda_hat[:, region] = _north_dakota_lambda(raw)
    return _run_shock(
        lambda_hat, baseline=baseline, raw=raw,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )


def shock_fire_nyc(
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    tol: float = 1e-8,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkShockResult:
    """Finance and Real Estate contraction in New York.

    Mirrors Shock_FIRE_NYC.m. The two shock values are hardcoded constants
    from the original script (no separate data file).
    """
    if raw is None:
        raw = load_inputs()
    ny = raw.regions.index("New York")
    finance = raw.sectors.index("Finance and Insurance")
    real_estate = raw.sectors.index("Real Estate")

    lambda_hat = np.ones((raw.J, raw.N))
    lambda_hat[finance, ny] = 0.927550878
    lambda_hat[real_estate, ny] = 0.965179628
    return _run_shock(
        lambda_hat, baseline=baseline, raw=raw,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )


def shock_katrina(
    *,
    baseline: Optional[BenchmarkResult] = None,
    raw: Optional[RawInputs] = None,
    tol: float = 1e-8,
    vfactor: float = -0.1,
    maxit: int = 1_000_000,
    verbose: bool = False,
) -> BenchmarkShockResult:
    """Hurricane Katrina structural damage in Louisiana.

    Mirrors Shock_Katrina.m. Unlike the productivity shocks, this is a
    structures shock H_hat (25.25% destruction in Louisiana) — lambda_hat
    stays at 1. The H_hat factor threads through Lchange / expenditure / GMC.
    """
    if raw is None:
        raw = load_inputs()
    la = raw.regions.index("Louisiana")

    lambda_hat = np.ones((raw.J, raw.N))
    H_hat = np.ones(raw.N)
    H_hat[la] = 0.7475
    return _run_shock(
        lambda_hat, baseline=baseline, raw=raw, H_hat=H_hat,
        tol=tol, vfactor=vfactor, maxit=maxit, verbose=verbose,
    )
