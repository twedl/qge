"""Full dynamic baseline economy — Phase 2d (Step 4 of CDP §3.1).

Direct port of Step_4_Baseline.m. Stitches the three earlier phases:

* Phase 2a's 29-quarter quarterly series (anchor 2000Q1 through 2007Q1)
* Phase 2b's 29-period dynamic baseline 2000-2007
* Phase 2c's 200-period forward simulation from 2007Q4

into a single ``BaselineEconomy`` covering 200 quarters of trade/wage/
labor series plus 220 quarters of migration flows. No new computation —
just array concatenation along the time axis.

Layout (matching the MATLAB workspace):

* ``series_xbilat`` / ``series_pi`` / ``series_wages`` — first 29 slots
  from Phase 2b; slots 29..199 from Phase 2c slices 1..171.
* ``series_Ljnhat`` — first 29 slots from Phase 2a's series_Ljn0hat with
  the non-employment row 0 dropped; slots 29..199 from Phase 2c's US
  block (cols 0..R-1) at slices 1..171.
* ``series_mu`` — first 28 slots from Phase 2a's gross flows (28
  transitions); slots 28..219 from Phase 2c's first 192 slices. Total
  220 transitions.
* ``L0_initial`` — unchanged from Phase 2a.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qge.dynamic import (
    N_QUARTERS, N_TRANS, QuarterlySeries, build_quarterly_series,
)
from qge.io import RawInputs, load_inputs
from qge.models.base_year import EquilibriumResult, compute_baseline
from qge.models.dynamic_baseline import (
    DynamicBaseline2000_2007, compute_dynamic_baseline_2000_2007,
)
from qge.models.forward_simulation import (
    ForwardSimulation, compute_baseline_forward_2007,
)

TOTAL_QUARTERS = 200
TOTAL_MU_TRANSITIONS = 220


@dataclass(frozen=True)
class BaselineEconomy:
    """The full 2000-2050-ish dynamic baseline (200 quarters, 220 μ slots)."""

    series_xbilat: np.ndarray         # (J*N, N, TOTAL_QUARTERS)
    series_pi: np.ndarray              # (J*N, N, TOTAL_QUARTERS)
    series_wages: np.ndarray           # (J, N, TOTAL_QUARTERS)
    series_Ljnhat: np.ndarray          # (J, R, TOTAL_QUARTERS)
    series_mu: np.ndarray              # (RJ1, RJ1, TOTAL_MU_TRANSITIONS)
    L0_initial: np.ndarray             # (RJ1,)


def stitch_baseline_economy(
    quarterly: QuarterlySeries,
    dynamic_2007: DynamicBaseline2000_2007,
    forward: ForwardSimulation,
) -> BaselineEconomy:
    """Combine the three phases into one 2000-forward dataset."""
    # The Phase 2c arrays carry 200 slices; we splice in slices 1..171
    # (Python 0-indexed) starting at TOTAL_QUARTERS slot 29. The MATLAB
    # convention ignores forward slice 0 because it equals the 2007Q4
    # anchor that's already saved as the last slot of Phase 2b.
    after_anchor = TOTAL_QUARTERS - N_QUARTERS         # 200 - 29 = 171

    def _splice(phase_2b: np.ndarray, phase_2c: np.ndarray) -> np.ndarray:
        out = np.empty(phase_2b.shape[:-1] + (TOTAL_QUARTERS,))
        out[..., :N_QUARTERS] = phase_2b
        out[..., N_QUARTERS:] = phase_2c[..., 1:1 + after_anchor]
        return out

    series_xbilat = _splice(
        dynamic_2007.New_series_xbilat, forward.xbilat_out,
    )
    series_pi = _splice(
        dynamic_2007.New_Din_baseline, forward.pi_baseline,
    )
    series_wages = _splice(
        dynamic_2007.New_series_wageshat, forward.wages0,
    )

    # series_Ljnhat: drop non-employment row 0 from Phase 2a's hat series,
    # then splice Phase 2c's US block (the first R cols of Ljn_hat0).
    JNT1, R, _ = quarterly.series_Ljn0hat.shape
    J = JNT1 - 1
    series_Ljnhat = np.empty((J, R, TOTAL_QUARTERS))
    series_Ljnhat[..., :N_QUARTERS] = quarterly.series_Ljn0hat[1:, :, :]
    series_Ljnhat[..., N_QUARTERS:] = forward.Ljn_hat0[:, :R, 1:1 + after_anchor]

    # series_mu: 28 Phase 2a slices + 192 Phase 2c slices.
    forward_mu_slots = TOTAL_MU_TRANSITIONS - N_TRANS  # 220 - 28 = 192
    series_mu = np.empty(
        quarterly.series_mu.shape[:-1] + (TOTAL_MU_TRANSITIONS,)
    )
    series_mu[..., :N_TRANS] = quarterly.series_mu
    series_mu[..., N_TRANS:] = forward.mu[..., :forward_mu_slots]

    return BaselineEconomy(
        series_xbilat=series_xbilat,
        series_pi=series_pi,
        series_wages=series_wages,
        series_Ljnhat=series_Ljnhat,
        series_mu=series_mu,
        L0_initial=quarterly.L0_initial,
    )


def compute_baseline_economy(
    Yt_seed: np.ndarray,
    raw: RawInputs | None = None,
    rep_dir: Path | None = None,
    *,
    verbose: bool = False,
) -> BaselineEconomy:
    """End-to-end: Phase 1 baseline → 2a quarterly → 2b dynamic 2000-07
    → 2c forward 2007+ → 2d stitch."""
    if raw is None:
        raw = load_inputs()
    baseline = compute_baseline(raw=raw, tol=1e-7, vfactor=-0.05)
    assert rep_dir is not None, "rep_dir required"
    quarterly = build_quarterly_series(rep_dir, baseline, raw.gamma, raw.B)
    dynamic_2007 = compute_dynamic_baseline_2000_2007(
        raw=raw, baseline=baseline, quarterly=quarterly, verbose=verbose,
    )
    forward = compute_baseline_forward_2007(
        Yt_seed=Yt_seed,
        raw=raw, baseline=baseline,
        dynamic_2000_2007=dynamic_2007, quarterly=quarterly,
        verbose=verbose,
    )
    return stitch_baseline_economy(quarterly, dynamic_2007, forward)
