"""Solve the initial equilibrium and set fiscal-transfer parameters.

Direct port of ``initialize_model.m``. Refines the loaded income vector via
the Leontief fixed point (which converges in ~1 iteration since the .mat
data is already consistent), then derives the transfer flows ``Tn`` and the
decomposition of trade imbalances into income-sensitive and exogenous parts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import RawInputs


@dataclass
class BaselineEquilibrium:
    """Initial equilibrium quantities plus fixed parameters.

    Mutable so counterfactual scripts can swap in alternative baselines
    (e.g., ``eliminate_transfers`` rebases with f=0 + SnBar).
    """

    # primitives (carried through from RawInputs for convenience)
    sectors: tuple[str, ...]
    regions: tuple[str, ...]
    N: int
    J: int
    PI: np.ndarray           # (N*J, N)
    IOmatrix: np.ndarray     # (J, J)
    gamma: np.ndarray        # (J, N*J)
    phi: np.ndarray          # (1, N*J)
    alpha: np.ndarray        # (J, N)
    theta: np.ndarray        # (J, 1)
    lambda_: np.ndarray      # (N-1, 1)
    mu: np.ndarray           # (N-1, 1)

    # derived initial equilibrium
    I: np.ndarray            # (1, N) total absorption
    R: np.ndarray            # (1, N*J) gross output per (region, sector)
    wL: np.ndarray           # (1, N) labor income per region
    A: np.ndarray            # (N*J, N*J) intermediate-use coefficient matrix
    AX: np.ndarray           # (N*J, N*J) intermediate-use flows
    Fmat: np.ndarray         # (N*J, N) final-demand flows
    Sn: np.ndarray           # (1, N) trade surplus per region (sum=0)
    Sn0: np.ndarray          # (1, N) saved baseline trade surplus

    # fiscal-transfer parameters & decomposition
    kappa: float             # migration Frechet shape parameter (default 1.5)
    f: float                 # income-sensitive transfer elasticity (zeta in paper)
    Tn: np.ndarray           # (N-1, 1) per-capita income-equalizing transfers
    Sn_exogenous: np.ndarray  # (1, N) S_n minus the income-sensitive component
    SnBar: np.ndarray        # (1, N) S_n + Tn (exogenous imbalances incl. transfers)
    wLshare: np.ndarray      # (1, N-1) labor-income share by province
    Sn_EQ: np.ndarray        # (1, N) balanced-budget equalization-only counterfactual

    # weights for aggregating welfare / GDP
    omega: np.ndarray        # (N-1, 1) GDP-share weights
    omega2: np.ndarray       # (N-1, 1) income-share weights

    # upstreamness measure used by Figure 3
    upstream_fally: np.ndarray  # (J, 1)


def _solve_initial_equilibrium(raw: RawInputs, Sn: np.ndarray) -> dict:
    """Run the Leontief fixed-point from initialize_model.m given fixed Sn.

    Returns a dict with R, A, AX, I, wL, Fmat consistent with ``Sn``.
    """
    N, J = raw.N, raw.J
    PI, alpha, phi, gamma = raw.PI, raw.alpha, raw.phi, raw.gamma

    sum_to_region = np.kron(np.eye(N), np.ones((J, 1)))  # (N*J, N)
    eye_NJ = np.eye(N * J)
    ones_NJ_col = np.ones((N * J, 1))
    ones_J_row = np.ones((1, J))

    # Initial guess: use the loaded I from the data
    I_new = (raw.I_baseline / raw.I_baseline.sum()).reshape(1, N)
    I_prev = np.zeros((1, N))

    while np.sum((I_prev - I_new) ** 2) > 1e-20:
        I_prev = I_new
        I = I_prev

        # Final demand: Fmat[k = i*J+j, n] = I[n] · α^j_n · π_{n,i}^j
        Fmat = (ones_NJ_col @ I) * np.tile(alpha, (N, 1)) * PI

        # Intermediate-use coefficients:
        # A[k = i*J+j, q = n*J+j'] = (1 − φ^j'_n) · γ^{j,j'}_n · π_{n,i}^j
        one_minus_phi_mat = ones_NJ_col @ (1 - phi)        # (N*J, N*J)
        gamma_tiled = np.tile(gamma, (N, 1))                # (N*J, N*J)
        PI_repeated = np.kron(PI, ones_J_row)               # (N*J, N*J)
        A = one_minus_phi_mat * gamma_tiled * PI_repeated

        # Leontief inverse
        R_col = np.linalg.solve(eye_NJ - A, Fmat.sum(axis=1))
        R = R_col.reshape(1, N * J)

        # Labor income per region and updated income
        wL = (phi * R) @ sum_to_region          # (1, N)
        I_new = wL - Sn                          # (1, N)
        I_new = np.abs(I_new)
        I_new = I_new / I_new.sum()

    AX = A * (ones_NJ_col @ R)                    # (N*J, N*J)
    return {"R": R, "A": A, "AX": AX, "I": I_new,
            "wL": wL, "Fmat": Fmat}


def initialize(raw: RawInputs, kappa: float = 1.5, f: float = 0.3268
                ) -> BaselineEquilibrium:
    """Build the baseline equilibrium from the raw inputs (initialize_model.m).

    Parameters
    ----------
    raw : RawInputs
    kappa : migration Frechet shape parameter (default 1.5)
    f : income-sensitive transfer elasticity ζ (default 0.3268)
    """
    N, J = raw.N, raw.J
    PI = raw.PI; alpha = raw.alpha; phi = raw.phi; gamma = raw.gamma

    # ── Step 1: Leontief fixed point given the loaded Sn ─────────────────
    Sn = raw.Sn_baseline.copy()                  # (1, N)
    eq = _solve_initial_equilibrium(raw, Sn)
    I = eq["I"]; R = eq["R"]; A = eq["A"]; AX = eq["AX"]
    wL = eq["wL"]; Fmat = eq["Fmat"]

    # Recompute Sn from the converged R (matches initialize_model.m lines
    # 31-35: Snj = R_nj - Xnj; Sn = sum(Snj))
    Rnj = R.reshape(N, J).T                       # (J, N)
    Dnj = np.tile(I, (J, 1)) * alpha              # (J, N) final demand by (j, n)
    Xnj = Dnj + raw.IOmatrix @ Rnj                # (J, N) total absorption by (j, n)
    Snj = Rnj - Xnj
    Sn = Snj.sum(axis=0).reshape(1, N)
    Sn0 = Sn.copy()

    # ── Step 2: Fiscal-transfer parameters ───────────────────────────────
    # Tn = transfers_share_GDP .* I(1:N-1)'  (provinces only)
    # Then rebalanced: Tn = (Tn ./ λ - sum(Tn)) .* λ   (per-capita centering)
    I_provs = I[0, : N - 1].reshape(N - 1, 1)
    lambda_ = raw.lambda_
    Tn = raw.transfers_share_GDP * I_provs
    Tn = (Tn / lambda_ - Tn.sum()) * lambda_      # (N-1, 1) ⇒ per-capita transfers

    # Labor income share by province (denominator for weighted income-sensitive
    # transfer distribution)
    wL_provs = wL[0, : N - 1].reshape(1, N - 1)
    wLshare = wL_provs / wL_provs.sum()           # (1, N-1)
    w = wL_provs / lambda_.T                      # (1, N-1) per-worker income

    # Income-sensitive transfers: t_n ∝ w_n^(-f), normalized so labor income
    # weighted across provinces equals 1 (i.e., a is the "transfer factor"
    # such that wL_n · a_n sums to total labor income).
    a = w ** (-f) / (wLshare @ (w ** (-f)).T)     # (1, N-1)

    # Exogenous baseline imbalances (data) and decomposition
    SnBar = Sn + np.concatenate([Tn.T, np.array([[0.0]])], axis=1)   # (1, N)
    Sn_fiscal_provs = (1 - a) * w * lambda_.T                       # (1, N-1)
    Sn_fiscal = np.concatenate([Sn_fiscal_provs,
                                 np.array([[0.0]])], axis=1)        # (1, N)
    Sn_exogenous = Sn - Sn_fiscal                                    # (1, N)

    # ── Step 3: aggregation weights ──────────────────────────────────────
    mu = raw.mu
    omega_unnorm = lambda_ * mu                                      # (N-1, 1)
    omega = omega_unnorm / omega_unnorm.sum()                        # (N-1, 1)
    I_provs_flat = I_provs.ravel()
    wL_provs_flat = wL_provs.ravel()
    income_per_labor = I_provs_flat / wL_provs_flat                  # (N-1,)
    omega2_unnorm = (lambda_.ravel() * mu.ravel() * income_per_labor)
    omega2 = (omega2_unnorm / omega2_unnorm.sum()).reshape(N - 1, 1)

    # ── Step 4: closed-economy upstreamness (used by Figure 3) ───────────
    alpha_us = alpha[:, 0:1]                                          # (J, 1)
    eye_J = np.eye(J)
    Y = np.linalg.solve(eye_J - raw.IOmatrix, alpha_us)               # (J, 1)
    DELTA_closed = raw.IOmatrix * (Y.T / Y)                          # (J, J)
    upstream_fally = np.linalg.solve(eye_J - DELTA_closed, np.ones((J, 1)))

    # ── Step 5: equalization-only counterfactual transfers (Table 4 col 3) ─
    EQshare = np.array([0, 0, 0.0343, 0.0523, 0, 0.0301, 0.0015,
                         0.0631, 0.0261, 0]).reshape(N - 1, 1)
    I10 = I[0, : N - 1].reshape(N - 1, 1)
    Sn_EQ_provs = EQshare * I10 - (EQshare * I10).sum() * (I10 / I10.sum())
    Sn_EQ = np.concatenate([Sn_EQ_provs.T,
                              np.array([[0.0]])], axis=1)             # (1, N)

    return BaselineEquilibrium(
        sectors=raw.sectors, regions=raw.regions, N=N, J=J,
        PI=PI, IOmatrix=raw.IOmatrix, gamma=gamma, phi=phi,
        alpha=alpha, theta=raw.theta, lambda_=lambda_, mu=mu,
        I=I, R=R, wL=wL, A=A, AX=AX, Fmat=Fmat,
        Sn=Sn, Sn0=Sn0,
        kappa=kappa, f=f, Tn=Tn,
        Sn_exogenous=Sn_exogenous, SnBar=SnBar, wLshare=wLshare,
        Sn_EQ=Sn_EQ,
        omega=omega, omega2=omega2,
        upstream_fally=upstream_fally,
    )


def eliminate_transfers(baseline: BaselineEquilibrium) -> BaselineEquilibrium:
    """Re-solve the initial equilibrium assuming f=0 and Sn = SnBar.

    Matches eliminate_transfers.m. Used as an alternative baseline for
    several counterfactuals (Figure 2 panel (a), Table 7, Table 8 alt).
    """
    raw_like = type("RawLike", (), {})()
    # Use SnBar (exogenous imbalances + transfers) as the new "exogenous" Sn.
    Sn_new = baseline.SnBar.copy()

    # Run the Leontief fixed point with Sn fixed at SnBar.
    N, J = baseline.N, baseline.J
    PI, alpha, phi, gamma = baseline.PI, baseline.alpha, baseline.phi, baseline.gamma

    sum_to_region = np.kron(np.eye(N), np.ones((J, 1)))
    eye_NJ = np.eye(N * J)
    ones_NJ_col = np.ones((N * J, 1))
    ones_J_row = np.ones((1, J))

    I_new = (baseline.I / baseline.I.sum()).reshape(1, N)
    I_prev = np.zeros((1, N))
    while np.sum((I_prev - I_new) ** 2) > 1e-20:
        I_prev = I_new
        I = I_prev
        Fmat = (ones_NJ_col @ I) * np.tile(alpha, (N, 1)) * PI
        A = (ones_NJ_col @ (1 - phi)) * np.tile(gamma, (N, 1)) * np.kron(PI, ones_J_row)
        R_col = np.linalg.solve(eye_NJ - A, Fmat.sum(axis=1))
        R = R_col.reshape(1, N * J)
        wL = (phi * R) @ sum_to_region
        I_new = wL - Sn_new
        I_new = I_new / I_new.sum()
    AX = A * (ones_NJ_col @ R)

    return BaselineEquilibrium(
        sectors=baseline.sectors, regions=baseline.regions, N=N, J=J,
        PI=PI, IOmatrix=baseline.IOmatrix, gamma=gamma, phi=phi,
        alpha=alpha, theta=baseline.theta,
        lambda_=baseline.lambda_, mu=baseline.mu,
        I=I_new, R=R, wL=wL, A=A, AX=AX, Fmat=Fmat,
        Sn=Sn_new, Sn0=Sn_new.copy(),
        kappa=baseline.kappa, f=0.0,
        Tn=baseline.Tn,            # not used after rebasing
        Sn_exogenous=Sn_new,        # f=0 ⇒ no income-sensitive component
        SnBar=Sn_new, wLshare=baseline.wLshare,
        Sn_EQ=baseline.Sn_EQ,
        omega=baseline.omega, omega2=baseline.omega2,
        upstream_fally=baseline.upstream_fally,
    )
