"""Per-iteration math for the CDP Base_Year solver.

Direct ports of P_h_om.m, Dinprime.m, expenditurenew.m, GMCnew.m. The
MATLAB code's per-region Python loops are vectorized via reshape and
einsum without altering numerics.

Array conventions (matching the MATLAB code):
* ``J`` sectors, ``N`` regions, ``R`` of which are US states (first ``R``).
* ``xbilat`` and ``Din`` are stacked ``(J*N, N)``: rows are
  (sector, destination), columns are sources. The trade-cost / share
  matrices have the same layout.
* ``om`` is the labor-input-bundle wage ``ω``, shape ``(J, N)``. For US
  states ``om = w · L^(-B)``; for foreign countries ``om = w = r``.
"""

from __future__ import annotations

import numpy as np

# CDP paper constants (Table 4, Caliendo-Dvorkin-Parro 2019).
BETA = 0.99       # quarterly discount factor
NU = 5.3436       # dispersion of taste shocks (inverse migration elasticity)


def scrub(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN and Inf with ``fill``. Used at multiple boundary
    points where 0/0 or x/0 arise from data inputs that contain zeros
    (non-tradable trade flows, dropout migration cells)."""
    return np.where(np.isnan(arr) | np.isinf(arr), fill, arr)


def _inv_theta_per_jn(T: np.ndarray, N: int) -> np.ndarray:
    """(J*N, 1) column of 1/θ_j repeated N times — for kappa_hat ** (-1/T)."""
    return np.repeat(T.ravel(), N).reshape(-1, 1)


def P_h_om(
    om: np.ndarray,
    kappa_hat: np.ndarray,
    lambda_hat: np.ndarray,
    T: np.ndarray,
    G: np.ndarray,
    gamma: np.ndarray,
    Din: np.ndarray,
    J: int,
    N: int,
    maxit: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-point on (phat, c) given factor-price guess om.

    Returns (phat, c) with shapes (J, N), (J, N).
    """
    G_3d = G.reshape(N, J, J)                            # [n, source, dest]
    Din_k = Din * (kappa_hat ** (-1.0 / _inv_theta_per_jn(T, N)))
    Din_k_3d = Din_k.reshape(J, N, N)                    # [j, dest, source]
    T_col = T.reshape(-1, 1)

    pf0 = np.ones((J, N))
    pfmax = 1.0
    it = 1
    while it <= maxit and pfmax > tol:
        # log c_{k,n} = γ_{k,n} log ω_{k,n} + Σ_s G_{n,s,k} log p_{s,n}
        lc = gamma * np.log(om) + np.einsum("nsk,sn->kn", G_3d, np.log(pf0))
        c = np.exp(lc)

        # phat_{j,d} = (Σ_m Din_k_{j,d,m} · λ_{j,m}^(γ/θ) · c_{j,m}^(-1/θ))^(-θ_j)
        adjusted = (lambda_hat ** (gamma / T_col)) * (c ** (-1.0 / T_col))
        phat = np.einsum("jdm,jm->jd", Din_k_3d, adjusted) ** (-T_col)

        pfmax = float(np.abs(phat - pf0).max())
        pf0 = phat
        it += 1
    return pf0, c


def Dinprime(
    Din: np.ndarray,
    kappa_hat: np.ndarray,
    lambda_hat: np.ndarray,
    c: np.ndarray,
    phat: np.ndarray,
    T: np.ndarray,
    J: int,
    N: int,
    gamma: np.ndarray,
) -> np.ndarray:
    """New bilateral trade shares given factor prices and goods prices."""
    T_col = T.reshape(-1, 1)
    cp = c ** (-1.0 / T_col)         # (J, N)
    phatp = phat ** (-1.0 / T_col)   # (J, N)

    Din_k = Din * (kappa_hat ** (-1.0 / _inv_theta_per_jn(T, N)))

    # DD[j*N + dest, source] = Din_k[..] · cp[j, source] · λ_{j,source}^(γ/θ)
    # — per-source adjustment, constant across destinations.
    DD = Din_k * np.kron(cp * (lambda_hat ** (gamma / T_col)), np.ones((N, 1)))
    # Normalize by the destination-side price index (constant across sources).
    return DD / phatp.ravel()[:, None]


def expenditurenew(
    J: int,
    N: int,
    alphas: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    Dinp: np.ndarray,
    om: np.ndarray,
    Ljn_hat: np.ndarray,
    Snp: np.ndarray,
    VARjn0: np.ndarray,
    VALjn0: np.ndarray,
    io: np.ndarray,
) -> np.ndarray:
    """Solve (I − Ω) X = α (income − deficits) for total expenditure."""
    VARjnp = VARjn0 * om * (Ljn_hat ** (1 - B))
    VARp = VARjnp.sum(axis=0)
    Chip = VARp.sum()
    Bnp = Snp.ravel() - io.ravel() * Chip + VARp

    # NBP[source, n*J + k] = Dinp_3d[k, n, source]: regroup Dinp axes so
    # the per-source-per-region trade share for sector k sits contiguously.
    Dinp_3d = Dinp.reshape(J, N, N)                                # [k, n, source]
    NBP = Dinp_3d.transpose(2, 1, 0).reshape(N, N * J)
    NNBP = np.kron(NBP, np.ones((J, 1)))
    GG = np.kron(np.ones((1, N)), G)
    OM = np.eye(J * N) - GG * NNBP

    aux = (om * (Ljn_hat ** (1 - B)) * (VARjn0 + VALjn0)).sum(axis=0) - Bnp
    aux2 = np.kron(aux, np.ones(J))
    rhs = alphas.flatten("F") * aux2  # MATLAB reshape(alphas, N*J, 1) is column-major
    X = np.linalg.solve(OM, rhs)
    return X.reshape(N, J).T


def GMCnew(
    Xp: np.ndarray,
    Dinp: np.ndarray,
    J: int,
    N: int,
    B: np.ndarray,
    gamma: np.ndarray,
    Ljn_hat: np.ndarray,
    VARjn0: np.ndarray,
    VALjn0: np.ndarray,
    R: int,
) -> np.ndarray:
    """Implied factor-price changes from market clearing.

    US states (cols 0..R-1) get sectoral wages; foreign countries (R..N-1)
    get a single national wage that's identical across sectors.
    """
    # MATLAB reshape(Xp', 1, J*N) is row-major on Xp (sector outer, dest inner).
    DP = Dinp * Xp.flatten()[:, None]                              # (J*N, N)
    Exjnp = DP.reshape(J, N, N).sum(axis=1)                         # (J, N) by source

    aux4 = gamma * Exjnp
    omef0 = np.empty((J, N))
    omef0[:, :R] = aux4[:, :R] / (
        (Ljn_hat[:, :R] ** (1 - B[:, :R])) * (VARjn0[:, :R] + VALjn0[:, :R])
    )

    VAR = VARjn0.sum(axis=0)
    VAL = VALjn0.sum(axis=0)
    aux5 = aux4.sum(axis=0)
    omef0[:, R:] = (aux5[R:] / (VAR[R:] + VAL[R:]))[None, :]
    return omef0
