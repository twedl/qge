"""Per-iteration math for the CDP Base_Year solver.

Direct ports of P_h_om.m, Dinprime.m, expenditurenew.m, GMCnew.m. Each
function mirrors its MATLAB counterpart in indexing conventions; see the
module-level note on the trailing dimension order.

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


def P_h_om(
    om: np.ndarray,
    kappa_hat: np.ndarray,
    lambda_hat: np.ndarray,
    T: np.ndarray,
    B: np.ndarray,
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
    pf0 = np.ones((J, N))
    pfmax = 1.0
    it = 1
    while it <= maxit and pfmax > tol:
        lom = np.log(om)
        lp = np.log(pf0)
        # Input bundle cost: log c_jn = γ_jn log ω_jn + G_n^T log p_n
        lc = np.empty((J, N))
        for i in range(N):
            G_i = G[i * J:(i + 1) * J, :]  # (J, J): rows = source sec, cols = dest sec
            lc[:, i] = gamma[:, i] * lom[:, i] + G_i.T @ lp[:, i]
        c = np.exp(lc)

        # Reshape theta vector into LT of length J*N.
        LT = np.repeat(T.ravel(), N).reshape(-1, 1)  # (J*N, 1)
        Din_k = Din * (kappa_hat ** (-1.0 / LT))

        phat = np.empty((J, N))
        for j in range(J):
            block = Din_k[j * N:(j + 1) * N, :]  # (N, N)
            inner = (lambda_hat[j, :] ** (gamma[j, :] / T[j])) * (c[j, :] ** (-1.0 / T[j]))
            # phat[j, n] = (sum over source m of block[n, m] * inner[m]) ^ (-T[j])
            phat[j, :] = (block @ inner) ** (-T[j])

        pfdev = np.abs(phat - pf0)
        pf0 = phat
        pfmax = pfdev.max()
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
    LT = np.repeat(T.ravel(), N).reshape(-1, 1)
    cp = c ** (-1.0 / T.reshape(-1, 1))         # (J, N)
    phatp = phat ** (-1.0 / T.reshape(-1, 1))   # (J, N)

    Din_k = Din * (kappa_hat ** (-1.0 / LT))

    # DD[j*N + dest, source] = Din_k[j*N + dest, source]
    #                          · cp[j, source] · lambda_hat[j, source]^(γ/θ)
    # — the per-source adjustment is constant across destinations.
    DD = Din_k * np.kron(
        cp * (lambda_hat ** (gamma / T.reshape(-1, 1))),
        np.ones((N, 1)),
    )
    # Dinp[j*N + dest, source] = DD[...] / phatp[j, dest]
    # — denominator is the destination-side price index for sector j,
    # constant across sources. phatp.ravel()[j*N + dest] = phatp[j, dest].
    Dinp = DD / phatp.ravel()[:, None]
    return Dinp


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
    """Solve the linear system (I - Ω) X = α (income − deficits) for total expenditure."""
    VARjnp = VARjn0 * om * (Ljn_hat ** (1 - B))
    VARp = VARjnp.sum(axis=0)             # (N,)
    Chip = VARp.sum()
    Bnp = Snp.ravel() - io.ravel() * Chip + VARp

    # NBP[j, n*J + k] = Dinp[k*N + n, j]: rearrange (J*N, N) → (N, N*J)
    NBP = np.zeros((N, J * N))
    for j_row in range(N):  # MATLAB "for j = 1:N" but indexes destinations
        for n in range(N):
            NBP[j_row, n * J:(n + 1) * J] = Dinp[n::N, j_row]
    NNBP = np.kron(NBP, np.ones((J, 1)))
    GG = np.kron(np.ones((1, N)), G)
    GP = GG * NNBP

    OM = np.eye(J * N) - GP
    aux = (om * (Ljn_hat ** (1 - B)) * (VARjn0 + VALjn0)).sum(axis=0) - Bnp  # (N,)
    aux2 = np.kron(aux, np.ones(J))  # (J*N,)
    rhs = alphas.flatten("F") * aux2  # MATLAB reshape(alphas, N*J, 1) is column-major
    X = np.linalg.solve(OM, rhs)
    return X.reshape(N, J).T  # → (J, N)


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
    PQ_vec = Xp.flatten()
    DP = Dinp * PQ_vec[:, None]  # (J*N, N): per-(sector, dest, source) flow
    Exjnp = np.empty((J, N))
    for j in range(J):
        Exjnp[j, :] = DP[j * N:(j + 1) * N, :].sum(axis=0)

    aux4 = gamma * Exjnp
    omef0 = np.ones((J, N))
    omef0[:, :R] = aux4[:, :R] / (
        (Ljn_hat[:, :R] ** (1 - B[:, :R])) * (VARjn0[:, :R] + VALjn0[:, :R])
    )

    VAR = VARjn0.sum(axis=0)
    VAL = VALjn0.sum(axis=0)
    aux5 = aux4.sum(axis=0)
    omef0[:, R:] = np.broadcast_to(aux5[R:] / (VAR[R:] + VAL[R:]), (J, N - R))
    return omef0
