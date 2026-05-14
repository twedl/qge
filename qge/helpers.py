"""Per-iteration helpers shared across the MATLAB drivers.

The MATLAB layout uses a (J*N, N) "stacked" matrix where each row r encodes
(sector j, destination n) as r = j*N + n (0-indexed), and the column is the
source. Reshaping to (J, N, N) gives axes (sector, destination, source) which
is what most of these operations want.
"""

from __future__ import annotations

import numpy as np


def _exports_by_source(D: np.ndarray, X: np.ndarray, J: int, N: int) -> np.ndarray:
    """Trade flows summed over destinations: (J, N) [sector, source].

    D is the (J*N, N) stacked bilateral-share matrix; X is total expenditure
    per (sector, dest), of shape (J, N) or (J*N,). The MATLAB equivalent is
    `sum(D(1+N*(j-1):N*j, :), 1)'` per sector j (then stacked).
    """
    return (D * X.reshape(-1)[:, None]).reshape(J, N, N).sum(axis=1)


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
    *,
    tol: float = 1e-25,
    maxit: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Inner fixed point on goods prices given factor prices om.

    Mirrors P_h_om.m. Returns (phat (J,N), c (J,N)).
    """
    T = np.asarray(T).reshape(-1)
    om = np.asarray(om).reshape(-1)

    LT = np.repeat(T, N)  # (J*N,)
    Din_k = Din * (kappa_hat ** (-1.0 / LT[:, None]))  # (J*N, N)
    Din_k_3d = Din_k.reshape(J, N, N)
    G_3d = G.reshape(N, J, J)
    lom = np.log(om)  # (N,)

    pf0 = np.ones((J, N))
    pfmax = np.inf
    it = 0
    while it < maxit and pfmax > tol:
        lp = np.log(pf0)
        # lc[j, n] = gamma[j, n] * lom[n] + sum_k G_3d[n, k, j] * lp[k, n]
        lc = gamma * lom[None, :] + np.einsum("nkj,kn->jn", G_3d, lp)
        c = np.exp(lc)

        inner = (c ** (-1.0 / T[:, None])) * (lambda_hat ** (gamma / T[:, None]))  # (J,N)
        # phat_raw[j, n] = sum_k Din_k_3d[j, n, k] * inner[j, k]
        phat_raw = np.einsum("jnk,jk->jn", Din_k_3d, inner)
        phat = phat_raw ** (-T[:, None])

        pfmax = float(np.max(np.abs(phat - pf0)))
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
    """Updated bilateral trade shares D'_in."""
    T = np.asarray(T).reshape(-1)
    LT = np.repeat(T, N)
    cp = c ** (-1.0 / T[:, None])  # (J, N)
    phatp = phat ** (-1.0 / T[:, None])  # (J, N)

    Din_k = Din * (kappa_hat ** (-1.0 / LT[:, None]))
    M = cp * (lambda_hat ** (gamma / T[:, None]))  # (J, N)

    Din_k_3d = Din_k.reshape(J, N, N)
    DD_3d = Din_k_3d * M[:, None, :]
    Dinp_3d = DD_3d / phatp[:, :, None]
    return Dinp_3d.reshape(J * N, N)


def Lchange(
    om: np.ndarray,
    phat: np.ndarray,
    alphas: np.ndarray,
    B: np.ndarray,
    L_hat: np.ndarray,
    Ln: np.ndarray,
    LnIn: np.ndarray,
    Bn,
    Snp,
    VAR: np.ndarray,
    io: np.ndarray,
    *,
    tol: float = 1e-10,
    maxit: int = 1_000_000,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Inner fixed point on relative employment changes L_hat."""
    om = np.asarray(om).reshape(-1)
    L_hat = np.asarray(L_hat, dtype=float).reshape(-1).copy()
    Ln = np.asarray(Ln).reshape(-1)
    LnIn = np.asarray(LnIn).reshape(-1)
    VAR = np.asarray(VAR).reshape(-1)
    io = np.asarray(io).reshape(-1)
    Bn_arr = np.asarray(Bn, dtype=float).reshape(-1) if np.ndim(Bn) else float(Bn)
    Snp_arr = np.asarray(Snp, dtype=float).reshape(-1) if np.ndim(Snp) else float(Snp)

    phi_n = LnIn / (LnIn + Bn_arr)
    b1 = B[0, :]
    P_index_hat = np.prod(phat ** alphas, axis=0)  # (N,)
    # Matches `if Bn == 0; Bn = 1` in MATLAB: a sentinel for autarky-like cases
    Bn_eff = 1.0 if np.all(Bn_arr == 0) else Bn_arr

    Lmax = np.inf
    V_hat = 0.0
    it = 0
    while it < maxit and Lmax > tol:
        VARp = VAR * om * (L_hat ** (1 - b1))
        Chip = float(np.sum(io * VARp))
        Chinp = Ln * L_hat * Chip
        Bnp = Snp_arr - Chinp + io * VARp
        omreal = om / P_index_hat

        V_hat = float(np.sum(
            (Ln / phi_n) * omreal * (L_hat ** (1 - b1))
            - Ln * ((1 - phi_n) / phi_n) * (Bnp / Bn_eff) / P_index_hat
        ))

        num = (
            om
            / (
                phi_n * (P_index_hat * V_hat)
                + (1 - phi_n) * ((Bnp / Bn_eff) / L_hat)
            )
        ) ** (1.0 / b1)
        den = float(np.sum(Ln * num))
        L_hat1 = num / den

        Lmax = float(np.sum(np.abs(L_hat1 - L_hat)))
        L_hat = L_hat1
        it += 1

    return L_hat, V_hat, P_index_hat


def expenditure(
    J: int,
    N: int,
    alphas: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    Dinp: np.ndarray,
    om: np.ndarray,
    L_hat: np.ndarray,
    Ln: np.ndarray,
    Snp,
    VAR: np.ndarray,
    VAL: np.ndarray,
    io: np.ndarray,
) -> np.ndarray:
    """Total expenditure X' by sector and region, (J, N)."""
    om = np.asarray(om).reshape(-1)
    L_hat = np.asarray(L_hat).reshape(-1)
    Ln = np.asarray(Ln).reshape(-1)
    VAR = np.asarray(VAR).reshape(-1)
    VAL = np.asarray(VAL).reshape(-1)
    io = np.asarray(io).reshape(-1)
    Snp_arr = np.asarray(Snp, dtype=float) if np.ndim(Snp) else float(Snp)
    b1 = B[0, :]

    Lnp = Ln * L_hat
    VARp = VAR * om * (L_hat ** (1 - b1))
    Chip = float(np.sum(io * VARp))
    Chinp = Lnp * Chip
    Bnp = Snp_arr - Chinp + io * VARp  # (N,)

    # NBP[j, n*J + k] = Dinp_3d[k, n, j]   (j=source, n=dest, k=sector)
    Dinp_3d = Dinp.reshape(J, N, N)
    NBP = Dinp_3d.transpose(2, 1, 0).reshape(N, N * J)
    NNBP = np.kron(NBP, np.ones((J, 1)))  # (J*N, J*N)
    GG = np.kron(np.ones((1, N)), G)  # (J*N, J*N)
    GP = GG * NNBP
    OM = np.eye(J * N) - GP

    aux = om * (L_hat ** (1 - b1)) * (VAR + VAL) - Bnp  # (N,)
    aux2 = np.repeat(aux, J)  # (N*J,)
    alphas_flat = alphas.ravel(order="F")  # MATLAB column-major: (N*J,)
    rhs = alphas_flat * aux2
    X = np.linalg.solve(OM, rhs)
    return X.reshape(N, J).T  # (J, N)


def GMC(
    Xp: np.ndarray,
    Dinp: np.ndarray,
    J: int,
    N: int,
    B: np.ndarray,
    gamma: np.ndarray,
    LnIn: np.ndarray,
    L_hat: np.ndarray,
    VAR: np.ndarray,
    VAL: np.ndarray,
) -> np.ndarray:
    """Goods-market-clearing implied wage update omef0, shape (N,)."""
    L_hat = np.asarray(L_hat).reshape(-1)
    VAR = np.asarray(VAR).reshape(-1)
    VAL = np.asarray(VAL).reshape(-1)
    b1 = B[0, :]

    DDDinpt = _exports_by_source(Dinp, Xp, J, N)  # (J, N)
    aux5 = np.sum(gamma * DDDinpt, axis=0)  # (N,)
    return aux5 / ((L_hat ** (1 - b1)) * (VAR + VAL))


def GOTFP(
    c: np.ndarray,
    phat: np.ndarray,
    Exjn0: np.ndarray,
    J: int,
    N: int,
):
    """Gross-output TFP accounting (GOTFP.m).

    Returns (ATFP, TFPj, TFPn, TFPjn, Yjn, Yj, Yn, Y).
    """
    TFPjn = c / phat
    Yjn = Exjn0
    Yj = np.nansum(Yjn, axis=1)  # (J,)
    Yn = np.nansum(Yjn, axis=0)  # (N,)
    Y = float(np.nansum(Yj))

    TFPj = np.nansum((Yjn / Yj[:, None]) * TFPjn, axis=1)  # (J,)
    TFPn = np.nansum((Yjn / Yn[None, :]) * TFPjn, axis=0)  # (N,)
    ATFP = float(np.nansum((Yj / Y) * TFPj))
    return ATFP, TFPj, TFPn, TFPjn, Yjn, Yj, Yn, Y


def GDP(
    phat: np.ndarray,
    wf0: np.ndarray,
    Ljn_hat: np.ndarray,
    VALjn: np.ndarray,
    VARjn: np.ndarray,
    J: int,
    N: int,
):
    """GDP accounting (GDP.m).

    Returns (AGDP, GDPj, GDPn, VAjn0, VAj0, VAn0, VA0).
    """
    wf0 = np.asarray(wf0).reshape(-1)
    GDPjn_hat = (Ljn_hat * wf0[None, :]) / phat
    VAjn0 = VALjn + VARjn
    VAj0 = np.nansum(VAjn0, axis=1)  # (J,)
    VAn0 = np.nansum(VAjn0, axis=0)  # (N,)
    VA0 = float(np.nansum(VAj0))

    GDPj = np.nansum((VAjn0 / VAj0[:, None]) * GDPjn_hat, axis=1)  # (J,)
    GDPn = np.nansum((VAjn0 / VAn0[None, :]) * GDPjn_hat, axis=0)  # (N,)
    AGDP = float(np.nansum((VAj0 / VA0) * GDPj))
    return AGDP, GDPj, GDPn, VAjn0, VAj0, VAn0, VA0


def neweq(
    J: int,
    N: int,
    Xp: np.ndarray,
    Dinp: np.ndarray,
    G: np.ndarray,
    B: np.ndarray,
    gamma: np.ndarray,
    L_j_n: np.ndarray,
    wf0: np.ndarray,
    VALjn0: np.ndarray,
    io: np.ndarray,
    L_hat: np.ndarray,
    Ln: np.ndarray,
) -> dict:
    """Repackage converged state as a new baseline economy.

    Returns a dict; the keys mirror the MATLAB output positions of neweq.m
    (Ln_out, xbilat_out, VAR_out, VAL_out, Ljn_out, Chi_out, Chin_out,
     LnIn_out, Sn_out, TD_out).
    """
    wf0 = np.asarray(wf0).reshape(-1)
    L_hat = np.asarray(L_hat).reshape(-1)
    Ln = np.asarray(Ln).reshape(-1)
    io = np.asarray(io).reshape(-1)
    b1 = B[0, :]

    PQ_vec = Xp.reshape(-1)  # (J*N,)
    xbilat_new = Dinp * PQ_vec[:, None]  # (J*N, N)
    xbilat_3d = xbilat_new.reshape(J, N, N)  # [sector, dest, source]

    M = xbilat_3d.sum(axis=2)  # (J, N) [sector, dest]
    E = xbilat_3d.sum(axis=1)  # (J, N) [sector, source]
    TD = E.sum(axis=0) - M.sum(axis=0)  # (N,)

    X0 = xbilat_new.sum(axis=1).reshape(J, N)  # (J, N)
    G_3d = G.reshape(N, J, J)
    # aux2[:, n] = X0[:, n] - G_3d[n] @ E[:, n]
    aux2 = X0 - np.einsum("njk,kn->jn", G_3d, E)
    VAL_out = (1 - b1) * (TD + aux2.sum(axis=0))  # (N,)
    VAR_out = (b1 / (1 - b1)) * VAL_out  # (N,)

    # New Ljn using new exports E (== Exjn in MATLAB)
    aux_w = wf0[None, :]
    Ljn_hat = (1.0 / (VALjn0 * aux_w)) * gamma * (1 - B) * E  # (J, N)
    Ljn_norm = L_j_n / L_j_n.sum()
    Ljn_out = Ljn_hat * Ljn_norm
    Ln_out = L_hat * Ln  # MATLAB overwrites the nansum-based value; we skip the dead line

    Chi_out = float(np.sum(io * VAR_out))
    Chin_out = Ln_out * Chi_out
    Sn_out = (Chin_out - io * VAR_out) + TD
    LnIn_out = VAL_out + VAR_out + Chin_out - io * VAR_out

    return dict(
        Ln=Ln_out,
        xbilat=xbilat_new,
        VAR=VAR_out,
        VAL=VAL_out,
        Ljn=Ljn_out,
        Chi=Chi_out,
        Chin=Chin_out,
        LnIn=LnIn_out,
        Sn=Sn_out,
        TD=TD,
    )
