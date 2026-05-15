"""Per-iteration math for the dynamic temporary-equilibrium solver (TVF).

Direct ports of P_h_om_tvf.m and Dinprime_tvf.m. The dynamic variants
differ from the static (Phase 1) helpers in two ways:

* Factor prices ``om`` are compared to a data-target ``om0`` — the
  input-bundle cost uses ``log(om / om0)`` instead of ``log(om)``.
* Trade shares are seeded with the empirical share ratio
  ``pi_tilde1 / pi_tilde0`` between consecutive quarters, scaled by
  ``pi`` (the previous quarter's converged shares). The combined
  ``pi_k`` term is precomputed once per quarter by ``solve_tvf`` and
  passed into both helpers.

``expenditure_tvf.m`` and ``GMC_tvf.m`` are mathematically identical to
the Phase 1 ``expenditurenew`` and ``GMCnew``, so we import those.
"""

from __future__ import annotations

import numpy as np

from qge.helpers import _inv_theta_per_jn


def shifted_pi_k(
    pi: np.ndarray,
    kappa_hat: np.ndarray,
    pi_tilde0: np.ndarray,
    pi_tilde1: np.ndarray,
    T: np.ndarray,
    N: int,
) -> np.ndarray:
    """``(pi_tilde1 / pi_tilde0) · pi · κ^(-1/θ)`` with NaN-from-0/0 zeroed."""
    LT_col = _inv_theta_per_jn(T, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi_k = (pi_tilde1 / pi_tilde0) * pi * (kappa_hat ** (-1.0 / LT_col))
    return np.where(np.isnan(pi_k), 0.0, pi_k)


def P_h_om_tvf(
    om: np.ndarray,
    om0: np.ndarray,
    A_hat: np.ndarray,
    T: np.ndarray,
    G: np.ndarray,
    gamma: np.ndarray,
    pi_k: np.ndarray,
    J: int,
    N: int,
    maxit: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-point on (p_hat, x_hat) given factor-price guess om."""
    G_3d = G.reshape(N, J, J)                            # [n, source, dest]
    pi_k_3d = pi_k.reshape(J, N, N)                       # [j, dest, source]
    T_col = T.reshape(-1, 1)
    # log(om / om0) is invariant in the inner while-loop — hoist it.
    log_om_term = gamma * np.log(om / om0)

    p_hat0 = np.ones((J, N))
    pfmax = 1.0
    it = 1
    while it <= maxit and pfmax > tol:
        lc = log_om_term + np.einsum("nsk,sn->kn", G_3d, np.log(p_hat0))
        x_hat = np.exp(lc)

        adjusted = (A_hat ** (gamma / T_col)) * (x_hat ** (-1.0 / T_col))
        p_hat = np.einsum("jdm,jm->jd", pi_k_3d, adjusted) ** (-T_col)

        pfmax = float(np.abs(p_hat - p_hat0).max())
        p_hat0 = p_hat
        it += 1
    return p_hat0, x_hat


def Dinprime_tvf(
    A_hat: np.ndarray,
    x_hat: np.ndarray,
    p_hat: np.ndarray,
    T: np.ndarray,
    J: int,
    N: int,
    gamma: np.ndarray,
    pi_k: np.ndarray,
) -> np.ndarray:
    """New bilateral trade shares given prices and the precomputed pi_k."""
    T_col = T.reshape(-1, 1)
    xp = x_hat ** (-1.0 / T_col)
    php = p_hat ** (-1.0 / T_col)
    DD = pi_k * np.kron(xp * (A_hat ** (gamma / T_col)), np.ones((N, 1)))
    return DD / php.ravel()[:, None]
