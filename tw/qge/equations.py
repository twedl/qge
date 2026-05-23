"""Hat-algebra inner loop (port of model_equations.m).

Given guesses for wage changes ``what`` (1, N) and employment changes ``dL``
(1, N) with ``dL[-1] = 1`` (ROW employment fixed), compute:

- Phat: change in sectoral price indices, shape (J, N)
- PIhat, PInew: change in bilateral trade shares and the new shares
- chat: change in input-bundle costs, shape (1, N*J)
- Sn: new trade surplus
- Inew: new total absorption per region
- wL_new, what1: implied vs guessed wage changes (residual driver)
- Lhat, Uhat, realGDPhat, lambda2: outcomes for the migration block
- aggregate_rGDPhat, aggregate_Uhat, aggregate_gain: aggregates
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .initialize import BaselineEquilibrium


@dataclass
class ModelOutputs:
    """All quantities produced by one inner pass."""

    Phat: np.ndarray         # (J, N)
    PIhat: np.ndarray        # (N*J, N)
    PInew: np.ndarray        # (N*J, N)
    chat: np.ndarray         # (1, N*J)
    Sn: np.ndarray           # (1, N)
    Inew: np.ndarray         # (1, N)
    wL_new: np.ndarray       # (1, N)
    what1: np.ndarray        # (1, N)
    Lhat: np.ndarray         # (1, N*J)
    aggPhat: np.ndarray      # (1, N)
    Uhat: np.ndarray         # (N, 1)
    realGDPhat: np.ndarray   # (N, 1)
    mu_new: np.ndarray       # (N-1, 1)
    lambda2: np.ndarray      # (N-1, 1)
    aggregate_rGDPhat: float
    aggregate_Uhat: float
    aggregate_gain: float

    # Leontief solve outputs (needed for some counterfactuals)
    Rnew: np.ndarray         # (1, N*J)
    Rnjnew: np.ndarray       # (J, N)
    Anew: np.ndarray         # (N*J, N*J)
    AXnew: np.ndarray        # (N*J, N*J)
    Fmatnew: np.ndarray      # (N*J, N)


def model_equations(
    baseline: BaselineEquilibrium,
    what: np.ndarray,        # (1, N) guess for wage changes
    dL: np.ndarray,          # (1, N) employment changes (dL[-1] = 1)
    *,
    tauhat: np.ndarray | None = None,   # (N*J, N), default ones
    Ahat: np.ndarray | None = None,     # (1, N*J), default ones
    Sn_override: np.ndarray | None = None,   # if set, use as new Sn (e.g. Sn0 + Sn_EQ)
    frozen_idx: tuple[int, ...] = (),    # province indices with exogenous labor
                                          # (like WLD). They get lambda2_p = lambda_p
                                          # and dL_p = 1; the Frechet pool is
                                          # restricted to the remaining provinces
                                          # sharing (1 - Σ_{frozen} λ) workers.
) -> ModelOutputs:
    """One pass of the hat-algebra inner loop. Returns all derived quantities.

    Mirrors model_equations.m line by line.
    """
    b = baseline
    N, J = b.N, b.J
    theta = b.theta                  # (J, 1)
    phi = b.phi                      # (1, N*J)
    gamma = b.gamma                  # (J, N*J)
    alpha = b.alpha                  # (J, N)
    PI = b.PI                        # (N*J, N)
    wL = b.wL                        # (1, N)
    lambda_ = b.lambda_              # (N-1, 1)
    mu = b.mu                        # (N-1, 1)
    R = b.R                          # (1, N*J)
    f = b.f
    kappa = b.kappa

    if tauhat is None:
        tauhat = np.ones((N * J, N))
    if Ahat is None:
        Ahat = np.ones((1, N * J))

    ones_NJ_col = np.ones((N * J, 1))
    ones_J_row = np.ones((1, J))
    sum_to_region = np.kron(np.eye(N), np.ones((J, 1)))

    # ── Given wage changes, find prices via fixed point ───────────────────
    # Phat: (J, N) — change in sectoral price indices
    Phat = np.ones((J, N))
    Phat_prev = np.zeros((J, N))
    while np.sum((Phat_prev - Phat) ** 2) > 1e-20:
        Phat_prev = Phat
        # chat: change in cost of input bundle for each (region, sector).
        # Shape (1, N*J).  chat[(n)*J+j] = w_n^φ * Π_k (P̂^k_n)^γ^{k,j}_n / Â_{n,j}
        what_rep = np.kron(what, ones_J_row)              # (1, N*J)
        Phat_kron = np.kron(Phat_prev, ones_J_row)        # (J, N*J)
        chat = (what_rep ** phi) * np.prod(Phat_kron ** gamma, axis=0
                                            ).reshape(1, N * J) ** (1 - phi) / Ahat

        # Phat update from CES price index formula
        # PI is (N*J, N), tauhat is (N*J, N), chat'·ones(1,N) is (N*J, N).
        # power(-theta) where theta[j] broadcasts over rows k = i*J+j
        theta_mat = np.tile(theta.reshape(J, 1), (N, N))      # (N*J, N)
        cost_term = tauhat * (chat.T @ np.ones((1, N)))       # (N*J, N)
        inner = PI * (cost_term ** (-theta_mat))               # (N*J, N)
        # Sum over the J entries within each region n (k = (n')*J + j, NOT n).
        # MATLAB: kron(ones(1, N), eye(J, J)) * (...) is (J, N) where row j and
        # col n sums entries at rows k where (k mod J) == j (across all
        # source regions n'). Equivalent: reshape to (N, J, N), sum over
        # axis 0 (source). But the original code uses axes (k_row, n_dest)
        # — we sum across the source-rows that share sector j.
        # Use a sparse pattern: M[(n')*J + j, n] sums to entry [j, n] iff
        # they share j across all n'.
        # NumPy equivalent: inner.reshape(N, J, N).sum(axis=0)  → shape (J, N).
        # That matches kron(ones(1,N), eye(J,J)) @ inner.
        inner_sum = inner.reshape(N, J, N).sum(axis=0)        # (J, N)
        Phat = inner_sum ** (-1.0 / np.tile(theta, (1, N)))    # (J, N)
        Phat[np.isinf(Phat)] = 1

    # ── Given prices and wages, change in trade shares ────────────────────
    PIhat = (tauhat * (chat.T @ np.ones((1, N))) / np.kron(np.ones((N, 1)),
              Phat)) ** (-np.tile(theta, (N, N)))                 # (N*J, N)
    PInew = PI * PIhat

    # ── New trade imbalances ─────────────────────────────────────────────
    if Sn_override is None:
        w = wL[0, : N - 1] / lambda_.ravel()                       # (N-1,)
        what_provs = what[0, : N - 1]                               # (N-1,)
        dL_provs = dL[0, : N - 1]                                   # (N-1,)
        wL_provs = wL[0, : N - 1]
        wLshare_new = (what_provs * dL_provs * wL_provs) / (what_provs * dL_provs * wL_provs).sum()
        B = 1.0 / (wLshare_new * (what_provs * w) ** (-f)).sum()
        a = B * (what_provs * w) ** (-f)                            # (N-1,)
        Sn_provs = (1 - a) * what_provs * w * lambda_.ravel() * dL_provs
        Sn = np.concatenate([Sn_provs, [0.0]]).reshape(1, N) + b.Sn_exogenous
    else:
        Sn = Sn_override

    # ── New incomes ──────────────────────────────────────────────────────
    Inew = what * dL * wL - Sn                                    # (1, N)

    # ── Counterfactual wage changes via Leontief inverse ─────────────────
    Fmatnew = (ones_NJ_col @ Inew) * np.tile(alpha, (N, 1)) * PInew    # (N*J, N)
    Anew = (ones_NJ_col @ (1 - phi)) * np.tile(gamma, (N, 1)) * np.kron(PInew, ones_J_row)
    Rnew_col = np.linalg.solve(np.eye(N * J) - Anew, Fmatnew.sum(axis=1))
    Rnew = Rnew_col.reshape(1, N * J)
    wL_new = (phi * Rnew) @ sum_to_region                         # (1, N)
    AXnew = Anew * (ones_NJ_col @ Rnew)                           # (N*J, N*J)
    what1 = (wL_new / wL) / dL                                    # (1, N)
    Rnjnew = Rnew.reshape(N, J).T                                  # (J, N)

    # ── Outcomes ─────────────────────────────────────────────────────────
    Lhat = (Rnew / R) / np.kron(what, ones_J_row)                 # (1, N*J)
    aggPhat = np.prod(Phat ** alpha, axis=0).reshape(1, N)        # (1, N)
    Uhat = (((Inew / b.I) / dL) / aggPhat).reshape(N, 1)          # (N, 1)
    realGDPhat = (what / aggPhat).reshape(N, 1)                   # (N, 1)

    mu_new_unnorm = mu * realGDPhat[: N - 1]                       # (N-1, 1)
    mu_new = mu_new_unnorm / mu_new_unnorm.mean()                  # (N-1, 1)

    # Migration: lambda2 = lambda * Uhat^κ / sum(lambda * Uhat^κ)
    Uhat_provs = Uhat[: N - 1]                                     # (N-1, 1)
    weighted = lambda_ * Uhat_provs ** kappa
    if frozen_idx:
        # Pin frozen provinces at baseline labor share; remaining workers
        # (1 - Σ_frozen λ) get distributed across the free provinces by Frechet.
        frozen_mask = np.zeros(N - 1, dtype=bool)
        for k in frozen_idx:
            frozen_mask[k] = True
        free_mask = ~frozen_mask
        lambda2 = lambda_.copy()
        free_sum = weighted[free_mask].sum()
        remaining = 1.0 - lambda_[frozen_mask].sum()
        lambda2[free_mask] = weighted[free_mask] / free_sum * remaining
    else:
        lambda2 = weighted / weighted.sum()                        # (N-1, 1)

    # Aggregate changes
    omega = b.omega
    omega2 = b.omega2
    aggregate_rGDPhat = float((omega * realGDPhat[: N - 1] * lambda2 / lambda_).sum())
    aggregate_Uhat = float((omega2 * Uhat_provs * lambda2 / lambda_).sum())
    aggregate_gain = float((lambda_ * Uhat_provs * (lambda2 / lambda_) ** (-1 / kappa)).sum())

    return ModelOutputs(
        Phat=Phat, PIhat=PIhat, PInew=PInew, chat=chat, Sn=Sn,
        Inew=Inew, wL_new=wL_new, what1=what1, Lhat=Lhat,
        aggPhat=aggPhat, Uhat=Uhat, realGDPhat=realGDPhat,
        mu_new=mu_new, lambda2=lambda2,
        aggregate_rGDPhat=aggregate_rGDPhat,
        aggregate_Uhat=aggregate_Uhat,
        aggregate_gain=aggregate_gain,
        Rnew=Rnew, Rnjnew=Rnjnew, Anew=Anew, AXnew=AXnew, Fmatnew=Fmatnew,
    )
