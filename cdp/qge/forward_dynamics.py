"""Forward-looking dynamics primitives — Phase 2c (Step 3 of CDP §3.1).

Pure-math helpers used by the post-2007 constant-fundamentals forward
simulation. None of these functions solve a temporary equilibrium; they
manipulate value functions, migration flows, and labor allocations.

* ``compute_mu_path``  — given a value-function time series ``Yt``, build
  the (RJ1, RJ1, time) migration-probability path via the Bellman
  formula. Each ``mu[t]`` is row-stochastic.
* ``evolve_labor_forward`` — apply the mu path forward to a starting
  labor allocation.
* ``bellman_update_Y`` — compute the next iterate of the value function
  given a candidate path of mu and the equilibrium real-wage path.

The dispersion ν and discount β are CDP-paper constants.
"""

from __future__ import annotations

import numpy as np

BETA = 0.99       # quarterly discount factor
NU = 5.3436       # dispersion of taste shocks (CDP Table 4)


def compute_mu_path(
    mu_init: np.ndarray, Yt: np.ndarray, *, beta: float = BETA
) -> np.ndarray:
    """Build a (RJ1, RJ1, time) migration-probability path from values.

    ``mu_init`` is the seed migration matrix (last quarter of Phase 2a).
    ``Yt`` is ``(RJ1, time)`` — the value function ``Y_l(t) = (1/ν) ·
    exp(V_l(t))`` for each labor market l and quarter t. The recursion is

        mu[t][i, k] ∝ mu[t-1][i, k] · Yt[k, t+1]^β

    with ``mu[0][i, k] ∝ mu_init[i, k] · Yt[k, 1]^β`` and row-sum
    normalization.
    """
    RJ1, time = Yt.shape
    mu = np.empty((RJ1, RJ1, time))

    # mu[0] uses the Phase 2a boundary and Yt at t = 1.
    num = mu_init * (Yt[:, 1] ** beta)[None, :]
    mu[..., 0] = num / num.sum(axis=1, keepdims=True)

    for t in range(time - 2):
        num = mu[..., t] * (Yt[:, t + 2] ** beta)[None, :]
        mu[..., t + 1] = num / num.sum(axis=1, keepdims=True)

    # Last slice left as the carry-forward of the previous iteration.
    mu[..., time - 1] = mu[..., time - 2]
    return mu


def evolve_labor_forward(
    mu_path: np.ndarray, L0: np.ndarray, *, J: int, R: int
) -> np.ndarray:
    """Apply ``mu_path`` forward to produce a (J+1, R, time) labor path.

    ``L0`` is the (RJ1,) labor allocation at t = 0; ``mu_path`` has
    shape ``(RJ1, RJ1, time)``. Returns ``Ldyn`` reshaped as US labor
    markets with the non-employment row at index 0. The final slice is
    set to zero per the MATLAB convention.
    """
    RJ1, _, time = mu_path.shape
    Ldyn = np.empty((RJ1, time))
    Ldyn[:, 0] = L0
    for t in range(time - 2):
        Ldyn[:, t + 1] = mu_path[..., t].T @ Ldyn[:, t]
    Ldyn[:, time - 1] = 0.0
    return Ldyn.reshape(J + 1, R, time, order="F")


def bellman_update_Y(
    Yt: np.ndarray, mu_path: np.ndarray, realwages: np.ndarray,
    *, R: int, J: int, beta: float = BETA, nu: float = NU,
) -> np.ndarray:
    """One Bellman update on the value function ``Yt``.

    Returns the next iterate ``Y_new``. The recursion is

        Y_new[i, t] = Σ_k μ[t-1][i, k] · rw[k, t]^(1/ν) · Yt[k, t+1]^β

    for US labor markets (the non-employment row at index 0 in each
    state's block is treated as rw = 1). Foreign countries are not
    workers in this model, so the update is US-only.
    """
    RJ1, time = Yt.shape

    # Pad realwages with a non-employment "row 0" of ones, US states only.
    realwages_padded = np.ones((J + 1, R, time))
    realwages_padded[1:, :, :] = realwages[:, :R, :]
    # Reshape to (RJ1, time) with the same Fortran ordering as labor flow.
    rw_us = realwages_padded.reshape(RJ1, time, order="F")

    # rwagenu[i, k, t] = μ[t-1][i, k] · rw[k, t]^(1/ν), for t >= 1.
    rwnu_per_k = rw_us ** (1.0 / nu)                          # (RJ1, time)
    rwagenu = np.zeros((RJ1, RJ1, time))
    rwagenu[..., 1:] = mu_path[..., :-1] * rwnu_per_k[None, :, 1:]

    # num[i, k, t] = rwagenu[i, k, t] · Yt[k, t+1]^β
    num = np.zeros_like(rwagenu)
    for t in range(time - 1):
        num[..., t] = rwagenu[..., t] * (Yt[:, t + 1] ** beta)[None, :]

    Y_new = num.sum(axis=1)                                   # (RJ1, time)
    Y_new[:, -1] = 1.0
    return Y_new
