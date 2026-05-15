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

from qge.helpers import BETA, NU


def compute_mu_path(
    mu_init: np.ndarray, Yt: np.ndarray, *, beta: float = BETA
) -> np.ndarray:
    """Build a (RJ1, RJ1, time) migration-probability path from values.

    ``mu_init`` is the seed migration matrix (last quarter of Phase 2a).
    ``Yt`` is ``(RJ1, time)`` — the value function ``Y_l(t) = (1/ν) ·
    exp(V_l(t))`` for each labor market l and quarter t. The recursion is

        mu[t][i, k] ∝ mu[t-1][i, k] · Yt[k, t+1]^β

    with ``mu[0][i, k] ∝ mu_init[i, k] · Yt[k, 1]^β`` and row-sum
    normalization. The last slice is left as a carry-forward of the
    previous (the MATLAB reference leaves it at zeros; both conventions
    are unused downstream because consumers index ``[..., :-1]``).
    """
    RJ1, time = Yt.shape
    mu = np.empty((RJ1, RJ1, time))

    num = mu_init * (Yt[:, 1] ** beta)[None, :]
    mu[..., 0] = num / num.sum(axis=1, keepdims=True)

    for t in range(time - 2):
        num = mu[..., t] * (Yt[:, t + 2] ** beta)[None, :]
        mu[..., t + 1] = num / num.sum(axis=1, keepdims=True)

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

    for ``t = 1..time-2``; ``Y_new[:, 0] = 0`` (the boundary lookback
    has no μ predecessor) and ``Y_new[:, -1] = 1`` (the steady-state
    terminal). Foreign countries are not workers, so the update is
    US-only — we pad the non-employment row with ones.

    Computed in one einsum over the k axis so the (RJ1, RJ1, time)
    tensor never materializes — peak memory stays at the (RJ1, time-2)
    weights array (~1.8 MB) instead of two (RJ1, RJ1, time) ones (~4.2 GB).
    """
    RJ1, time = Yt.shape

    realwages_padded = np.ones((J + 1, R, time))
    realwages_padded[1:, :, :] = realwages[:, :R, :]
    rw_us = realwages_padded.reshape(RJ1, time, order="F")
    rwnu_per_k = rw_us ** (1.0 / nu)

    # weights[k, s] = rwnu_per_k[k, s+1] · Yt[k, s+2]^β for s = 0..time-3.
    weights = rwnu_per_k[:, 1:time - 1] * (Yt[:, 2:time] ** beta)
    Y_new = np.zeros((RJ1, time))
    Y_new[:, 1:time - 1] = np.einsum(
        "iks,ks->is", mu_path[..., :time - 2], weights,
    )
    Y_new[:, -1] = 1.0
    return Y_new
