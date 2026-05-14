"""Aggregate TFP / GDP / welfare elasticities to fundamental productivity shocks.

Port of Aggregate_elasticities_regional_shocks.m and
Aggregate_elasticities_sectoral_shocks.m.

The formula is
    elasticity = 10 * (X_hat - 1) / (shocked unit's share of total)
The factor of 10 renormalizes to a 1% shock — the runs use a 10% TFP
perturbation (shock = 1.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ElasticityRow:
    """Aggregate elasticities for one shocked region or sector."""

    TFP: float
    GDP: float
    welfare: float


def regional_elasticities(
    shock,
    *,
    region: int,
    Ln: np.ndarray,
) -> ElasticityRow:
    """Aggregate elasticities for one region's shock.

    `shock` must expose .TFP_hat, .GDP_hat, .V_hat, .Yn, .Y, .VAn0, .VA0.
    `Ln` is the baseline (N,) state-level population share.
    """
    return ElasticityRow(
        TFP=10.0 * (shock.TFP_hat - 1) / (shock.Yn[region] / shock.Y),
        GDP=10.0 * (shock.GDP_hat - 1) / (shock.VAn0[region] / shock.VA0),
        welfare=10.0 * (shock.V_hat - 1) / Ln[region],
    )


def sectoral_elasticities(
    shock,
    *,
    sector: int,
    Ljn: np.ndarray,
) -> ElasticityRow:
    """Aggregate elasticities for one sector's shock.

    `shock` must expose .TFP_hat, .GDP_hat, .V_hat, .Yj, .Y, .VAj0, .VA0.
    `Ljn` is (J, N) baseline sector-state employment shares.
    """
    return ElasticityRow(
        TFP=10.0 * (shock.TFP_hat - 1) / (shock.Yj[sector] / shock.Y),
        GDP=10.0 * (shock.GDP_hat - 1) / (shock.VAj0[sector] / shock.VA0),
        welfare=10.0 * (shock.V_hat - 1) / Ljn[sector, :].sum(),
    )
