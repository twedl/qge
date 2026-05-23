"""Alberta-only fiscal-transfer removal counterfactual.

Models "Alberta withdraws from the federal transfer system; the missing
contribution is absorbed uniformly per worker by the other nine provinces."

Mechanics:
  Baseline structural transfer vector Tn (N-1 provinces, sum=0 by closure).
  Construct T̃ such that:
    - T̃_AB = 0
    - T̃_p  = Tn_p + Tn_AB × λ_p / Σ_{q≠AB} λ_q   for p ≠ AB
  Verify Σ T̃_p = 0 (within-Canada closure preserved; WLD untouched).

  Then the only deltas vs baseline are:
    - Shift Sn_exogenous by (Tn − T̃) per province
    - Keep f at the baseline value 0.3268 (income-sensitive piece unchanged)
    - No trade-cost shock

Run from tw/:
    TW_CALIBRATION=canada_2022 uv run python scripts/counterfactual_ab_transfer_exit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qge.initialize import initialize  # noqa: E402
from qge.io import load_inputs  # noqa: E402
from qge.solver import solve  # noqa: E402


DISPLAY_ORDER = ("BC", "AB", "SK", "MB", "ON", "QC", "NB", "NS", "PE", "NL")


def _display(label, result, b, region_index):
    out = result.outputs
    N = b.N
    ri = (out.Uhat.ravel()[: N - 1] - 1) * 100
    rw = (out.realGDPhat.ravel()[: N - 1] - 1) * 100
    empl = (out.lambda2.ravel() / b.lambda_.ravel() - 1) * 100
    print(f"\n=== {label} ===\n")
    print(f"  {'Prov':<5} {'R. Inc.':>10} {'R. Wage':>10} {'Empl':>10}")
    print(f"  {'-' * 38}")
    for name in DISPLAY_ORDER:
        i = region_index[name]
        print(f"  {name:<5} {ri[i]:+10.2f} {rw[i]:+10.2f} {empl[i]:+10.2f}")
    print(f"  {'-' * 38}")
    print(f"  {'CAN':<5} {100*(out.aggregate_Uhat-1):+10.2f} "
          f"{100*(out.aggregate_rGDPhat-1):+10.2f} {'-':>10}")


def main() -> None:
    raw = load_inputs()
    region_index = {r: i for i, r in enumerate(raw.regions)}
    provs = [r for r in raw.regions if r != "WLD"]
    ab_idx = provs.index("AB")
    N = len(raw.regions)

    # ── Build the modified transfer vector ────────────────────────────────
    b = initialize(raw)
    Tn = b.Tn.ravel().copy()                       # (N-1,) per-province dollars
    lam = b.lambda_.ravel()
    T_AB = Tn[ab_idx]
    other_idx = [i for i in range(len(provs)) if i != ab_idx]
    sum_other_lam = lam[other_idx].sum()

    T_new = Tn.copy()
    T_new[ab_idx] = 0.0
    for i in other_idx:
        T_new[i] = Tn[i] + T_AB * lam[i] / sum_other_lam

    # Closure check
    assert abs(T_new.sum()) < 1e-12, (
        f"Closure violated: sum(T_new) = {T_new.sum():.2e}"
    )

    # ── Diagnostics ───────────────────────────────────────────────────────
    print("=== Per-worker structural transfer flows ===\n")
    nat_inc = raw.I_baseline.sum()  # normalized to 1
    print(f"  {'Prov':<5} {'baseline T*λ':>14} {'new T*λ':>12} "
          f"{'Δ (pp inc)':>14}")
    print(f"  {'-' * 50}")
    for name in DISPLAY_ORDER:
        i = provs.index(name)
        baseline_flow_pct = Tn[i] * lam[i] / nat_inc * 100
        new_flow_pct = T_new[i] * lam[i] / nat_inc * 100
        delta_pct = (new_flow_pct - baseline_flow_pct)
        print(f"  {name:<5} {baseline_flow_pct:+14.3f}% "
              f"{new_flow_pct:+11.3f}% {delta_pct:+13.3f}pp")
    print(f"  {'-' * 50}")
    print(f"  AB net contribution removed: {abs(T_AB*lam[ab_idx])/nat_inc*100:.3f}% of nat'l income")
    print(f"  Absorbed by 9 others, per worker: "
          f"{-T_AB/sum_other_lam:+.6f} (dollars per λ unit)")
    print()

    # ── Counterfactual: shift Sn_exogenous by (Tn − T̃) ───────────────────
    # Sn_exogenous_p ← Sn_exogenous_p + (Tn_p − T_new_p)
    delta_T = (Tn - T_new).reshape(1, -1)            # (1, N-1)
    delta_T_full = np.concatenate(
        [delta_T, np.array([[0.0]])], axis=1)         # (1, N) — WLD untouched
    b.Sn_exogenous = b.Sn_exogenous + delta_T_full
    # f stays at the baseline value (0.3268) — income-sensitive piece active.

    result = solve(b, verbose=True)
    _display("AB exits federal transfer system "
              "(per-worker absorption by RoC, income-sensitive transfers active)",
              result, b, region_index)

    # ── Reference: Table 4 col 1 — remove ALL fiscal transfers ────────────
    b_ref = initialize(raw)
    b_ref.Sn_exogenous = b_ref.SnBar.copy()
    b_ref.f = 0.0
    result_ref = solve(b_ref, verbose=True)
    # NOTE: Table 4 reports 1/Uhat - 1 (i.e., the effect of putting transfers
    # back IN). Here we report Uhat - 1 directly (effect of removing them) so
    # the sign convention matches the AB-only experiment.
    _display("Reference: remove ALL fiscal transfers (Table 4 col 1, sign-flipped)",
              result_ref, b_ref, region_index)


if __name__ == "__main__":
    main()
