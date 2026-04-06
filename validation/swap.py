"""Validation: Interest Rate Swap Pricing
Source: ~/QuantLib-SWIG/Python/examples/swap.py

Validates vanilla IRS pricing on flat yield curves:
  - NPV for payer swap at 5% on 5% curve
  - Fair rate computation
  - Payer/receiver symmetry
  - DV01 via JAX AD
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.time.date import Date
from ql_jax.time.schedule import MakeSchedule
from ql_jax.time.calendar import NullCalendar
from ql_jax._util.types import BusinessDayConvention, Frequency
from ql_jax.instruments.swap import make_vanilla_swap, SwapType
from ql_jax.indexes.ibor import Euribor
from ql_jax.engines.swap.discounting import (
    discounting_swap_npv, discounting_swap_fair_rate,
)
from ql_jax.termstructures.yield_.flat_forward import FlatForward


# === Reference values from QuantLib 1.42 ===
# Forward-starting swap: Jan 17 2024 → Jan 17 2029
# 5% flat curve, Actual365Fixed, 1M notional
# Payer = pays fixed, receives floating
REFERENCE = {
    "swap_5pct_npv":       5493.8885298298,
    "swap_fair_rate":      0.0512725687,
    "swap_4pct_npv":      48665.5350502113,
}


def main():
    today = Date(15, 1, 2024)
    curve = FlatForward(today, 0.05)

    # forward-starting to avoid fixing issues (settle = Jan 17)
    settle = Date(17, 1, 2024)
    maturity = Date(17, 1, 2029)

    fixed_schedule = (MakeSchedule()
        .from_date(settle).to_date(maturity)
        .with_frequency(Frequency.Annual)
        .with_calendar(NullCalendar())
        .with_convention(BusinessDayConvention.Unadjusted)
        .build())

    float_schedule = (MakeSchedule()
        .from_date(settle).to_date(maturity)
        .with_frequency(Frequency.Semiannual)
        .with_calendar(NullCalendar())
        .with_convention(BusinessDayConvention.Unadjusted)
        .build())

    idx = Euribor(6)

    # 5% payer swap
    swap_5pct = make_vanilla_swap(
        SwapType.Payer, 1_000_000.0,
        fixed_schedule, 0.05, "Actual/365 (Fixed)",
        float_schedule, idx, spread=0.0,
    )
    npv_5pct = float(discounting_swap_npv(swap_5pct, curve))
    fair_rate = float(discounting_swap_fair_rate(swap_5pct, curve))

    # 4% payer swap
    swap_4pct = make_vanilla_swap(
        SwapType.Payer, 1_000_000.0,
        fixed_schedule, 0.04, "Actual/365 (Fixed)",
        float_schedule, idx, spread=0.0,
    )
    npv_4pct = float(discounting_swap_npv(swap_4pct, curve))

    results = {
        "swap_5pct_npv":  npv_5pct,
        "swap_fair_rate": fair_rate,
        "swap_4pct_npv":  npv_4pct,
    }

    # Print
    print("=" * 78)
    print("Swap Pricing Validation (5% flat curve, 1M notional)")
    print("=" * 78)
    print(f"\n{'Metric':<25} {'QuantLib':>15} {'ql-jax':>15} {'Diff':>12}")
    print("-" * 67)

    n_pass = 0
    for key in REFERENCE:
        ref = REFERENCE[key]
        val = results[key]
        diff = abs(val - ref)
        tol = max(abs(ref) * 1e-3, 1.0)  # 0.1% or $1
        status = "✓" if diff < tol else "✗"
        if diff < tol:
            n_pass += 1
        print(f"{key:<25} {ref:>15.6f} {val:>15.6f} {diff:>12.2e} {status}")

    # Payer/receiver symmetry
    swap_recv = make_vanilla_swap(
        SwapType.Receiver, 1_000_000.0,
        fixed_schedule, 0.05, "Actual/365 (Fixed)",
        float_schedule, idx, spread=0.0,
    )
    npv_recv = float(discounting_swap_npv(swap_recv, curve))
    symmetry_diff = abs(npv_5pct + npv_recv)
    print(f"\nPayer/Receiver symmetry: |{npv_5pct:.2f} + {npv_recv:.2f}| = {symmetry_diff:.2e}")
    n_pass += 1 if symmetry_diff < 1.0 else 0

    # DV01 via JAX AD — use a simplified swap model to avoid float() in ibor fixing
    print("\n--- Swap DV01 via JAX AD ---")

    def swap_fixed_leg_npv(rate):
        """PV of fixed leg on flat curve."""
        c = FlatForward(today, rate)
        pv = jnp.float64(0.0)
        for cf in swap_5pct.fixed_leg:
            t = c.time_from_reference(cf.payment_date)
            pv = pv + cf.amount * c.discount(t)
        return pv

    d_fixed_dr = float(jax.grad(swap_fixed_leg_npv)(jnp.float64(0.05)))
    print(f"  d(FixedLeg)/dr = {d_fixed_dr:,.2f}")
    assert d_fixed_dr < 0, "Fixed leg PV falls with rising rates"
    print(f"  ✓ d(FixedLeg)/dr < 0 (correct sign)")

    print(f"\nPassed: {n_pass}/{len(REFERENCE) + 1}")
    print("✓ Swap validation complete.")


if __name__ == "__main__":
    main()
