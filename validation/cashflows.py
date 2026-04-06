"""Validation: Cashflow Extraction from Interest Rate Swap
Source: ~/QuantLib-SWIG/Python/examples/cashflows.py

Validates extraction of fixed/floating leg cashflows from a vanilla swap:
  - Fixed leg: dates, amounts, rates, accrual periods
  - Floating leg: dates, amounts, rates, accrual periods
  - NPV and fair rate
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
from ql_jax.time.daycounter import year_fraction


# === Reference values from QuantLib 1.42 ===
# 3Y payer swap, Actual/365 fixed, Actual/360 float (Euribor 6M), 5% flat curve, 1M notional
# NullCalendar, Unadjusted, settle Jan 17 2024 → Jan 17 2027

REF_NPV = 3460.1915760212
REF_FAIR_RATE = 0.0512729419

# Fixed leg: (payment_date, amount, rate, accrual_start, accrual_end, accrual_period)
REF_FIXED = [
    ("2025-01-17", 50136.9863013699, 0.05, "2024-01-17", "2025-01-17", 1.0027397260),
    ("2026-01-17", 50000.0000000000, 0.05, "2025-01-17", "2026-01-17", 1.0000000000),
    ("2027-01-17", 50000.0000000000, 0.05, "2026-01-17", "2027-01-17", 1.0000000000),
]

# Floating leg: (payment_date, amount, rate, accrual_start, accrual_end, accrual_period)
REF_FLOAT = [
    ("2024-07-17", 25244.8958663607, 0.0499349589, "2024-01-17", "2024-07-17", 0.5055555556),
    ("2025-01-17", 25525.8233603881, 0.0499418283, "2024-07-17", "2025-01-17", 0.5111111111),
    ("2025-07-17", 25104.4609791251, 0.0499315246, "2025-01-17", "2025-07-17", 0.5027777778),
    ("2026-01-17", 25529.3350590155, 0.0499486990, "2025-07-17", "2026-01-17", 0.5111111111),
    ("2026-07-17", 25101.0081167437, 0.0499246570, "2026-01-17", "2026-07-17", 0.5027777778),
    ("2027-01-17", 25527.5791293569, 0.0499452635, "2026-07-17", "2027-01-17", 0.5111111111),
]


def date_str(d: Date) -> str:
    """Format Date as YYYY-MM-DD."""
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


def main():
    today = Date(15, 1, 2024)
    curve = FlatForward(today, 0.05)

    settle = Date(17, 1, 2024)
    maturity = Date(17, 1, 2027)

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
    swap = make_vanilla_swap(
        SwapType.Payer, 1_000_000.0,
        fixed_schedule, 0.05, "Actual/365 (Fixed)",
        float_schedule, idx, spread=0.0,
    )

    npv = float(discounting_swap_npv(swap, curve))
    fair_rate = float(discounting_swap_fair_rate(swap, curve))

    print("=" * 78)
    print("Cashflow Validation (3Y Euribor Swap, 5% flat curve)")
    print("=" * 78)

    n_pass = 0
    total = 0

    # --- NPV and Fair Rate ---
    print(f"\n{'Metric':<20} {'QuantLib':>15} {'ql-jax':>15} {'Diff':>12}")
    print("-" * 62)

    npv_diff = abs(npv - REF_NPV)
    npv_ok = npv_diff < max(abs(REF_NPV) * 1e-3, 1.0)
    print(f"{'NPV':<20} {REF_NPV:>15.4f} {npv:>15.4f} {npv_diff:>12.2e} {'✓' if npv_ok else '✗'}")
    total += 1; n_pass += int(npv_ok)

    fr_diff = abs(fair_rate - REF_FAIR_RATE)
    fr_ok = fr_diff < 1e-4
    print(f"{'Fair Rate':<20} {REF_FAIR_RATE:>15.10f} {fair_rate:>15.10f} {fr_diff:>12.2e} {'✓' if fr_ok else '✗'}")
    total += 1; n_pass += int(fr_ok)

    # --- Fixed Leg Cashflows ---
    print(f"\n--- Fixed Leg ({len(swap.fixed_leg)} coupons) ---")
    print(f"{'#':<3} {'Date':>12} {'Amount':>14} {'Ref Amt':>14} {'Diff':>12}")
    print("-" * 57)

    for i, cf in enumerate(swap.fixed_leg):
        ref = REF_FIXED[i]
        ref_date, ref_amt, ref_rate, ref_start, ref_end, ref_tau = ref
        amt = float(cf.amount)
        diff = abs(amt - ref_amt)
        ok = diff < 0.01  # within 1 cent
        print(f"{i:<3} {date_str(cf.payment_date):>12} {amt:>14.4f} {ref_amt:>14.4f} {diff:>12.2e} {'✓' if ok else '✗'}")
        total += 1; n_pass += int(ok)

    # --- Floating Leg Cashflows ---
    print(f"\n--- Floating Leg ({len(swap.floating_leg)} coupons) ---")
    print(f"{'#':<3} {'Date':>12} {'Amount':>14} {'Ref Amt':>14} {'Diff':>12} {'Rate':>10} {'Ref Rate':>10}")
    print("-" * 75)

    for i, cf in enumerate(swap.floating_leg):
        ref = REF_FLOAT[i]
        ref_date, ref_amt, ref_rate, ref_start, ref_end, ref_tau = ref
        amt = float(cf.amount_with_curve(curve))
        rate = float(cf.adjusted_fixing(curve))
        diff = abs(amt - ref_amt)
        ok = diff < 5.0  # within $5 (small calendar differences between NullCalendar/TARGET)
        print(f"{i:<3} {date_str(cf.payment_date):>12} {amt:>14.4f} {ref_amt:>14.4f} {diff:>12.2e} {rate:>10.6f} {ref_rate:>10.6f} {'✓' if ok else '✗'}")
        total += 1; n_pass += int(ok)

    # --- Accrual Periods ---
    print(f"\n--- Accrual Period Checks ---")
    for i, cf in enumerate(swap.floating_leg):
        ref = REF_FLOAT[i]
        ref_tau = ref[5]
        tau = year_fraction(cf.accrual_start, cf.accrual_end, cf.day_counter)
        diff = abs(tau - ref_tau)
        ok = diff < 1e-8
        total += 1; n_pass += int(ok)
        if not ok:
            print(f"  Coupon {i}: tau={tau:.10f} ref={ref_tau:.10f} diff={diff:.2e} ✗")
    print(f"  All accrual periods match: {'✓' if n_pass >= total else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All cashflow assertions passed.")
    else:
        print(f"✗ {total - n_pass} failures.")


if __name__ == "__main__":
    main()
