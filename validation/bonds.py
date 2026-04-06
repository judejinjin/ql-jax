"""Validation: Bond Pricing
Source: ~/QuantLib-SWIG/Python/examples/bonds.py
       ~/QuantLib/Examples/Bonds/Bonds.cpp

Validates bond pricing on flat yield curves:
  - Par bond (5% coupon, 5% curve) ≈ par
  - Discount bond (3% coupon, 5% curve)
  - Premium bond (7% coupon, 5% curve)
  - Zero-coupon bond
  - Duration via JAX AD
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
from ql_jax.instruments.bond import make_fixed_rate_bond, make_zero_coupon_bond
from ql_jax.engines.bond.discounting import (
    discounting_bond_npv, discounting_bond_clean_price,
    discounting_bond_dirty_price,
)
from ql_jax.termstructures.yield_.flat_forward import FlatForward


# === Reference values from QuantLib 1.42 ===
# eval = Jan 15 2024, 5% flat curve, Actual365Fixed
REFERENCE = {
    "par_bond_clean":       99.4504822166,
    "discount_bond_clean":  90.8137870187,
    "premium_bond_clean":  108.0871774144,
    "zc_bond_npv":          77.8587442220,
}


def main():
    today = Date(15, 1, 2024)
    curve = FlatForward(today, 0.05)

    schedule = (MakeSchedule()
        .from_date(today)
        .to_date(Date(15, 1, 2029))
        .with_frequency(Frequency.Annual)
        .with_calendar(NullCalendar())
        .with_convention(BusinessDayConvention.Unadjusted)
        .build())

    # Par bond: 5% coupon
    par_bond = make_fixed_rate_bond(
        settlement_days=0, face_amount=100.0,
        schedule=schedule, coupons=0.05,
        day_counter="Actual/365 (Fixed)",
    )
    par_clean = float(discounting_bond_clean_price(par_bond, curve))

    # Discount bond: 3% coupon
    disc_bond = make_fixed_rate_bond(
        settlement_days=0, face_amount=100.0,
        schedule=schedule, coupons=0.03,
        day_counter="Actual/365 (Fixed)",
    )
    disc_clean = float(discounting_bond_clean_price(disc_bond, curve))

    # Premium bond: 7% coupon
    prem_bond = make_fixed_rate_bond(
        settlement_days=0, face_amount=100.0,
        schedule=schedule, coupons=0.07,
        day_counter="Actual/365 (Fixed)",
    )
    prem_clean = float(discounting_bond_clean_price(prem_bond, curve))

    # Zero-coupon bond
    zc_bond = make_zero_coupon_bond(
        settlement_days=0, calendar=NullCalendar(),
        face_amount=100.0, maturity_date=Date(15, 1, 2029),
    )
    zc_npv = float(discounting_bond_npv(zc_bond, curve))

    results = {
        "par_bond_clean":      par_clean,
        "discount_bond_clean": disc_clean,
        "premium_bond_clean":  prem_clean,
        "zc_bond_npv":         zc_npv,
    }

    # Print comparison
    print("=" * 78)
    print("Bond Pricing Validation (5% flat curve, Actual365Fixed)")
    print("=" * 78)
    print(f"\n{'Bond':<25} {'QuantLib':>15} {'ql-jax':>15} {'Diff':>12}")
    print("-" * 67)

    n_pass = 0
    for key in REFERENCE:
        ref = REFERENCE[key]
        val = results[key]
        diff = abs(val - ref)
        status = "✓" if diff < 0.01 else "✗"
        if diff < 0.01:
            n_pass += 1
        print(f"{key:<25} {ref:>15.6f} {val:>15.6f} {diff:>12.2e} {status}")

    # DV01 via JAX AD
    print("\n--- Bond DV01 via JAX AD ---")

    def bond_price_fn(rate):
        c = FlatForward(today, rate)
        return discounting_bond_clean_price(par_bond, c)

    dv01 = float(jax.grad(bond_price_fn)(jnp.float64(0.05)))
    print(f"  DV01 (dP/dr) = {dv01:.4f}")
    assert dv01 < 0, "DV01 should be negative (price falls with rising rates)"
    print(f"  ✓ DV01 < 0 (correct sign)")

    # Convexity: d²P/dr²
    convexity = float(jax.grad(jax.grad(bond_price_fn))(jnp.float64(0.05)))
    print(f"  Convexity (d²P/dr²) = {convexity:.4f}")
    assert convexity > 0, "Convexity should be positive"
    print(f"  ✓ Convexity > 0 (correct sign)")

    print(f"\nPassed: {n_pass}/{len(REFERENCE)}")

    # Assertions
    for key in REFERENCE:
        np.testing.assert_allclose(
            results[key], REFERENCE[key], atol=0.05,
            err_msg=f"{key} mismatch")

    print("✓ All bond price assertions passed.")


if __name__ == "__main__":
    main()
