"""Example 2: Bond Pricing and Yield Curve — classic QuantLib-style.

Demonstrates:
  - Flat yield curve construction
  - Fixed-rate bond pricing via discounting
  - AD sensitivity (dP/dr) via jax.grad
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.time.date import Date
from ql_jax.time.schedule import MakeSchedule
from ql_jax.time.calendar import NullCalendar
from ql_jax._util.types import Frequency, BusinessDayConvention
from ql_jax.termstructures.yield_.flat_forward import FlatForward
from ql_jax.instruments.bond import make_fixed_rate_bond
from ql_jax.engines.bond.discounting import (
    discounting_bond_clean_price,
    discounting_bond_dirty_price,
    discounting_bond_npv,
)


def main():
    print("=" * 60)
    print("QL-JAX Example: Bond Pricing")
    print("=" * 60)

    # ── Build a flat yield curve ─────────────────────────────
    today = Date(15, 1, 2025)
    rate = 0.045  # 4.5% flat curve
    curve = FlatForward(today, rate)

    print(f"  Yield curve:   Flat forward at {rate:.2%}")
    print(f"  Discount(1y):  {float(curve.discount(1.0)):.6f}")
    print(f"  Discount(5y):  {float(curve.discount(5.0)):.6f}")
    print(f"  Discount(10y): {float(curve.discount(10.0)):.6f}")
    print()

    # ── Fixed-rate bond ──────────────────────────────────────
    schedule = (MakeSchedule()
                .from_date(today)
                .to_date(Date(15, 1, 2035))
                .with_frequency(Frequency.Semiannual)
                .with_calendar(NullCalendar())
                .with_convention(BusinessDayConvention.Unadjusted)
                .build())

    bond = make_fixed_rate_bond(
        settlement_days=0,
        face_amount=100.0,
        schedule=schedule,
        coupons=0.05,
        day_counter="Actual/365 (Fixed)",
    )

    npv = float(discounting_bond_npv(bond, curve))
    clean = float(discounting_bond_clean_price(bond, curve))
    dirty = float(discounting_bond_dirty_price(bond, curve))

    print(f"  Bond: 5% coupon, 10y maturity, semi-annual")
    print(f"  NPV:           {npv:.6f}")
    print(f"  Clean price:   {clean:.6f}")
    print(f"  Dirty price:   {dirty:.6f}")
    print()

    # ── DV01 via AD ──────────────────────────────────────────
    # We compute dP/dr by building a simple functional version
    # that uses a flat discount curve's math directly.
    print("-" * 60)
    print("  DV01 via automatic differentiation")
    print("-" * 60)

    # Build a functional bond pricer using flat rate
    coupon_rate = 0.05
    freq = 2
    n_periods = 10 * freq
    tau = 1.0 / freq

    def bond_price_fn(r):
        """Clean-price approximation for flat curve at rate r."""
        total = 0.0
        for i in range(1, n_periods + 1):
            t = i * tau
            df = jnp.exp(-r * t)
            total = total + coupon_rate * tau * 100.0 * df
        total = total + 100.0 * jnp.exp(-r * n_periods * tau)
        return total

    dv01_fn = jax.grad(bond_price_fn)
    price_val = float(bond_price_fn(jnp.float64(rate)))
    dv01 = float(dv01_fn(jnp.float64(rate)))
    print(f"  Price (func):  {price_val:.6f}")
    print(f"  dP/dr (AD):    {dv01:.6f}")
    print(f"  DV01 (1bp):    {abs(dv01) * 0.0001:.6f}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
