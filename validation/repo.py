"""Validation: Repo — Bond Forward Pricing
Source: ~/QuantLib/Examples/Bonds/ (BondForward)
       ~/QuantLib/ql/instruments/bondforward.hpp

Validates bond forward (repo) pricing:
  1. Zero-coupon forward = spot compounded at repo rate
  2. Coupon bond: spot income reduces forward price
  3. Clean = dirty - accrued at delivery
  4. AD Greeks: d(fwd)/d(repo_rate), d(fwd)/d(yield)
  5. Batch pricing: vmap across repo rates
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
    discounting_bond_npv,
    discounting_bond_dirty_price,
    bond_forward_dirty_price,
    bond_forward_clean_price,
    bond_forward_npv,
    bond_forward_spot_income,
)
from ql_jax.termstructures.yield_.flat_forward import FlatForward
from ql_jax.cashflows.analytics import accrued_amount


HEADER = """
==============================================================================
Repo — Bond Forward Pricing
=============================================================================="""


def main():
    print(HEADER)
    n_pass, n_total = 0, 5

    today = Date(15, 1, 2024)
    delivery_date = Date(15, 7, 2024)  # 6 months forward

    # Curves
    repo_rate = 0.04
    bond_yield = 0.05
    repo_curve = FlatForward(today, repo_rate)
    bond_curve = FlatForward(today, bond_yield)

    # ===================================================================
    # Test 1: Zero-coupon bond forward = spot * exp(r * T)
    # ===================================================================
    print("\n  Test 1: Zero-coupon bond forward = spot compounded")
    zc_bond = make_zero_coupon_bond(
        settlement_days=0, face_amount=100.0,
        maturity_date=Date(15, 1, 2029),
        calendar=NullCalendar(),
    )
    dirty_spot = float(discounting_bond_npv(zc_bond, bond_curve, today))
    dirty_fwd = float(bond_forward_dirty_price(
        zc_bond, delivery_date, repo_curve,
        bond_curve=bond_curve, settlement_date=today,
    ))
    T_delivery = repo_curve.time_from_reference(delivery_date)
    # For zero-coupon bond, no intermediate coupons:
    # fwd = spot / DF_repo(delivery)
    expected_fwd = dirty_spot / float(repo_curve.discount(T_delivery))
    err = abs(dirty_fwd - expected_fwd)
    ok = err < 1e-10
    n_pass += int(ok)
    print(f"    Dirty spot:     {dirty_spot:.6f}")
    print(f"    Dirty forward:  {dirty_fwd:.6f}")
    print(f"    Expected:       {expected_fwd:.6f}")
    print(f"    {'✓' if ok else '✗'} Error: {err:.2e}")

    # ===================================================================
    # Test 2: Coupon bond — spot income reduces forward price
    # ===================================================================
    print("\n  Test 2: Coupon bond — spot income reduces forward price")
    # Bond pays a coupon in April 2024, between settlement and delivery
    schedule = (MakeSchedule()
        .from_date(Date(15, 1, 2023))
        .to_date(Date(15, 1, 2029))
        .with_frequency(Frequency.Semiannual)
        .with_calendar(NullCalendar())
        .with_convention(BusinessDayConvention.Unadjusted)
        .build())
    coupon_bond = make_fixed_rate_bond(
        settlement_days=0, face_amount=100.0,
        schedule=schedule, coupons=0.06,
        day_counter="Actual/365 (Fixed)",
    )
    dirty_spot_cpn = float(discounting_bond_npv(coupon_bond, bond_curve, today))
    dirty_fwd_cpn = float(bond_forward_dirty_price(
        coupon_bond, delivery_date, repo_curve,
        bond_curve=bond_curve, settlement_date=today,
    ))
    spot_income = float(bond_forward_spot_income(
        coupon_bond, today, delivery_date, repo_curve,
    ))
    # Verify forward formula: fwd = (spot - income) / DF_repo
    df_del = float(repo_curve.discount(T_delivery))
    manual_fwd = (dirty_spot_cpn - spot_income) / df_del
    err_formula = abs(dirty_fwd_cpn - manual_fwd)
    # Income should be positive (coupon falls in period)
    ok = spot_income > 0.1 and err_formula < 1e-10
    n_pass += int(ok)
    print(f"    Dirty spot:     {dirty_spot_cpn:.6f}")
    print(f"    Spot income:    {spot_income:.6f}")
    print(f"    Dirty forward:  {dirty_fwd_cpn:.6f}")
    print(f"    Manual forward: {manual_fwd:.6f}")
    print(f"    {'✓' if ok else '✗'} Income>0: {spot_income:.4f}, formula err: {err_formula:.2e}")

    # ===================================================================
    # Test 3: Clean = dirty - accrued at delivery
    # ===================================================================
    print("\n  Test 3: Clean = dirty - AI(delivery)")
    clean_fwd = float(bond_forward_clean_price(
        coupon_bond, delivery_date, repo_curve,
        bond_curve=bond_curve, settlement_date=today,
    ))
    ai_delivery = float(accrued_amount(coupon_bond.cashflows, delivery_date))
    expected_clean = dirty_fwd_cpn - ai_delivery
    err_clean = abs(clean_fwd - expected_clean)
    ok = err_clean < 1e-10
    n_pass += int(ok)
    print(f"    Dirty forward:  {dirty_fwd_cpn:.6f}")
    print(f"    AI at delivery: {ai_delivery:.6f}")
    print(f"    Clean forward:  {clean_fwd:.6f}")
    print(f"    Expected:       {expected_clean:.6f}")
    print(f"    {'✓' if ok else '✗'} Error: {err_clean:.2e}")

    # ===================================================================
    # Test 4: AD Greeks — d(fwd)/d(repo_rate)
    # ===================================================================
    print("\n  Test 4: AD Greeks via jax.grad")

    def fwd_price_of_rate(r):
        c = FlatForward(today, r)
        return bond_forward_dirty_price(
            coupon_bond, delivery_date, c,
            bond_curve=bond_curve, settlement_date=today,
        )

    ad_delta = float(jax.grad(fwd_price_of_rate)(jnp.float64(repo_rate)))
    # FD check
    dr = 1e-5
    fwd_up = float(fwd_price_of_rate(jnp.float64(repo_rate + dr)))
    fwd_dn = float(fwd_price_of_rate(jnp.float64(repo_rate - dr)))
    fd_delta = (fwd_up - fwd_dn) / (2 * dr)
    rel_err = abs(ad_delta - fd_delta) / (abs(fd_delta) + 1e-15)
    ok = rel_err < 1e-4
    n_pass += int(ok)
    print(f"    AD d(fwd)/d(r): {ad_delta:.6f}")
    print(f"    FD d(fwd)/d(r): {fd_delta:.6f}")
    print(f"    {'✓' if ok else '✗'} Rel error: {rel_err:.2e}")

    # ===================================================================
    # Test 5: Batch pricing — vmap across repo rates
    # ===================================================================
    print("\n  Test 5: vmap batch across repo rates")
    rates = jnp.linspace(0.01, 0.08, 8)

    def fwd_for_rate(r):
        c = FlatForward(today, r)
        return bond_forward_dirty_price(
            coupon_bond, delivery_date, c,
            bond_curve=bond_curve, settlement_date=today,
        )

    batch_fwd = jax.vmap(fwd_for_rate)(rates)
    batch_fwd_np = np.array(batch_fwd)
    # Forward price should increase with repo rate (higher financing cost)
    diffs = np.diff(batch_fwd_np)
    monotone = np.all(diffs > 0)
    ok = bool(monotone) and len(batch_fwd_np) == 8
    n_pass += int(ok)
    print(f"    Rates:    {np.array(rates)}")
    print(f"    Forwards: {batch_fwd_np}")
    print(f"    {'✓' if ok else '✗'} Monotone increasing: {monotone}, count={len(batch_fwd_np)}")

    # === Summary ===
    print(f"\n  Passed: {n_pass}/{n_total}")
    if n_pass == n_total:
        print("  ✓ All repo (bond forward) validations passed.")
    else:
        print(f"  ✗ {n_total - n_pass} test(s) failed.")
    return n_pass == n_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
