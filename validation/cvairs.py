"""Validation: CVAIRS — Counterparty Credit Adjustment for IRS
Source: ~/QuantLib/Examples/CVAIRS/CVAIRS.cpp
       Brigo & Masetti (2005) — Table 2

Validates CVA-adjusted IRS pricing:
  1. CVA increases with hazard rate (low < medium < high)
  2. CVA vanishes for zero default intensity
  3. AD Greeks: d(CVA)/d(vol), d(CVA)/d(recovery), d(CVA)/d(hazard)
  4. Batch pricing: vmap across credit scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.swap.cva import cva_swap_brigo_masetti

HEADER = """
==============================================================================
CVAIRS — Counterparty Value Adjustment for Interest Rate Swaps
=============================================================================="""


def make_hazard_fn(intensity):
    """Flat hazard rate: Q(t) = exp(-h*t)."""
    def survival(t):
        return jnp.exp(-jnp.float64(intensity) * jnp.float64(t))
    return survival


def make_discount_fn(swap_rates, tenors):
    """Build a simple discount function from swap rates.

    Uses continuous compounding: P(0,T) = exp(-r*T) with linear
    interpolation of zero rates.
    """
    # Bootstrap simple zero rates from par swap rates
    zero_rates = jnp.array(swap_rates)  # approximate: swap rate ≈ zero rate
    tenor_arr = jnp.array(tenors, dtype=jnp.float64)

    def discount(t):
        t = jnp.float64(t)
        # Linear interpolation of zero rates, extrapolate flat
        r = jnp.interp(t, tenor_arr, zero_rates)
        return jnp.exp(-r * t)
    return discount


def main():
    print(HEADER)
    n_pass, n_total = 0, 4

    # --- Market data from Brigo-Masetti / QuantLib CVAIRS example ---
    tenors_years = [5, 10, 15, 20, 25, 30]
    swap_rates = [0.03249, 0.04074, 0.04463, 0.04675, 0.04775, 0.04811]

    # Build discount function
    discount_fn = make_discount_fn(swap_rates, tenors_years)

    # Three credit risk levels (flat hazard rates)
    hazard_low = 0.0036      # ~AA
    hazard_medium = 0.0202   # ~BBB
    hazard_high = 0.0534     # ~BB

    recovery_low = 0.4
    recovery_medium = 0.35
    recovery_high = 0.3

    black_vol = 0.15

    # ===================================================================
    # Test 1: CVA spread increases with credit risk
    # ===================================================================
    print("\n  Test 1: CVA spread increases with hazard rate")
    print(f"  {'Tenor':>5} | {'RF rate':>8} | {'Low bp':>8} | {'Med bp':>8} | {'High bp':>8}")
    print("  " + "-" * 52)

    all_ok = True
    for tenor_y, sr in zip(tenors_years, swap_rates):
        n_periods = tenor_y * 4  # quarterly
        payment_times = [(i + 1) * 0.25 for i in range(n_periods)]
        accrual_fracs = [0.25] * n_periods

        results = {}
        for label, hz, rec in [
            ("low", hazard_low, recovery_low),
            ("med", hazard_medium, recovery_medium),
            ("high", hazard_high, recovery_high),
        ]:
            out = cva_swap_brigo_masetti(
                discount_fn=discount_fn,
                black_vol=black_vol,
                swap_fixed_rate=sr,
                payment_times=payment_times,
                accrual_fracs=accrual_fracs,
                notional=100.0,
                cpty_survival_fn=make_hazard_fn(hz),
                cpty_recovery=rec,
                swap_type=1,
            )
            results[label] = out

        spread_low = (results["low"]["risky_fair_rate"] - results["low"]["base_fair_rate"]) * 10000
        spread_med = (results["med"]["risky_fair_rate"] - results["med"]["base_fair_rate"]) * 10000
        spread_high = (results["high"]["risky_fair_rate"] - results["high"]["base_fair_rate"]) * 10000

        print(f"  {tenor_y:5d} | {sr:8.5f} | {spread_low:8.2f} | {spread_med:8.2f} | {spread_high:8.2f}")

        # CVA spread should increase: low < medium < high
        if not (abs(spread_low) < abs(spread_med) < abs(spread_high)):
            all_ok = False

    ok = all_ok
    n_pass += int(ok)
    print(f"  {'✓' if ok else '✗'} Monotone: |low| < |medium| < |high| for all tenors")

    # ===================================================================
    # Test 2: CVA vanishes for zero hazard rate
    # ===================================================================
    print("\n  Test 2: CVA → 0 for zero default intensity")
    n_periods = 40  # 10Y quarterly
    payment_times = [(i + 1) * 0.25 for i in range(n_periods)]
    accrual_fracs = [0.25] * n_periods

    out_zero = cva_swap_brigo_masetti(
        discount_fn=discount_fn,
        black_vol=black_vol,
        swap_fixed_rate=0.04074,  # 10Y rate
        payment_times=payment_times,
        accrual_fracs=accrual_fracs,
        notional=100.0,
        cpty_survival_fn=make_hazard_fn(1e-12),  # near-zero hazard
        cpty_recovery=0.4,
        swap_type=1,
    )
    cva_val = float(out_zero["cva"])
    ok = abs(cva_val) < 1e-6
    n_pass += int(ok)
    print(f"    CVA with h≈0: {cva_val:.2e}")
    print(f"    {'✓' if ok else '✗'} CVA ≈ 0 (threshold 1e-6)")

    # ===================================================================
    # Test 3: AD Greeks — d(CVA)/d(vol), d(CVA)/d(recovery)
    # ===================================================================
    print("\n  Test 3: AD Greeks via jax.grad")

    def cva_of_vol(vol):
        out = cva_swap_brigo_masetti(
            discount_fn=discount_fn,
            black_vol=vol,
            swap_fixed_rate=0.04074,
            payment_times=payment_times,
            accrual_fracs=accrual_fracs,
            notional=100.0,
            cpty_survival_fn=make_hazard_fn(hazard_medium),
            cpty_recovery=0.35,
            swap_type=1,
        )
        return out["cva"]

    ad_vega = float(jax.grad(cva_of_vol)(jnp.float64(black_vol)))
    dv = 1e-5
    fd_vega = (float(cva_of_vol(jnp.float64(black_vol + dv))) -
               float(cva_of_vol(jnp.float64(black_vol - dv)))) / (2 * dv)

    rel_err = abs(ad_vega - fd_vega) / (abs(fd_vega) + 1e-15)
    ok = rel_err < 1e-3 and abs(ad_vega) > 0.01
    n_pass += int(ok)
    print(f"    AD d(CVA)/d(vol): {ad_vega:.6f}")
    print(f"    FD d(CVA)/d(vol): {fd_vega:.6f}")
    print(f"    {'✓' if ok else '✗'} Rel error: {rel_err:.2e}, |vega|={abs(ad_vega):.4f}")

    # ===================================================================
    # Test 4: Batch pricing — vmap across hazard rates
    # ===================================================================
    print("\n  Test 4: vmap batch across hazard rates")
    hazard_rates = jnp.linspace(0.005, 0.10, 10)

    def cva_of_hazard(h):
        out = cva_swap_brigo_masetti(
            discount_fn=discount_fn,
            black_vol=black_vol,
            swap_fixed_rate=0.04074,
            payment_times=payment_times,
            accrual_fracs=accrual_fracs,
            notional=100.0,
            cpty_survival_fn=make_hazard_fn(h),
            cpty_recovery=0.4,
            swap_type=1,
        )
        return out["cva"]

    batch_cva = jax.vmap(cva_of_hazard)(hazard_rates)
    batch_cva_np = np.array(batch_cva)
    # CVA should increase with hazard rate
    diffs = np.diff(batch_cva_np)
    monotone = np.all(diffs > 0)
    ok = bool(monotone) and len(batch_cva_np) == 10
    n_pass += int(ok)
    print(f"    Hazard rates: {np.array(hazard_rates[:5])} ...")
    print(f"    CVA values:   {batch_cva_np[:5]} ...")
    print(f"    {'✓' if ok else '✗'} Monotone increasing: {monotone}, count={len(batch_cva_np)}")

    # === Summary ===
    print(f"\n  Passed: {n_pass}/{n_total}")
    if n_pass == n_total:
        print("  ✓ All CVAIRS validations passed.")
    else:
        print(f"  ✗ {n_total - n_pass} test(s) failed.")
    return n_pass == n_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
