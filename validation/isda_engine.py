"""Validation: ISDA CDS Engine
Source: ~/QuantLib/Examples/CDS/

ISDA-standard CDS pricing: NPV, fair spread, recovery sensitivity, AD Greeks.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.credit.isda import isda_cds_npv, isda_cds_fair_spread


def main():
    print("=" * 78)
    print("ISDA CDS Engine Validation")
    print("=" * 78)

    # Setup: 5Y CDS, quarterly payments
    notional = 10_000_000.0
    spread = 0.01  # 100 bps
    recovery = 0.40

    # Payment dates: quarterly for 5Y = 20 periods
    n_periods = 20
    payment_dates = jnp.linspace(0.25, 5.0, n_periods)
    day_fractions = jnp.full(n_periods, 0.25)

    # Flat discount curve: 3% cont. compounded
    r = 0.03
    disc_fn = lambda t: jnp.exp(-r * t)

    # Flat hazard rate: λ s.t. fair spread ≈ 100bps
    hazard = 0.0167  # ≈ spread / (1 - recovery)
    surv_fn = lambda t: jnp.exp(-hazard * t)

    print(f"  Notional: {notional:,.0f}")
    print(f"  Spread: {spread*10000:.0f} bps, Recovery: {recovery:.0%}")
    print(f"  Discount rate: {r:.0%}, Hazard rate: {hazard:.4f}")

    # === Test 1: NPV computation ===
    npv_buyer = float(isda_cds_npv(
        spread, recovery, payment_dates, day_fractions,
        disc_fn, surv_fn, notional, is_buyer=True, accrual_on_default=True))
    npv_seller = float(isda_cds_npv(
        spread, recovery, payment_dates, day_fractions,
        disc_fn, surv_fn, notional, is_buyer=False, accrual_on_default=True))
    print(f"\n  NPV (buyer):  {npv_buyer:,.2f}")
    print(f"  NPV (seller): {npv_seller:,.2f}")
    print(f"  Sum:          {npv_buyer + npv_seller:,.2f}")

    # === Test 2: Fair spread ===
    fair = float(isda_cds_fair_spread(
        recovery, payment_dates, day_fractions,
        disc_fn, surv_fn, accrual_on_default=True))
    print(f"\n  Fair spread: {fair*10000:.2f} bps")

    # At fair spread, NPV should be ~0
    npv_at_fair = float(isda_cds_npv(
        fair, recovery, payment_dates, day_fractions,
        disc_fn, surv_fn, notional, is_buyer=True, accrual_on_default=True))
    print(f"  NPV at fair spread: {npv_at_fair:,.2f}")

    # === Test 3: Recovery sensitivity ===
    recoveries = [0.20, 0.30, 0.40, 0.50, 0.60]
    fair_spreads = []
    for rec in recoveries:
        fs = float(isda_cds_fair_spread(
            rec, payment_dates, day_fractions,
            disc_fn, surv_fn, accrual_on_default=True))
        fair_spreads.append(fs)
        print(f"  Recovery={rec:.0%}: fair spread={fs*10000:.2f} bps")
    # Higher recovery → lower fair spread
    recovery_monotone = all(fair_spreads[i] >= fair_spreads[i+1]
                           for i in range(len(fair_spreads)-1))

    # === Test 4: JAX AD Greeks ===
    def cds_price_fn(params):
        sp, rec, hz, rate = params
        df = lambda t: jnp.exp(-rate * t)
        sf = lambda t: jnp.exp(-hz * t)
        return isda_cds_npv(
            sp, rec, payment_dates, day_fractions,
            df, sf, notional, is_buyer=True, accrual_on_default=True)

    base = jnp.array([spread, recovery, hazard, r])
    greeks = np.array(jax.grad(cds_price_fn)(base))
    names = ['spread', 'recovery', 'hazard', 'rate']
    print(f"\n  --- AD Greeks ---")
    for name, g in zip(names, greeks):
        print(f"    d(NPV)/d({name:>8s}): {g:>14,.2f}")

    # FD check
    H = 1e-6
    fd = np.zeros(4)
    for i in range(4):
        xp = base.at[i].add(H)
        xm = base.at[i].add(-H)
        fd[i] = (float(cds_price_fn(xp)) - float(cds_price_fn(xm))) / (2 * H)
    max_rel = np.max(np.abs(greeks - fd) / np.maximum(np.abs(fd), 1.0))
    print(f"  AD vs FD max rel diff: {max_rel:.2e}")

    # === Validation ===
    n_pass = 0
    total = 5

    ok = abs(npv_buyer + npv_seller) < notional * 1e-10
    n_pass += int(ok)
    print(f"\n  Buyer+Seller=0:     {'✓' if ok else '✗'}")

    ok = abs(npv_at_fair) < notional * 1e-4
    n_pass += int(ok)
    print(f"  NPV@fair ≈ 0:       {'✓' if ok else '✗'}")

    ok = abs(fair - spread) < 0.005  # fair spread near 100bps
    n_pass += int(ok)
    print(f"  Fair spread ok:     {'✓' if ok else '✗'} ({fair*10000:.2f} bps)")

    ok = recovery_monotone
    n_pass += int(ok)
    print(f"  Recovery monotone:  {'✓' if ok else '✗'}")

    ok = max_rel < 1e-4
    n_pass += int(ok)
    print(f"  AD vs FD:           {'✓' if ok else '✗'} ({max_rel:.2e})")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All ISDA CDS engine validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
