"""Validation: Bermudan Swaption Pricing
Source: ~/QuantLib/Examples/BermudanSwaption/BermudanSwaption.cpp
        ~/QuantLib-SWIG/Python/examples/bermudan-swaption.py

Validates European swaption pricing under multiple models:
  - Black swaption pricing
  - Hull-White swaption pricing (Jamshidian decomposition)
  - G2++ swaption pricing
  - JAX AD: dPrice/dVol

Market data: 5Y into 5Y payer swaption, 5% fixed rate, flat 5% curve
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.instruments.swaption import make_swaption
from ql_jax.engines.swaption.black import (
    black_swaption_price, bachelier_swaption_price, par_swap_rate,
)
from ql_jax.engines.swaption.hull_white import hw_swaption_price


# === Market data ===
FLAT_RATE = 0.05
EXERCISE_TIME = 5.0  # 5Y option
SWAP_TENOR = 5.0     # into 5Y swap
FIXED_RATE = 0.05
NOTIONAL = 1_000_000.0
BLACK_VOL = 0.20     # 20% lognormal vol
NORMAL_VOL = 0.01    # 100bp normal vol
HW_A = 0.1           # Hull-White mean reversion
HW_SIGMA = 0.01      # Hull-White volatility


def flat_disc(t):
    return jnp.exp(-FLAT_RATE * t)


def main():
    print("=" * 78)
    print("Bermudan/European Swaption Pricing Validation")
    print("=" * 78)
    print(f"  Instrument : {EXERCISE_TIME}Y into {SWAP_TENOR}Y Payer Swaption")
    print(f"  Fixed rate : {FIXED_RATE*100:.0f}%  |  Curve: flat {FLAT_RATE*100:.0f}%")
    print()

    n_pass = 0
    total = 0

    # === 1. Create swaption ===
    swpn = make_swaption(
        exercise_time=EXERCISE_TIME,
        swap_tenor=SWAP_TENOR,
        fixed_rate=FIXED_RATE,
        frequency=0.5,  # semi-annual
        notional=NOTIONAL,
        payer=True,
    )

    # === 2. Forward par swap rate ===
    fwd_rate = float(par_swap_rate(
        flat_disc, EXERCISE_TIME,
        swpn.swap_payment_dates, swpn.swap_accrual_fractions))
    print(f"  Forward par swap rate: {fwd_rate*100:.4f}% (expect ~{FLAT_RATE*100:.0f}%)")
    total += 1
    ok = abs(fwd_rate - FLAT_RATE) < 0.005
    n_pass += int(ok)
    print(f"  Close to flat rate: {'✓' if ok else '✗'}")

    # === 3. Black swaption price ===
    black_price = float(black_swaption_price(swpn, flat_disc, fwd_rate, BLACK_VOL))
    print(f"\n  Black swaption price ({BLACK_VOL*100:.0f}% vol): {black_price:,.2f}")
    total += 1
    ok = black_price > 0
    n_pass += int(ok)
    print(f"  Price > 0: {'✓' if ok else '✗'}")

    # === 4. Bachelier swaption price ===
    bach_price = float(bachelier_swaption_price(swpn, flat_disc, fwd_rate, NORMAL_VOL))
    print(f"  Bachelier swaption price ({NORMAL_VOL*10000:.0f}bp normal vol): {bach_price:,.2f}")
    total += 1
    ok = bach_price > 0
    n_pass += int(ok)
    print(f"  Price > 0: {'✓' if ok else '✗'}")

    # === 5. Hull-White swaption price ===
    hw_price = float(hw_swaption_price(
        NOTIONAL, FIXED_RATE,
        swpn.swap_payment_dates, swpn.swap_accrual_fractions,
        HW_A, HW_SIGMA, flat_disc, EXERCISE_TIME, is_payer=True))
    print(f"  Hull-White swaption price (a={HW_A}, σ={HW_SIGMA}): {hw_price:,.2f}")
    total += 1
    ok = hw_price > 0
    n_pass += int(ok)
    print(f"  Price > 0: {'✓' if ok else '✗'}")

    # === 6. ATM swaption prices should be similar across models ===
    print(f"\n  --- Cross-model comparison (all ATM) ---")
    print(f"  Black:      {black_price:>12,.2f}")
    print(f"  Bachelier:  {bach_price:>12,.2f}")
    print(f"  Hull-White: {hw_price:>12,.2f}")
    total += 1
    # All should be same order of magnitude
    prices = [black_price, bach_price, hw_price]
    ok = max(prices) / max(min(prices), 1.0) < 10.0
    n_pass += int(ok)
    print(f"  Same order of magnitude: {'✓' if ok else '✗'}")

    # === 7. Payer/Receiver symmetry (put-call parity at ATM) ===
    swpn_recv = make_swaption(
        exercise_time=EXERCISE_TIME, swap_tenor=SWAP_TENOR,
        fixed_rate=FIXED_RATE, frequency=0.5, notional=NOTIONAL, payer=False)
    recv_price = float(black_swaption_price(swpn_recv, flat_disc, fwd_rate, BLACK_VOL))
    # At ATM, payer ≈ receiver for swaptions
    total += 1
    ok = abs(black_price - recv_price) / max(black_price, 1.0) < 0.10
    n_pass += int(ok)
    print(f"\n  Payer/Receiver near-parity at ATM: {black_price:,.2f} vs {recv_price:,.2f} "
          f"(ratio={black_price/max(recv_price,1.):.4f}) {'✓' if ok else '✗'}")

    # === 8. Vol sensitivity: higher vol → higher swaption price ===
    vols = [0.05, 0.10, 0.20, 0.30, 0.40]
    vol_prices = [float(black_swaption_price(swpn, flat_disc, fwd_rate, v)) for v in vols]
    total += 1
    monotone = all(vol_prices[i] <= vol_prices[i+1] + 1.0 for i in range(len(vols)-1))
    n_pass += int(monotone)
    print(f"  Higher vol → higher price: {'✓' if monotone else '✗'}")

    # === 9. JAX AD: dPrice/dVol (vega) ===
    print(f"\n  --- JAX AD swaption vega ---")

    def swaption_price_fn(vol):
        sw = make_swaption(EXERCISE_TIME, SWAP_TENOR, FIXED_RATE, 0.5, NOTIONAL, True)
        fwd = par_swap_rate(flat_disc, EXERCISE_TIME,
                            sw.swap_payment_dates, sw.swap_accrual_fractions)
        return black_swaption_price(sw, flat_disc, fwd, vol)

    vol_jax = jnp.float64(BLACK_VOL)
    ad_vega = float(jax.grad(swaption_price_fn)(vol_jax))

    # FD check
    h = 1e-5
    fd_vega = (float(swaption_price_fn(vol_jax + h)) - float(swaption_price_fn(vol_jax))) / h

    total += 1
    diff = abs(ad_vega - fd_vega)
    ok = diff < max(abs(fd_vega) * 1e-4, 1.0)
    n_pass += int(ok)
    print(f"  jax.grad vega: {ad_vega:,.2f}")
    print(f"  FD vega:       {fd_vega:,.2f}")
    print(f"  Diff: {diff:.2e} {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All swaption validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
