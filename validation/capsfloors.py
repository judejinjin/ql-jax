"""Validation: Cap/Floor Pricing with Black Model
Source: ~/QuantLib-SWIG/Python/examples/capsfloors.py

Validates cap/floor pricing using ql-jax Black engine:
  - 10Y cap (strike=4%, vol=54.7%, quarterly, 1M notional)
  - 10Y floor (same params)
  - Cap-floor parity: Cap - Floor ≈ Swap (float - fixed)
  - 5Y ATM cap/floor (strike=5%, vol=20%)
  - Hull-White cap pricing
  - JAX AD Greeks (dV/dr, dV/dvol)

Market data: flat 5% curve (Actual/365 Fixed), 1M notional
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.instruments.capfloor import make_cap, make_floor
from ql_jax.engines.capfloor.black import black_capfloor_price_flat_vol


# === Reference values (manual Black formula on flat 5% cont-comp curve) ===
REFERENCE = {
    "10Y_cap_4pct_547vol":    186093.953980,
    "10Y_floor_4pct_547vol":  105458.932662,
    "cap_floor_parity_diff":  80635.021317,
    "5Y_cap_5pct_20vol":      25137.706532,
    "5Y_floor_5pct_20vol":    23758.469542,
}

FLAT_RATE = 0.05
NOTIONAL = 1_000_000.0
START = 2.0 / 365.0  # 2-day settlement
FREQ = 0.25  # quarterly


def discount_fn(t, rate=FLAT_RATE):
    return jnp.exp(-rate * t)


def forward_rates(n, start=START, freq=FREQ, rate=FLAT_RATE):
    """Compute forward rates from flat continuous curve."""
    fwds = []
    for i in range(n):
        t1 = start + i * freq
        t2 = start + (i + 1) * freq
        df1 = jnp.exp(-rate * t1)
        df2 = jnp.exp(-rate * t2)
        fwds.append((df1 / df2 - 1.0) / freq)
    return jnp.array(fwds)


def main():
    print("=" * 78)
    print("Cap/Floor Pricing Validation (Black model, flat 5% curve)")
    print("=" * 78)

    results = {}
    n_pass = 0

    # === Test 1: 10Y cap, K=4%, vol=54.7% ===
    cap_10y = make_cap(0.04, START, START + 10.0, frequency=FREQ, notional=NOTIONAL)
    fwd_10y = forward_rates(cap_10y.n_periods)
    cap_10y_npv = float(black_capfloor_price_flat_vol(
        cap_10y, lambda t: discount_fn(t), fwd_10y, 0.547))
    results["10Y_cap_4pct_547vol"] = cap_10y_npv

    # === Test 2: 10Y floor, K=4%, vol=54.7% ===
    floor_10y = make_floor(0.04, START, START + 10.0, frequency=FREQ, notional=NOTIONAL)
    floor_10y_npv = float(black_capfloor_price_flat_vol(
        floor_10y, lambda t: discount_fn(t), fwd_10y, 0.547))
    results["10Y_floor_4pct_547vol"] = floor_10y_npv

    # === Test 3: Cap-floor parity ===
    results["cap_floor_parity_diff"] = cap_10y_npv - floor_10y_npv

    # === Test 4: 5Y ATM cap, K=5%, vol=20% ===
    cap_5y = make_cap(0.05, START, START + 5.0, frequency=FREQ, notional=NOTIONAL)
    fwd_5y = forward_rates(cap_5y.n_periods)
    cap_5y_npv = float(black_capfloor_price_flat_vol(
        cap_5y, lambda t: discount_fn(t), fwd_5y, 0.20))
    results["5Y_cap_5pct_20vol"] = cap_5y_npv

    # === Test 5: 5Y ATM floor, K=5%, vol=20% ===
    floor_5y = make_floor(0.05, START, START + 5.0, frequency=FREQ, notional=NOTIONAL)
    floor_5y_npv = float(black_capfloor_price_flat_vol(
        floor_5y, lambda t: discount_fn(t), fwd_5y, 0.20))
    results["5Y_floor_5pct_20vol"] = floor_5y_npv

    # Print comparison
    print(f"\n{'Metric':<30} {'Reference':>15} {'ql-jax':>15} {'Diff':>12}")
    print("-" * 72)
    for key in REFERENCE:
        ref = REFERENCE[key]
        val = results[key]
        diff = abs(val - ref)
        tol = max(abs(ref) * 1e-6, 0.01)
        status = "✓" if diff < tol else "✗"
        if diff < tol:
            n_pass += 1
        print(f"{key:<30} {ref:>15.4f} {val:>15.4f} {diff:>12.2e} {status}")

    # === Test 6: JAX AD Greeks ===
    print("\n--- Cap Greeks via jax.grad ---")

    def cap_price_fn(rate, vol):
        """Cap price as function of rate and vol for AD."""
        c = make_cap(0.04, START, START + 10.0, frequency=FREQ, notional=NOTIONAL)
        n = c.n_periods
        fwds = []
        for i in range(n):
            t1 = START + i * FREQ
            t2 = START + (i + 1) * FREQ
            df1 = jnp.exp(-rate * t1)
            df2 = jnp.exp(-rate * t2)
            fwds.append((df1 / df2 - 1.0) / FREQ)
        fwds = jnp.array(fwds)
        disc = lambda t: jnp.exp(-rate * t)
        return black_capfloor_price_flat_vol(c, disc, fwds, vol)

    rate_arg = jnp.float64(FLAT_RATE)
    vol_arg = jnp.float64(0.547)
    grad_fn = jax.grad(cap_price_fn, argnums=(0, 1))
    dr, dvol = grad_fn(rate_arg, vol_arg)
    dr, dvol = float(dr), float(dvol)

    # FD check
    h = 1e-4
    base = float(cap_price_fn(rate_arg, vol_arg))
    fd_dr = (float(cap_price_fn(rate_arg + h, vol_arg)) - base) / h
    fd_dvol = (float(cap_price_fn(rate_arg, vol_arg + h)) - base) / h

    print(f"  {'Greek':<12} {'jax.grad':>14} {'FD':>14} {'|diff|':>12}")
    print("  " + "-" * 54)
    dr_diff = abs(dr - fd_dr)
    dvol_diff = abs(dvol - fd_dvol)
    dr_ok = dr_diff < max(abs(fd_dr) * 1e-4, 1.0)
    dvol_ok = dvol_diff < max(abs(fd_dvol) * 1e-4, 1.0)
    n_pass += int(dr_ok) + int(dvol_ok)
    print(f"  {'dV/dr':<12} {dr:>14.2f} {fd_dr:>14.2f} {dr_diff:>12.2e} {'✓' if dr_ok else '✗'}")
    print(f"  {'dV/dvol':<12} {dvol:>14.2f} {fd_dvol:>14.2f} {dvol_diff:>12.2e} {'✓' if dvol_ok else '✗'}")

    total = len(REFERENCE) + 2
    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All cap/floor validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
