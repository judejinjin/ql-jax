"""Validation: Gaussian1d / Hull-White Short-Rate Models
Source: ~/QuantLib/Examples/Gaussian1dModels/Gaussian1dModels.cpp

Validates Hull-White one-factor short-rate model:
  - Bond pricing: P(t,T) recovers market curve at r=r_0
  - Hull-White caplet/swaption pricing
  - JAX AD through model prices
  - Sensitivity to mean reversion and volatility

Market data: flat 5% curve, HW with a=0.1, sigma=0.01
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.models.shortrate.hull_white import (
    hull_white_bond_price,
    hull_white_swaption_price_jamshidian,
    hull_white_caplet_price,
)

FLAT_RATE = 0.05
A = 0.1
SIGMA = 0.01
R0 = FLAT_RATE

def flat_disc(t):
    return jnp.exp(-FLAT_RATE * t)

def main():
    print("=" * 78)
    print("Hull-White Short-Rate Model Validation")
    print("=" * 78)
    print(f"  Model: a={A}, sigma={SIGMA}  |  Curve: flat {FLAT_RATE*100:.0f}%")
    print()
    n_pass = 0
    total = 0

    # 1. Bond pricing recovery
    print("  --- Bond prices P(0,T) at r=r_0 ---")
    print(f"  {'T':>6} {'Market':>14} {'HW':>14} {'Diff':>12}")
    print("  " + "-" * 48)
    for T in [0.5, 1.0, 2.0, 5.0, 10.0]:
        total += 1
        ref = float(jnp.exp(-FLAT_RATE * T))
        hw = float(hull_white_bond_price(
            jnp.float64(R0), jnp.float64(A), jnp.float64(SIGMA),
            jnp.float64(0.0), jnp.float64(T), flat_disc))
        diff = abs(hw - ref)
        ok = diff < 1e-6
        n_pass += int(ok)
        print(f"  {T:>6.1f} {ref:>14.8f} {hw:>14.8f} {diff:>12.2e} {'✓' if ok else '✗'}")

    # 2. Bond monotonicity
    total += 1
    p_low = float(hull_white_bond_price(
        jnp.float64(0.03), jnp.float64(A), jnp.float64(SIGMA),
        jnp.float64(0.0), jnp.float64(5.0), flat_disc))
    p_high = float(hull_white_bond_price(
        jnp.float64(0.07), jnp.float64(A), jnp.float64(SIGMA),
        jnp.float64(0.0), jnp.float64(5.0), flat_disc))
    ok = p_low > p_high
    n_pass += int(ok)
    print(f"\n  Bond monotone: P(r=3%)={p_low:.6f} > P(r=7%)={p_high:.6f} {'✓' if ok else '✗'}")

    # 3. HW Caplet price
    caplet = float(hull_white_caplet_price(
        jnp.float64(R0), jnp.float64(A), jnp.float64(SIGMA),
        jnp.float64(FLAT_RATE), jnp.float64(1.0), jnp.float64(1.25),
        flat_disc, notional=1_000_000.0))
    print(f"\n  HW Caplet (K=5%, reset=1Y, pay=1.25Y): {caplet:.2f}")
    total += 1
    ok = caplet > 0
    n_pass += int(ok)
    print(f"  Price > 0: {'✓' if ok else '✗'}")

    # 4. HW Swaption (Jamshidian)
    swap_tenors = jnp.array([5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
    swpn = float(hull_white_swaption_price_jamshidian(
        jnp.float64(A), jnp.float64(SIGMA), flat_disc,
        jnp.float64(5.0), swap_tenors, jnp.float64(FLAT_RATE),
        notional=1_000_000.0))
    print(f"\n  HW Swaption (5Y into 5Y, semi): {swpn:,.2f}")
    total += 1
    ok = swpn > 0
    n_pass += int(ok)
    print(f"  Price > 0: {'✓' if ok else '✗'}")

    # 5. Vol sensitivity
    print(f"\n  --- Swaption vs HW sigma ---")
    vols = [0.005, 0.01, 0.015, 0.02, 0.03]
    prices_v = []
    for sig in vols:
        p = float(hull_white_swaption_price_jamshidian(
            jnp.float64(A), jnp.float64(sig), flat_disc,
            jnp.float64(5.0), swap_tenors, jnp.float64(FLAT_RATE),
            notional=1_000_000.0))
        prices_v.append(p)
        print(f"    sigma={sig:.3f}: {p:>12,.2f}")
    total += 1
    monotone = all(prices_v[i] <= prices_v[i+1] + 1.0 for i in range(len(vols)-1))
    n_pass += int(monotone)
    print(f"  Higher sigma -> higher swaption: {'✓' if monotone else '✗'}")

    # 6. JAX AD
    print(f"\n  --- JAX AD: HW swaption Greeks ---")
    def swpn_fn(a_p, sig_p):
        return hull_white_swaption_price_jamshidian(
            a_p, sig_p, flat_disc,
            jnp.float64(5.0), swap_tenors, jnp.float64(FLAT_RATE),
            notional=1_000_000.0)

    a_jax, sigma_jax = jnp.float64(A), jnp.float64(SIGMA)
    da, dsig = jax.grad(swpn_fn, argnums=(0, 1))(a_jax, sigma_jax)
    h = 1e-6
    base = float(swpn_fn(a_jax, sigma_jax))
    fd_da = (float(swpn_fn(a_jax + h, sigma_jax)) - base) / h
    fd_dsig = (float(swpn_fn(a_jax, sigma_jax + h)) - base) / h

    print(f"  {'Param':<8} {'jax.grad':>14} {'FD':>14} {'|diff|':>12}")
    print("  " + "-" * 50)
    for name, ad, fd in [("a", float(da), fd_da), ("sigma", float(dsig), fd_dsig)]:
        total += 1
        diff = abs(ad - fd)
        ok = diff < max(abs(fd) * 1e-3, 1.0)
        n_pass += int(ok)
        print(f"  {name:<8} {ad:>14.2f} {fd:>14.2f} {diff:>12.2e} {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All Hull-White model validations passed.")
    else:
        print("✗ Some tests failed.")

if __name__ == "__main__":
    main()
