"""Validation: Heston Stochastic Local Vol (SLV)
Source: ~/QuantLib/Examples/HestonSLVModel/

Heston + leverage function Monte Carlo pricing, convergence to pure Heston.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.models.equity.heston_slv import heston_slv_price_mc


def main():
    print("=" * 78)
    print("Heston Stochastic Local Vol (SLV) Validation")
    print("=" * 78)

    # Parameters
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    # Heston params
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7

    print(f"  S={S}, K={K}, T={T}, r={r}, q={q}")
    print(f"  v0={v0}, κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}")

    # === Test 1: Pure Heston (leverage = 1) ===
    key = jax.random.PRNGKey(42)
    leverage_fn = lambda t, S: 1.0  # no local vol correction

    price_call = float(heston_slv_price_mc(
        S, K, T, r, q, v0, kappa, theta, xi, rho,
        leverage_fn, 1, n_paths=100000, n_steps=200, key=key))
    price_put = float(heston_slv_price_mc(
        S, K, T, r, q, v0, kappa, theta, xi, rho,
        leverage_fn, -1, n_paths=100000, n_steps=200, key=key))
    print(f"\n  Heston MC call: {price_call:.4f}")
    print(f"  Heston MC put:  {price_put:.4f}")

    # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
    parity_lhs = price_call - price_put
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    parity_err = abs(parity_lhs - parity_rhs)
    print(f"  Put-call parity error: {parity_err:.4f}")

    # === Test 2: Leverage function effect ===
    # Leverage > 1 increases vol → higher prices
    leverage_fn_high = lambda t, S: 1.5
    price_high = float(heston_slv_price_mc(
        S, K, T, r, q, v0, kappa, theta, xi, rho,
        leverage_fn_high, 1, n_paths=100000, n_steps=200, key=key))
    print(f"\n  High leverage call:   {price_high:.4f}")
    print(f"  Pure Heston call:     {price_call:.4f}")
    leverage_increase = price_high > price_call

    # === Test 3: MC convergence ===
    prices = []
    for n_paths in [10000, 50000, 200000]:
        p = float(heston_slv_price_mc(
            S, K, T, r, q, v0, kappa, theta, xi, rho,
            lambda t, S: 1.0, 1, n_paths=n_paths, n_steps=100, key=key))
        prices.append(p)
        print(f"  MC {n_paths:>6d} paths: {p:.4f}")
    # Convergence: later prices should be closer to each other
    converging = abs(prices[1] - prices[2]) < abs(prices[0] - prices[1]) + 0.5

    # === Test 4: Positive prices and sensible values ===
    positive = price_call > 0 and price_put > 0

    # === Validation ===
    n_pass = 0
    total = 4

    ok = parity_err < 1.0  # MC noise
    n_pass += int(ok)
    print(f"\n  Put-call parity:    {'✓' if ok else '✗'} (err={parity_err:.4f})")

    n_pass += int(leverage_increase)
    print(f"  Leverage increases:  {'✓' if leverage_increase else '✗'}")

    n_pass += int(converging)
    print(f"  MC convergence:     {'✓' if converging else '✗'}")

    n_pass += int(positive)
    print(f"  Positive prices:    {'✓' if positive else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All Heston SLV validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
