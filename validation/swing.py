"""Validation: Swing Option (FD)
Source: ~/QuantLib/Examples/SwingOption/

Swing option pricing via FD, Greeks via AD, exercise-count sensitivity.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.engines.fd.swing import fd_swing_price


def main():
    print("=" * 78)
    print("Swing Option (FD) Validation")
    print("=" * 78)

    # Parameters
    S0 = 50.0
    K = 50.0
    T = 1.0
    r = 0.06
    q = 0.02
    sigma = 0.30

    print(f"  S={S0}, K={K}, T={T}, r={r}, q={q}, σ={sigma}")

    # === Test 1: Swing price vs European (n_exercises=1) ===
    price_1 = float(fd_swing_price(S0, K, T, r, q, sigma, n_exercises=1,
                                    n_x=80, n_t=80))
    print(f"\n  Swing (1 exercise): {price_1:.4f}")

    # For comparison: BS European call
    from jax.scipy.stats import norm
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_call = S0 * np.exp(-q * T) * float(norm.cdf(d1)) - K * np.exp(-r * T) * float(norm.cdf(d2))
    print(f"  BS European call:   {bs_call:.4f}")

    # === Test 2: More exercises → higher price ===
    exercises = [1, 2]
    prices = []
    for n_ex in exercises:
        p = float(fd_swing_price(S0, K, T, r, q, sigma, n_exercises=n_ex,
                                  n_x=60, n_t=60))
        prices.append(p)
        print(f"  n_exercises={n_ex}: {p:.4f}")
    monotone = prices[1] >= prices[0] - 0.01

    # === Test 3: AD Greeks ===
    def swing_price_fn(params):
        s, strike, vol, rate = params
        return fd_swing_price(s, strike, T, rate, q, vol, n_exercises=1,
                              n_x=50, n_t=50)

    base = jnp.array([S0, K, sigma, r])
    greeks = np.array(jax.grad(swing_price_fn)(base))
    greek_names = ['delta', 'strike_sens', 'vega', 'rho']
    print(f"\n  --- AD Greeks (1 exercise) ---")
    for name, g in zip(greek_names, greeks):
        print(f"    {name:>12s}: {g:.6f}")

    # FD check
    H = 1e-4
    fd = np.zeros(4)
    for i in range(4):
        xp = base.at[i].add(H)
        xm = base.at[i].add(-H)
        fd[i] = (float(swing_price_fn(xp)) - float(swing_price_fn(xm))) / (2 * H)
    max_rel = np.max(np.abs(greeks - fd) / np.maximum(np.abs(fd), 1e-6))
    print(f"  AD vs FD max rel diff: {max_rel:.2e}")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = price_1 > 0
    n_pass += int(ok)
    print(f"\n  Positive price:     {'✓' if ok else '✗'}")

    n_pass += int(monotone)
    print(f"  Monotone exercises: {'✓' if monotone else '✗'}")

    ok = greeks[0] > 0  # positive delta
    n_pass += int(ok)
    print(f"  Positive delta:     {'✓' if ok else '✗'}")

    ok = max_rel < 0.01
    n_pass += int(ok)
    print(f"  AD vs FD:           {'✓' if ok else '✗'} ({max_rel:.2e})")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All swing option validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
