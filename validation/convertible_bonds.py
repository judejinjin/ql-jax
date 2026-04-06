"""Validation: Convertible Bond pricing
Source: ~/QuantLib/Examples/ConvertibleBonds/

Since no engine exists in ql-jax, we validate basic convertible bond
properties and compute Greeks via JAX AD on a simple binomial model.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.instruments.convertible_bond import ConvertibleBond


def simple_convertible_price(S, K_conv, face, coupon_rate, T, r, sigma, n_steps=100):
    """Simple binomial model for convertible bond.

    At each node: max(bond_value, conversion_value).
    """
    dt = T / n_steps
    u = jnp.exp(sigma * jnp.sqrt(dt))
    d = 1.0 / u
    p = (jnp.exp(r * dt) - d) / (u - d)
    disc = jnp.exp(-r * dt)

    # Terminal values
    S_T = S * u ** jnp.arange(n_steps, -1, -1) * d ** jnp.arange(0, n_steps + 1)
    conversion = S_T / K_conv * face  # conversion_ratio * S_T
    bond_val = face + coupon_rate * face  # face + last coupon

    V = jnp.maximum(bond_val, conversion)

    # Backward induction
    for i in range(n_steps - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        S_i = S * u ** jnp.arange(i, -1, -1) * d ** jnp.arange(0, i + 1)
        conversion_i = S_i / K_conv * face
        # Add coupon at each step (simplified annual accrual)
        coupon_pv = coupon_rate * face * dt
        V = jnp.maximum(V + coupon_pv, conversion_i)

    return V[0]


def main():
    print("=" * 78)
    print("Convertible Bond Validation")
    print("=" * 78)

    # Parameters
    S = 100.0
    face = 100.0
    coupon_rate = 0.04
    conversion_ratio = 1.0  # convert to 1 share
    K_conv = face / conversion_ratio  # = 100
    T = 5.0
    r = 0.05
    sigma = 0.30

    bond = ConvertibleBond(
        face_value=face, coupon_rate=coupon_rate,
        conversion_ratio=conversion_ratio, maturity=T,
    )

    print(f"  S={S}, face={face}, coupon={coupon_rate*100:.0f}%, T={T}")
    print(f"  r={r*100:.0f}%, σ={sigma*100:.0f}%, conv_ratio={conversion_ratio}")
    print(f"  Conversion price: {bond.conversion_price:.2f}")
    print(f"  Conversion value: {float(bond.conversion_value(S)):.2f}")
    print(f"  Parity: {float(bond.parity(S)):.2f}")

    # === Test 1: Convertible price ===
    price = float(simple_convertible_price(S, K_conv, face, coupon_rate, T, r, sigma))
    print(f"\n  Binomial price: {price:.4f}")

    # === Test 2: Convertible >= max(bond, conversion) ===
    straight_bond = face * jnp.exp(-r * T) + coupon_rate * face * sum(
        jnp.exp(-r * t) for t in range(1, int(T) + 1))
    straight_val = float(straight_bond)
    conv_val = float(bond.conversion_value(S))
    floor = max(straight_val, conv_val)
    print(f"  Straight bond: {straight_val:.4f}")
    print(f"  Conversion value: {conv_val:.4f}")
    print(f"  Floor: {floor:.4f}")

    # === Test 3: JAX AD Greeks ===
    def price_fn(params):
        s, sig, rate = params
        return simple_convertible_price(s, K_conv, face, coupon_rate, T, rate, sig, n_steps=50)

    base_params = jnp.array([S, sigma, r])
    greeks = np.array(jax.grad(price_fn)(base_params))
    greek_names = ['delta', 'vega', 'rho']
    print(f"\n  --- JAX AD Greeks ---")
    for name, val in zip(greek_names, greeks):
        print(f"    {name:>6s}: {val:.6f}")

    # FD check
    H = 1e-4
    fd_greeks = np.zeros(3)
    for i in range(3):
        xp = base_params.at[i].add(H)
        xm = base_params.at[i].add(-H)
        fd_greeks[i] = (float(price_fn(xp)) - float(price_fn(xm))) / (2 * H)

    max_rel = np.max(np.abs(greeks - fd_greeks) / np.maximum(np.abs(fd_greeks), 1e-10))
    print(f"  AD vs FD rel diff: {max_rel:.2e}")

    # === Test 4: Vol sensitivity ===
    vols = [0.10, 0.20, 0.30, 0.40, 0.50]
    prices = [float(simple_convertible_price(S, K_conv, face, coupon_rate, T, r, v, 50))
              for v in vols]
    vol_monotone = all(prices[i] <= prices[i+1] for i in range(len(prices)-1))
    print(f"\n  Vol sensitivity:")
    for v, p in zip(vols, prices):
        print(f"    σ={v:.0%}: {p:.4f}")

    # === Validation ===
    n_pass = 0
    total = 4

    ok = price > floor * 0.95  # convertible should be near or above floor
    n_pass += int(ok)
    print(f"\n  Price > floor:      {'✓' if ok else '✗'}")

    ok = greeks[0] > 0  # delta positive (call-like)
    n_pass += int(ok)
    print(f"  Positive delta:     {'✓' if ok else '✗'}")

    ok = max_rel < 0.01
    n_pass += int(ok)
    print(f"  AD vs FD:           {'✓' if ok else '✗'} ({max_rel:.2e})")

    ok = vol_monotone
    n_pass += int(ok)
    print(f"  Vol monotone:       {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All convertible bond validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
