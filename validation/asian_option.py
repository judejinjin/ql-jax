"""Validation: Asian Option Pricing
Source: ~/QuantLib/Examples/AsianOption/ (conceptual)

Validates Asian option pricing across multiple methods:
  - Continuous geometric Asian (analytic)
  - Discrete geometric Asian (analytic)
  - Turnbull-Wakeman arithmetic Asian (analytic approx)
  - Cross-method consistency checks
  - JAX AD Greeks

Market data: S=100, K=100, r=0.05, q=0.02, σ=0.20, T=1.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from ql_jax.instruments.asian import (
    analytic_continuous_geometric_asian_price,
    analytic_discrete_geometric_asian_price,
)
from ql_jax.engines.analytic.asian import (
    geometric_asian_price,
    turnbull_wakeman_price,
)


# === Market data ===
S0 = 100.0
K = 100.0
r = 0.05
q = 0.02
sigma = 0.20
T = 1.0
N_FIXINGS = 12  # monthly fixings


def main():
    print("=" * 78)
    print("Asian Option Pricing Validation")
    print("=" * 78)
    print(f"  Instrument : Asian Options  S={S0}, K={K}, σ={sigma}, r={r}, q={q}, T={T}")
    print(f"  Fixings    : {N_FIXINGS} (monthly)")
    print()

    n_pass = 0
    total = 0

    # === 1. Continuous geometric Asian call ===
    cga_call = float(analytic_continuous_geometric_asian_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma), 1))
    cga_put = float(analytic_continuous_geometric_asian_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma), -1))
    print(f"  Continuous geometric Asian call: {cga_call:.6f}")
    print(f"  Continuous geometric Asian put:  {cga_put:.6f}")
    total += 1
    ok = cga_call > 0 and cga_put > 0
    n_pass += int(ok)
    print(f"  Positive prices: {'✓' if ok else '✗'}")

    # === 2. Discrete geometric Asian ===
    dga_call = float(analytic_discrete_geometric_asian_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma), 1, N_FIXINGS))
    dga_put = float(analytic_discrete_geometric_asian_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma), -1, N_FIXINGS))
    print(f"\n  Discrete geometric Asian call ({N_FIXINGS} fixings): {dga_call:.6f}")
    print(f"  Discrete geometric Asian put  ({N_FIXINGS} fixings): {dga_put:.6f}")
    total += 1
    ok = dga_call > 0 and dga_put > 0
    n_pass += int(ok)
    print(f"  Positive prices: {'✓' if ok else '✗'}")

    # === 3. Turnbull-Wakeman arithmetic Asian ===
    tw_call = float(turnbull_wakeman_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma),
        N_FIXINGS, option_type='call'))
    tw_put = float(turnbull_wakeman_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma),
        N_FIXINGS, option_type='put'))
    print(f"\n  Turnbull-Wakeman arithmetic Asian call: {tw_call:.6f}")
    print(f"  Turnbull-Wakeman arithmetic Asian put:  {tw_put:.6f}")
    total += 1
    ok = tw_call > 0 and tw_put > 0
    n_pass += int(ok)
    print(f"  Positive prices: {'✓' if ok else '✗'}")

    # === 4. Engine geometric Asian price ===
    eng_call = float(geometric_asian_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma),
        N_FIXINGS, option_type='call'))
    print(f"\n  Engine geometric Asian call: {eng_call:.6f}")
    total += 1
    diff = abs(eng_call - dga_call)
    ok = diff < max(abs(dga_call) * 0.05, 0.01)  # 5% tolerance
    n_pass += int(ok)
    print(f"  vs discrete geometric: diff={diff:.6f} {'✓' if ok else '✗'}")

    # === 5. Geometric ≤ Arithmetic (Jensen's inequality) ===
    total += 1
    ok = dga_call <= tw_call + 0.01  # geometric ≤ arithmetic + tolerance
    n_pass += int(ok)
    print(f"\n  Geometric call ≤ Arithmetic call: {dga_call:.4f} ≤ {tw_call:.4f} {'✓' if ok else '✗'}")

    # === 6. Continuous → discrete convergence (many fixings) ===
    dga_many = float(analytic_discrete_geometric_asian_price(
        jnp.float64(S0), jnp.float64(K), jnp.float64(T),
        jnp.float64(r), jnp.float64(q), jnp.float64(sigma), 1, 252))
    total += 1
    diff = abs(dga_many - cga_call)
    ok = diff < max(abs(cga_call) * 0.01, 0.01)
    n_pass += int(ok)
    print(f"  Discrete(252) → Continuous: {dga_many:.6f} vs {cga_call:.6f}, diff={diff:.4f} {'✓' if ok else '✗'}")

    # === 7. JAX AD Greeks ===
    print(f"\n  --- JAX AD Asian option Greeks ---")

    def asian_price_fn(spot, vol, rate):
        return turnbull_wakeman_price(spot, jnp.float64(K), jnp.float64(T),
                                      rate, jnp.float64(q), vol,
                                      N_FIXINGS, option_type='call')

    args = (jnp.float64(S0), jnp.float64(sigma), jnp.float64(r))
    grads = jax.grad(asian_price_fn, argnums=(0, 1, 2))(*args)

    # FD check
    h = 1e-5
    base = float(asian_price_fn(*args))
    fd = [(float(asian_price_fn(*(args[j] + h if j == i else args[j] for j in range(3)))) - base) / h for i in range(3)]

    names = ["Delta (dV/dS)", "Vega (dV/dσ)", "Rho (dV/dr)"]
    print(f"  {'Greek':<16} {'jax.grad':>12} {'FD':>12} {'|diff|':>12}")
    print("  " + "-" * 54)
    for i, name in enumerate(names):
        total += 1
        ad = float(grads[i])
        diff = abs(ad - fd[i])
        ok = diff < max(abs(fd[i]) * 1e-3, 1e-4)
        n_pass += int(ok)
        print(f"  {name:<16} {ad:>12.6f} {fd[i]:>12.6f} {diff:>12.2e} {'✓' if ok else '✗'}")

    print(f"\nPassed: {n_pass}/{total}")
    if n_pass == total:
        print("✓ All Asian option validations passed.")
    else:
        print("✗ Some tests failed.")


if __name__ == "__main__":
    main()
