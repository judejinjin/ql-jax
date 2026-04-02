"""Example: Asian Option Pricing.

Prices Asian options with multiple methods:
  - Geometric Asian (closed-form)
  - Arithmetic Asian (Turnbull-Wakeman approximation)
  - Monte Carlo simulation
  - AD Greeks via JAX
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.asian import (
    geometric_asian_price,
    turnbull_wakeman_price,
)
from ql_jax.engines.mc.asian import mc_asian_arithmetic_bs


def main():
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.30
    n_fixings = 12  # monthly fixings

    print("=" * 60)
    print("QL-JAX Example: Asian Option Pricing")
    print("=" * 60)
    print(f"  Spot={S}, Strike={K}, T={T}y, r={r:.2%}, q={q:.2%}, σ={sigma:.2%}")
    print(f"  Monthly fixings: {n_fixings}")
    print()

    # ── Geometric Asian (closed-form) ────────────────────────
    geo_call = float(geometric_asian_price(S, K, T, r, q, sigma, n_fixings, option_type='call'))
    geo_put = float(geometric_asian_price(S, K, T, r, q, sigma, n_fixings, option_type='put'))
    print(f"  Geometric Asian Call:  {geo_call:.6f}")
    print(f"  Geometric Asian Put:   {geo_put:.6f}")

    # ── Arithmetic Asian (Turnbull-Wakeman) ──────────────────
    tw_call = float(turnbull_wakeman_price(S, K, T, r, q, sigma, n_fixings, option_type='call'))
    tw_put = float(turnbull_wakeman_price(S, K, T, r, q, sigma, n_fixings, option_type='put'))
    print(f"  T-W Approx Call:       {tw_call:.6f}")
    print(f"  T-W Approx Put:        {tw_put:.6f}")

    # ── Monte Carlo ──────────────────────────────────────────
    key = jax.random.PRNGKey(42)
    result = mc_asian_arithmetic_bs(
        S, K, T, r, q, sigma,
        option_type=1, n_fixings=n_fixings,
        n_paths=100_000,
        key=key,
    )
    mc_call = float(result[0]) if isinstance(result, tuple) else float(result)
    print(f"  MC Asian Call (100k):  {mc_call:.6f}")

    # ── AD Greeks ────────────────────────────────────────────
    print("\n  Greeks (Geometric Asian Call, via AD):")

    def price_fn(S, sigma):
        return geometric_asian_price(S, K, T, r, q, sigma, n_fixings, option_type='call')

    delta = float(jax.grad(price_fn, argnums=0)(S, sigma))
    vega = float(jax.grad(price_fn, argnums=1)(S, sigma))
    gamma = float(jax.grad(jax.grad(price_fn, argnums=0), argnums=0)(S, sigma))

    print(f"    Delta: {delta:.6f}")
    print(f"    Gamma: {gamma:.6f}")
    print(f"    Vega:  {vega:.6f}")


if __name__ == "__main__":
    main()
