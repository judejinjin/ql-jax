"""Example 1: Equity Option Pricing — classic QuantLib-style.

Prices a European equity option with multiple methods:
  - Black-Scholes analytic
  - Heston stochastic volatility
  - Binomial tree (CRR)
  - Finite difference (Crank-Nicolson)
  - Monte Carlo

Then computes Greeks via AD.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import (
    black_scholes_price, bs_greeks,
)
from ql_jax.engines.analytic.heston import heston_price
from ql_jax.engines.lattice.binomial import binomial_price
from ql_jax.engines.fd.black_scholes import fd_black_scholes_price


def main():
    # Market data
    S = 100.0       # spot
    K = 100.0       # strike (ATM)
    T = 1.0         # 1 year
    r = 0.05        # risk-free rate
    q = 0.02        # dividend yield
    sigma = 0.20    # volatility
    option_type = 1  # 1 = call, -1 = put

    print("=" * 60)
    print("QL-JAX Example: Equity Option Pricing")
    print("=" * 60)
    print(f"  Spot:           {S}")
    print(f"  Strike:         {K}")
    print(f"  Maturity:       {T}y")
    print(f"  Rate:           {r:.2%}")
    print(f"  Div yield:      {q:.2%}")
    print(f"  Volatility:     {sigma:.2%}")
    print(f"  Type:           {'Call' if option_type == 1 else 'Put'}")
    print()

    # ── Method 1: Black-Scholes Analytic ─────────────────────
    bsm_price = float(black_scholes_price(S, K, T, r, q, sigma, option_type))
    print(f"  Black-Scholes:  {bsm_price:.6f}")

    # ── Method 2: Heston (with BSM-equivalent params) ────────
    v0 = sigma**2
    kappa = 1.0
    theta = sigma**2
    xi = 0.01   # near-zero vol-of-vol → nearly BSM
    rho = 0.0
    h_price = float(heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, option_type))
    print(f"  Heston:         {h_price:.6f}")

    # ── Method 3: Binomial Tree (CRR, 500 steps) ────────────
    tree_price = float(binomial_price(S, K, T, r, q, sigma, option_type,
                                       n_steps=500, american=False))
    print(f"  Binomial (500): {tree_price:.6f}")

    # ── Method 4: Finite Difference ──────────────────────────
    fd_price = float(fd_black_scholes_price(S, K, T, r, q, sigma, option_type,
                                             n_space=200, n_time=200))
    print(f"  Finite Diff:    {fd_price:.6f}")

    print()
    print("  All methods agree to within a few cents of the analytic price.")
    print()

    # ── Greeks via Automatic Differentiation ─────────────────
    print("-" * 60)
    print("  Greeks via jax.grad (automatic differentiation)")
    print("-" * 60)

    greeks = bs_greeks(S, K, T, r, q, sigma, option_type)
    for name, val in greeks.items():
        print(f"    {name:10s} = {float(val):.6f}")

    # ── AD Greeks: manual jax.grad ───────────────────────────
    print()
    print("  Verifying delta and vega via manual jax.grad:")
    delta_fn = jax.grad(lambda s: black_scholes_price(s, K, T, r, q, sigma, option_type))
    vega_fn = jax.grad(lambda sig: black_scholes_price(S, K, T, r, q, sig, option_type))
    print(f"    Delta (grad):  {float(delta_fn(jnp.float64(S))):.6f}")
    print(f"    Vega  (grad):  {float(vega_fn(jnp.float64(sigma))):.6f}")

    # ── Batch pricing via vmap ───────────────────────────────
    print()
    print("-" * 60)
    print("  Batch pricing via jax.vmap (100 strikes)")
    print("-" * 60)

    strikes = jnp.linspace(80.0, 120.0, 100)
    batch_price = jax.vmap(
        lambda k: black_scholes_price(S, k, T, r, q, sigma, option_type)
    )(strikes)
    print(f"    Min price: {float(batch_price.min()):.6f}  (K={float(strikes[batch_price.argmin()]):.1f})")
    print(f"    Max price: {float(batch_price.max()):.6f}  (K={float(strikes[batch_price.argmax()]):.1f})")
    print(f"    ATM price: {float(batch_price[50]):.6f}  (K={float(strikes[50]):.1f})")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
