"""Example: Portfolio Greeks via AD (vmap showcase).

Demonstrates JAX's automatic differentiation and vectorization:
  - AD Greeks (delta, gamma, vega, theta, rho) for a single option
  - vmap to compute Greeks for an entire portfolio in parallel
  - Comparison: AD vs finite-difference Greeks
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import black_scholes_price


def main():
    print("=" * 60)
    print("QL-JAX Example: Portfolio Greeks via AD")
    print("=" * 60)
    print()

    # ── Single option Greeks ─────────────────────────────────
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20
    option_type = 1

    def price_fn(S, K, T, r, q, sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type)

    price = float(price_fn(S, K, T, r, q, sigma))
    print(f"  ATM Call: S={S}, K={K}, T={T}, σ={sigma:.0%}")
    print(f"  Price: {price:.6f}")
    print()

    # First-order Greeks via AD
    delta = float(jax.grad(price_fn, argnums=0)(S, K, T, r, q, sigma))
    vega = float(jax.grad(price_fn, argnums=5)(S, K, T, r, q, sigma))
    rho_greek = float(jax.grad(price_fn, argnums=3)(S, K, T, r, q, sigma))
    theta = -float(jax.grad(price_fn, argnums=2)(S, K, T, r, q, sigma))

    # Second-order Greeks
    gamma = float(jax.grad(jax.grad(price_fn, argnums=0), argnums=0)(S, K, T, r, q, sigma))
    vanna = float(jax.grad(jax.grad(price_fn, argnums=0), argnums=5)(S, K, T, r, q, sigma))
    volga = float(jax.grad(jax.grad(price_fn, argnums=5), argnums=5)(S, K, T, r, q, sigma))

    print("  AD Greeks (exact):")
    print(f"    Delta:  {delta:.6f}")
    print(f"    Gamma:  {gamma:.6f}")
    print(f"    Vega:   {vega:.6f}")
    print(f"    Theta:  {theta:.6f}")
    print(f"    Rho:    {rho_greek:.6f}")
    print(f"    Vanna:  {vanna:.6f}")
    print(f"    Volga:  {volga:.6f}")
    print()

    # Finite-difference comparison
    eps = 1e-4
    fd_delta = float((price_fn(S + eps, K, T, r, q, sigma) - price_fn(S - eps, K, T, r, q, sigma)) / (2 * eps))
    fd_gamma = float((price_fn(S + eps, K, T, r, q, sigma) - 2 * price_fn(S, K, T, r, q, sigma)
                       + price_fn(S - eps, K, T, r, q, sigma)) / eps**2)
    print(f"  FD Delta: {fd_delta:.6f}  (AD: {delta:.6f}, diff: {abs(delta - fd_delta):.2e})")
    print(f"  FD Gamma: {fd_gamma:.6f}  (AD: {gamma:.6f}, diff: {abs(gamma - fd_gamma):.2e})")
    print()

    # ── Portfolio Greeks with vmap ───────────────────────────
    print("  Portfolio: 10 options with different strikes/vols")
    n = 10
    spots = jnp.full(n, 100.0)
    strikes = jnp.linspace(80.0, 120.0, n)
    maturities = jnp.full(n, 1.0)
    rates = jnp.full(n, 0.05)
    divs = jnp.full(n, 0.02)
    vols = jnp.linspace(0.15, 0.35, n)
    positions = jnp.array([1, -1, 1, 1, -1, 1, -1, 1, 1, -1], dtype=jnp.float64)

    # Vectorized pricing
    portfolio_prices = jax.vmap(
        lambda s, k, t, r_, q_, sig: black_scholes_price(s, k, t, r_, q_, sig, 1)
    )(spots, strikes, maturities, rates, divs, vols)

    # Vectorized Greeks
    delta_fn = jax.vmap(
        jax.grad(lambda s, k, t, r_, q_, sig: black_scholes_price(s, k, t, r_, q_, sig, 1), argnums=0)
    )
    gamma_fn = jax.vmap(
        jax.grad(jax.grad(lambda s, k, t, r_, q_, sig: black_scholes_price(s, k, t, r_, q_, sig, 1), argnums=0), argnums=0)
    )
    vega_fn = jax.vmap(
        jax.grad(lambda s, k, t, r_, q_, sig: black_scholes_price(s, k, t, r_, q_, sig, 1), argnums=5)
    )

    deltas = delta_fn(spots, strikes, maturities, rates, divs, vols)
    gammas = gamma_fn(spots, strikes, maturities, rates, divs, vols)
    vegas = vega_fn(spots, strikes, maturities, rates, divs, vols)

    print(f"\n  {'Strike':>8s} {'Pos':>4s} {'Price':>10s} {'Delta':>8s} {'Gamma':>8s} {'Vega':>8s}")
    print("  " + "-" * 52)
    for i in range(n):
        print(f"  {float(strikes[i]):8.1f} {int(positions[i]):4d} "
              f"{float(portfolio_prices[i]):10.4f} {float(deltas[i]):8.4f} "
              f"{float(gammas[i]):8.4f} {float(vegas[i]):8.4f}")

    # Aggregate portfolio Greeks
    port_delta = float(jnp.sum(positions * deltas))
    port_gamma = float(jnp.sum(positions * gammas))
    port_vega = float(jnp.sum(positions * vegas))
    port_value = float(jnp.sum(positions * portfolio_prices))

    print("  " + "-" * 52)
    print(f"  {'Total':>8s} {'':>4s} {port_value:10.4f} {port_delta:8.4f} "
          f"{port_gamma:8.4f} {port_vega:8.4f}")


if __name__ == "__main__":
    main()
