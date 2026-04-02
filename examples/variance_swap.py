"""Example: Variance Swap Pricing.

Prices a variance swap using:
  - Analytic fair variance (BS model)
  - Realized variance from simulated paths
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.variance_swap import variance_swap_fair_strike
from ql_jax.models.volatility.garch import realized_volatility


def main():
    S = 100.0
    K_var = 0.04   # variance strike (σ² = 0.20²)
    T = 1.0
    r = 0.05
    q = 0.02
    sigma = 0.20

    print("=" * 60)
    print("QL-JAX Example: Variance Swap Pricing")
    print("=" * 60)
    print(f"  Spot={S}, T={T}y, σ={sigma:.2%}")
    print(f"  Variance strike: {K_var:.4f} (implied vol {jnp.sqrt(K_var):.2%})")
    print()

    # ── Fair Variance Strike ─────────────────────────────────
    fair_var = float(variance_swap_fair_strike(S, T, r, q, sigma))
    print(f"  Fair variance strike:   {fair_var:.6f}")
    print(f"  Fair implied vol:       {jnp.sqrt(fair_var):.4%}")
    print()

    # ── Simulate paths and compute realized vol ──────────────
    key = jax.random.PRNGKey(0)
    n_paths = 10_000
    n_steps = 252

    dt = T / n_steps
    z = jax.random.normal(key, (n_paths, n_steps))
    drift = (r - q - 0.5 * sigma**2) * dt
    log_returns = drift + sigma * jnp.sqrt(dt) * z
    paths = S * jnp.exp(jnp.cumsum(log_returns, axis=1))

    # Realized vol from log returns (annualized)
    realized_var = jnp.mean(jnp.sum(log_returns**2, axis=1) / T)
    print(f"  MC realized variance:   {float(realized_var):.6f}")
    print(f"  MC realized vol:        {float(jnp.sqrt(realized_var)):.4%}")
    print()

    # ── P&L for a variance swap ──────────────────────────────
    notional = 1e6
    pnl = notional * (realized_var - K_var)
    print(f"  Variance swap P&L (notional={notional/1e6:.0f}M):")
    print(f"    Per path mean:   {float(pnl):.2f}")


if __name__ == "__main__":
    main()
