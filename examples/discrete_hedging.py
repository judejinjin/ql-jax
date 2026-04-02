"""Example: Discrete Hedging Simulation.

Simulates delta hedging of a European call option:
  - Compares hedging P&L with different rebalancing frequencies
  - Shows hedging error distribution
  - Demonstrates vmap for parallel simulation of many paths
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from ql_jax.engines.analytic.black_formula import black_scholes_price


def main():
    print("=" * 60)
    print("QL-JAX Example: Discrete Hedging Simulation")
    print("=" * 60)
    print()

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0
    sigma = 0.20
    option_type = 1  # call

    n_paths = 50_000

    initial_price = float(black_scholes_price(S0, K, T, r, q, sigma, option_type))
    print(f"  Option: ATM Call, S₀={S0}, K={K}, T={T}y, σ={sigma:.0%}")
    print(f"  Initial BS price: {initial_price:.4f}")
    print()

    # Delta function via AD
    delta_fn = jax.jit(jax.grad(
        lambda s, t_rem: black_scholes_price(s, K, t_rem, r, q, sigma, option_type)
    ))

    rebalance_freqs = [1, 5, 21, 63, 252]  # daily through annual
    key = jax.random.PRNGKey(42)

    print(f"  {'Rebalances':>12s} {'Mean P&L':>10s} {'Std P&L':>10s} {'Hedge Eff':>10s}")
    print("  " + "-" * 46)

    for n_rebal in rebalance_freqs:
        dt = T / n_rebal
        t_grid = jnp.linspace(0, T, n_rebal + 1)

        # Simulate GBM paths
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n_paths, n_rebal))
        log_ret = (r - q - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * z
        log_S = jnp.concatenate([
            jnp.full((n_paths, 1), jnp.log(S0)),
            jnp.cumsum(log_ret, axis=1) + jnp.log(S0),
        ], axis=1)
        S_paths = jnp.exp(log_S)

        # Hedging P&L for each path
        def hedge_path(S_path):
            cash = initial_price  # option premium received
            shares = 0.0

            def step(carry, i):
                cash, shares = carry
                S_t = S_path[i]
                t_rem = jnp.maximum(T - t_grid[i], 1e-8)
                new_delta = delta_fn(S_t, t_rem)
                # Rebalance: buy/sell shares
                trade = new_delta - shares
                cash = cash - trade * S_t
                cash = cash * jnp.exp(r * dt)  # earn interest
                return (cash, new_delta), None

            (final_cash, final_shares), _ = jax.lax.scan(
                step, (cash, 0.0), jnp.arange(n_rebal)
            )

            # At expiry: unwind and settle
            S_T = S_path[-1]
            payoff = jnp.maximum(option_type * (S_T - K), 0.0)
            pnl = final_cash + final_shares * S_T - payoff
            return pnl

        hedge_pnl = jax.jit(jax.vmap(hedge_path))(S_paths)

        mean_pnl = float(jnp.mean(hedge_pnl))
        std_pnl = float(jnp.std(hedge_pnl))
        # Hedge effectiveness: 1 - var(hedge_error) / var(unhedged)
        unhedged_payoff = jnp.maximum(S_paths[:, -1] - K, 0.0)
        var_unhedged = float(jnp.var(unhedged_payoff))
        hedge_eff = 1.0 - float(jnp.var(hedge_pnl)) / var_unhedged if var_unhedged > 0 else 0.0

        print(f"  {n_rebal:12d} {mean_pnl:10.4f} {std_pnl:10.4f} {hedge_eff:10.4f}")

    print()
    print("  Hedge effectiveness → 1.0 as rebalancing frequency increases")
    print("  (assuming zero transaction costs)")


if __name__ == "__main__":
    main()
