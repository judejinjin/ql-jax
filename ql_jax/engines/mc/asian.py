"""Monte Carlo Asian option pricing with control variate."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.methods.montecarlo.path import generate_paths_bs


def mc_asian_arithmetic_bs(
    S, K, T, r, q, sigma, option_type: int,
    n_fixings: int = 12,
    n_paths: int = 100_000,
    n_steps: int = None,
    key=None,
    antithetic: bool = True,
    control_variate: bool = True,
):
    """Monte Carlo arithmetic average Asian option price.

    Uses geometric average as control variate (when control_variate=True).

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : OptionType.Call or Put
    n_fixings : number of averaging dates
    n_paths : number of MC paths
    n_steps : time steps (defaults to n_fixings)
    key : jax.random.PRNGKey
    antithetic : use antithetic variates
    control_variate : use geometric average as CV

    Returns
    -------
    price : MC estimate
    stderr : standard error
    """
    from ql_jax.processes.black_scholes import BlackScholesProcess

    if key is None:
        key = jax.random.PRNGKey(42)

    if n_steps is None:
        n_steps = n_fixings

    S = jnp.asarray(S, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)

    process = BlackScholesProcess(
        spot=float(S), rate=float(r), dividend=float(q), volatility=float(sigma),
    )

    log_paths = generate_paths_bs(process, T, n_steps, n_paths, key, antithetic=antithetic)
    spot_paths = jnp.exp(log_paths)

    # Fixing indices (equally spaced in [1, n_steps])
    fix_indices = jnp.linspace(1, n_steps, n_fixings, dtype=jnp.int32)
    fixing_spots = spot_paths[:, fix_indices]

    # Arithmetic average
    arith_avg = jnp.mean(fixing_spots, axis=1)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    arith_payoff = jnp.maximum(phi * (arith_avg - K), 0.0)

    df = jnp.exp(-r * T)

    if control_variate:
        # Geometric average (control)
        geom_avg = jnp.exp(jnp.mean(jnp.log(fixing_spots), axis=1))
        geom_payoff = jnp.maximum(phi * (geom_avg - K), 0.0)

        # Analytic geometric Asian price
        from ql_jax.instruments.asian import analytic_discrete_geometric_asian_price
        geom_analytic = analytic_discrete_geometric_asian_price(
            S, K, T, r, q, sigma, option_type, n_fixings=n_fixings,
        )

        # Control variate adjustment
        discounted_arith = df * arith_payoff
        discounted_geom = df * geom_payoff

        # Optimal beta
        cov = jnp.mean(discounted_arith * discounted_geom) - jnp.mean(discounted_arith) * jnp.mean(discounted_geom)
        var_geom = jnp.var(discounted_geom)
        beta = jnp.where(var_geom > 1e-15, cov / var_geom, 0.0)

        adjusted = discounted_arith - beta * (discounted_geom - geom_analytic)
        price = jnp.mean(adjusted)
        stderr = jnp.std(adjusted) / jnp.sqrt(jnp.float64(adjusted.shape[0]))
    else:
        discounted = df * arith_payoff
        price = jnp.mean(discounted)
        stderr = jnp.std(discounted) / jnp.sqrt(jnp.float64(discounted.shape[0]))

    return price, stderr
