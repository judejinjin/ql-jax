"""Monte Carlo European option pricing.

Supports Black-Scholes and Heston processes with optional variance reduction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.methods.montecarlo.path import generate_paths_bs, generate_paths_heston


def mc_european_bs(
    S, K, T, r, q, sigma, option_type: int,
    n_paths: int = 100_000,
    n_steps: int = 1,
    key=None,
    antithetic: bool = True,
):
    """Monte Carlo European option price under Black-Scholes.

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : OptionType.Call or Put
    n_paths : number of MC paths
    n_steps : number of time steps per path
    key : jax.random.PRNGKey
    antithetic : use antithetic variates

    Returns
    -------
    price : MC estimate
    stderr : standard error
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    S = jnp.asarray(S, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    q = jnp.asarray(q, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)

    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)
    drift = (r - q - 0.5 * sigma**2) * dt

    if antithetic:
        z = jax.random.normal(key, shape=(n_paths, n_steps))
        z = jnp.concatenate([z, -z], axis=0)
    else:
        z = jax.random.normal(key, shape=(n_paths, n_steps))

    # Build log-spot increments: drift*dt + sigma*sqrt(dt)*Z
    increments = drift + sigma * sqrt_dt * z  # (n_eff_paths, n_steps)
    log_S0 = jnp.log(S)
    log_ST = log_S0 + jnp.sum(increments, axis=1)
    S_T = jnp.exp(log_ST)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    payoffs = jnp.maximum(phi * (S_T - K), 0.0)

    df = jnp.exp(-r * T)
    discounted = df * payoffs

    price = jnp.mean(discounted)
    stderr = jnp.std(discounted) / jnp.sqrt(jnp.float64(discounted.shape[0]))

    return price, stderr


def mc_european_heston(
    S, K, T, r, q,
    v0, kappa, theta, xi, rho,
    option_type: int,
    n_paths: int = 100_000,
    n_steps: int = 100,
    key=None,
    antithetic: bool = True,
):
    """Monte Carlo European option price under Heston model.

    Parameters
    ----------
    S, K, T, r, q : market data
    v0, kappa, theta, xi, rho : Heston parameters
    option_type : OptionType.Call or Put
    n_paths : number of MC paths
    n_steps : time steps
    key : jax.random.PRNGKey
    antithetic : use antithetic variates

    Returns
    -------
    price : MC estimate
    stderr : standard error
    """
    from ql_jax.processes.heston import HestonProcess

    if key is None:
        key = jax.random.PRNGKey(42)

    S = jnp.asarray(S, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)

    process = HestonProcess(
        spot=float(S), rate=float(r), dividend=float(q),
        v0=float(v0), kappa=float(kappa), theta=float(theta),
        xi=float(xi), rho=float(rho),
    )

    log_paths, _ = generate_paths_heston(
        process, T, n_steps, n_paths, key, antithetic=antithetic,
    )
    S_T = jnp.exp(log_paths[:, -1])

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    payoffs = jnp.maximum(phi * (S_T - K), 0.0)

    df = jnp.exp(-r * T)
    discounted = df * payoffs

    price = jnp.mean(discounted)
    stderr = jnp.std(discounted) / jnp.sqrt(jnp.float64(discounted.shape[0]))

    return price, stderr
