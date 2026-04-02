"""MC lookback option engine."""

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType


def mc_lookback_price(S, K, T, r, q, sigma, option_type: int,
                       lookback_type='floating', n_paths=100_000,
                       n_steps=500, key=None):
    """Price a lookback option via Monte Carlo.

    Parameters
    ----------
    S : float – spot
    K : float – strike (for fixed strike lookback)
    T : float – maturity
    r, q, sigma : float
    option_type : int
    lookback_type : 'floating' or 'fixed'
    n_paths, n_steps : int
    key : PRNGKey

    Returns
    -------
    price : float
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    dt = T / n_steps
    Z = jax.random.normal(key, (n_steps, n_paths))

    log_S = jnp.log(jnp.float64(S)) * jnp.ones(n_paths)
    S_max = jnp.float64(S) * jnp.ones(n_paths)
    S_min = jnp.float64(S) * jnp.ones(n_paths)

    for i in range(n_steps):
        log_S = log_S + (r - q - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * Z[i]
        S_curr = jnp.exp(log_S)
        S_max = jnp.maximum(S_max, S_curr)
        S_min = jnp.minimum(S_min, S_curr)

    S_T = jnp.exp(log_S)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    if lookback_type == 'floating':
        # Call: S_T - S_min, Put: S_max - S_T
        if option_type == OptionType.Call:
            payoff = S_T - S_min
        else:
            payoff = S_max - S_T
    else:
        # Fixed strike: Call: max(S_max - K, 0), Put: max(K - S_min, 0)
        if option_type == OptionType.Call:
            payoff = jnp.maximum(S_max - K, 0.0)
        else:
            payoff = jnp.maximum(K - S_min, 0.0)

    return jnp.exp(-r * T) * jnp.mean(payoff)
