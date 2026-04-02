"""MC variance swap engine."""

import jax
import jax.numpy as jnp


def mc_variance_swap_price(S, T, r, q, sigma, strike_var,
                            notional=1.0, n_paths=100_000, n_steps=252,
                            key=None):
    """Price a variance swap via Monte Carlo.

    Parameters
    ----------
    S : float – spot
    T : float – maturity
    r, q, sigma : float
    strike_var : float – variance strike K_var
    notional : float – variance notional
    n_paths, n_steps : int
    key : PRNGKey or None

    Returns
    -------
    npv : float
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    dt = T / n_steps
    Z = jax.random.normal(key, (n_steps, n_paths))

    log_S = jnp.log(jnp.float64(S)) * jnp.ones(n_paths)
    sum_log_ret_sq = jnp.zeros(n_paths)

    for i in range(n_steps):
        new_log_S = log_S + (r - q - 0.5 * sigma**2) * dt + sigma * jnp.sqrt(dt) * Z[i]
        log_ret = new_log_S - log_S
        sum_log_ret_sq = sum_log_ret_sq + log_ret**2
        log_S = new_log_S

    # Annualized realized variance
    realized_var = sum_log_ret_sq / T

    # Payoff
    payoff = notional * (realized_var - strike_var)
    return jnp.exp(-r * T) * jnp.mean(payoff)


def mc_variance_swap_heston(S, T, r, q, v0, kappa, theta, sigma_v, rho,
                              strike_var, notional=1.0,
                              n_paths=100_000, n_steps=252, key=None):
    """Price a variance swap under Heston via Monte Carlo.

    Parameters
    ----------
    S, T, r, q : float
    v0, kappa, theta, sigma_v, rho : float – Heston params
    strike_var : float
    notional : float
    n_paths, n_steps : int
    key : PRNGKey or None

    Returns
    -------
    npv : float
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    dt = T / n_steps
    key1, key2 = jax.random.split(key)
    Z1 = jax.random.normal(key1, (n_steps, n_paths))
    Z2 = jax.random.normal(key2, (n_steps, n_paths))
    W2 = rho * Z1 + jnp.sqrt(1.0 - rho**2) * Z2

    log_S = jnp.log(jnp.float64(S)) * jnp.ones(n_paths)
    v = jnp.float64(v0) * jnp.ones(n_paths)
    sum_log_ret_sq = jnp.zeros(n_paths)

    for i in range(n_steps):
        v_pos = jnp.maximum(v, 0.0)
        new_log_S = log_S + (r - q - 0.5 * v_pos) * dt + jnp.sqrt(v_pos * dt) * Z1[i]
        v = v + kappa * (theta - v_pos) * dt + sigma_v * jnp.sqrt(v_pos * dt) * W2[i]

        log_ret = new_log_S - log_S
        sum_log_ret_sq = sum_log_ret_sq + log_ret**2
        log_S = new_log_S

    realized_var = sum_log_ret_sq / T
    payoff = notional * (realized_var - strike_var)
    return jnp.exp(-r * T) * jnp.mean(payoff)
