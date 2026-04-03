"""Forward option engines.

Forward-starting vanilla option pricing and performance (return) options.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType
from ql_jax.engines.analytic.black_formula import black_scholes_price


def forward_start_price(
    S, T_start, T_end, r, q, sigma, alpha,
    option_type: int = 1,
):
    """Forward-starting European option price.

    At time T_start, the strike is set to alpha * S(T_start).
    Payoff at T_end: max(phi * (S(T_end) - alpha * S(T_start)), 0).

    For a standard forward-start (alpha=1), this is an ATM option
    whose strike is set at the forward date.

    Parameters
    ----------
    S : current spot
    T_start : forward start date
    T_end : expiry date (T_end > T_start)
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    alpha : moneyness ratio (K = alpha * S(T_start))
    option_type : 1 for call, -1 for put

    Returns
    -------
    price : forward-starting option price
    """
    S, T_start, T_end, r, q, sigma, alpha = (
        jnp.float64(x) for x in (S, T_start, T_end, r, q, sigma, alpha)
    )

    # For GBM, forward-start price = S * exp(-q*T_start) * BS(1, alpha, tau, ...)
    # where tau = T_end - T_start
    tau = T_end - T_start
    scaling = S * jnp.exp(-q * T_start)

    # BS price with spot=1, K=alpha, T=tau
    bs_unit = black_scholes_price(1.0, alpha, tau, r, q, sigma, option_type)

    return scaling * bs_unit


def forward_performance_price(
    S, T_start, T_end, r, q, sigma,
    option_type: int = 1,
):
    """Forward performance (return) option price.

    Payoff at T_end: max(phi * (S(T_end)/S(T_start) - 1), 0).

    Parameters
    ----------
    S : current spot
    T_start : forward date
    T_end : expiry
    r, q : rates
    sigma : vol
    option_type : 1 for call, -1 for put

    Returns
    -------
    price : performance option price
    """
    # Performance option = forward-start with alpha=1, normalized by S(T_start)
    return forward_start_price(S, T_start, T_end, r, q, sigma, 1.0, option_type) / S


def mc_forward_european_price(
    S, T_start, T_end, r, q, sigma, alpha,
    option_type: int = 1, n_paths: int = 100000,
    key=None,
):
    """Monte Carlo forward-starting European option price.

    Parameters
    ----------
    S : spot
    T_start, T_end : forward start and expiry
    r, q, sigma : market params
    alpha : moneyness
    option_type : call/put
    n_paths : MC paths
    key : JAX random key

    Returns
    -------
    price : MC estimate
    """
    import jax
    if key is None:
        key = jax.random.PRNGKey(42)

    S, T_start, T_end, r, q, sigma, alpha = (
        jnp.float64(x) for x in (S, T_start, T_end, r, q, sigma, alpha)
    )
    phi = jnp.float64(option_type)

    key1, key2 = jax.random.split(key)
    z1 = jax.random.normal(key1, (n_paths,))
    z2 = jax.random.normal(key2, (n_paths,))

    # S at T_start
    S_start = S * jnp.exp((r - q - 0.5 * sigma**2) * T_start + sigma * jnp.sqrt(T_start) * z1)

    # S at T_end given S at T_start
    tau = T_end - T_start
    S_end = S_start * jnp.exp((r - q - 0.5 * sigma**2) * tau + sigma * jnp.sqrt(tau) * z2)

    K_fwd = alpha * S_start
    payoff = jnp.maximum(phi * (S_end - K_fwd), 0.0)

    return jnp.exp(-r * T_end) * jnp.mean(payoff)
