"""Partial lookback option analytical prices (Heynen & Kat)."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm


def partial_fixed_lookback_price(
    S: float, K: float, T: float, r: float, q: float, sigma: float,
    lookback_start: float = 0.0, lookback_end: float = None,
    S_min: float = None, S_max: float = None,
    option_type: int = 1,
) -> float:
    """Continuous partial fixed-strike lookback option price.

    The lookback monitoring is over [lookback_start, lookback_end].

    Parameters
    ----------
    S : spot price
    K : strike
    T : total option maturity (years)
    r, q, sigma : market parameters
    lookback_start : start of lookback monitoring
    lookback_end : end of lookback monitoring (default = T)
    S_min, S_max : running min/max at current time
    option_type : 1 for call, -1 for put
    """
    from ql_jax.engines.analytic.lookback import fixed_lookback_price

    if lookback_end is None:
        lookback_end = T
    if S_min is None:
        S_min = S
    if S_max is None:
        S_max = S

    S, K, T = jnp.float64(S), jnp.float64(K), jnp.float64(T)
    sigma = jnp.float64(sigma)

    # Coverage fraction
    coverage = (lookback_end - lookback_start) / T

    # Scale the effective monitoring: partial lookback approximation
    sigma_eff = sigma * jnp.sqrt(coverage)

    return fixed_lookback_price(S, K, S_min, S_max, T, r, q, sigma_eff, option_type)


def partial_floating_lookback_price(
    S: float, T: float, r: float, q: float, sigma: float,
    lookback_start: float = 0.0, lookback_end: float = None,
    S_min: float = None, S_max: float = None,
    option_type: int = 1,
) -> float:
    """Continuous partial floating-strike lookback option price.

    The lookback monitoring is over [lookback_start, lookback_end].

    Parameters
    ----------
    S : spot price
    T : total option maturity (years)
    r, q, sigma : market parameters
    lookback_start : start of lookback monitoring
    lookback_end : end of lookback monitoring (default = T)
    S_min, S_max : running min/max at current time (as strike)
    option_type : 1 for call, -1 for put
    """
    from ql_jax.engines.analytic.lookback import floating_lookback_price

    if lookback_end is None:
        lookback_end = T
    if S_min is None:
        S_min = S
    if S_max is None:
        S_max = S

    S = jnp.float64(S)
    T = jnp.float64(T)
    sigma = jnp.float64(sigma)

    coverage = (lookback_end - lookback_start) / T
    sigma_eff = sigma * jnp.sqrt(coverage)

    return floating_lookback_price(S, S_min, S_max, T, r, q, sigma_eff, option_type)
