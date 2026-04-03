"""European option pricing via direct integration.

Prices European options by integrating the terminal payoff against the
risk-neutral density. Supports Black-Scholes and general local-vol densities.
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.engines.analytic.black_formula import black_scholes_price


def integral_price(S, K, T, r, q, sigma, option_type: int, n_points: int = 200):
    """European option price via numerical integration of the terminal payoff.

    Uses Gauss-Legendre quadrature over the terminal stock price distribution.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    option_type : OptionType.Call (1) or OptionType.Put (-1)
    n_points : quadrature points

    Returns
    -------
    price : European option price (should match BSM)
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))

    F = S * jnp.exp((r - q) * T)
    disc = jnp.exp(-r * T)
    vol_sqrt_t = sigma * jnp.sqrt(T)

    # Integration over log-normal distribution using trapezoidal on z ~ N(0,1)
    # S_T = F * exp(-0.5*sig^2*T + sig*sqrt(T)*z)
    z_max = 8.0
    z = jnp.linspace(-z_max, z_max, n_points)
    dz = z[1] - z[0]

    s_t = F * jnp.exp(-0.5 * vol_sqrt_t**2 + vol_sqrt_t * z)
    phi_val = jnp.float64(option_type)
    payoff = jnp.maximum(phi_val * (s_t - K), 0.0)

    # N(0,1) density
    pdf = jnp.exp(-0.5 * z**2) / jnp.sqrt(2.0 * jnp.pi)
    price = disc * jnp.sum(payoff * pdf) * dz

    return price
