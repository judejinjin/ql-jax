"""Asian option instrument and analytic pricing (geometric average)."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType


@dataclass(frozen=True)
class AsianOption:
    """Asian (average price) option."""
    strike: float
    option_type: int
    exercise: object = None
    averaging_type: str = "geometric"  # 'geometric' or 'arithmetic'


def analytic_continuous_geometric_asian_price(
    S, K, T, r, q, sigma, option_type: int,
):
    """Analytic price for continuous geometric average Asian option.

    Closed-form solution: geometric average of GBM is itself log-normal.

    sigma_a = sigma / sqrt(3)
    r_a = 0.5 * (r - q - sigma^2/6) + sigma_a^2/2
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))

    sigma_a = sigma / jnp.sqrt(3.0)
    b = 0.5 * (r - q - sigma**2 / 6.0)

    d1 = (jnp.log(S / K) + (b + 0.5 * sigma_a**2) * T) / (sigma_a * jnp.sqrt(T))
    d2 = d1 - sigma_a * jnp.sqrt(T)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    price = (phi * (S * jnp.exp((b - r) * T) * norm.cdf(phi * d1)
                    - K * jnp.exp(-r * T) * norm.cdf(phi * d2)))
    return price


def analytic_discrete_geometric_asian_price(
    S, K, T, r, q, sigma, option_type: int,
    n_fixings: int,
):
    """Analytic price for discrete geometric average Asian option.

    Parameters
    ----------
    n_fixings : number of averaging observations (equally spaced)
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    n = jnp.float64(n_fixings)
    dt = T / n

    # Adjusted parameters for discrete geometric average
    sigma_a = sigma * jnp.sqrt((2.0 * n + 1.0) / (6.0 * (n + 1.0)))
    b = ((r - q - 0.5 * sigma**2) * (n + 1.0) / (2.0 * n)
         + 0.5 * sigma_a**2)

    d1 = (jnp.log(S / K) + (b + 0.5 * sigma_a**2) * T) / (sigma_a * jnp.sqrt(T))
    d2 = d1 - sigma_a * jnp.sqrt(T)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    price = (phi * (S * jnp.exp((b - r) * T) * norm.cdf(phi * d1)
                    - K * jnp.exp(-r * T) * norm.cdf(phi * d2)))
    return price
