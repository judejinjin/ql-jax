"""European options with known cash dividends (escrowed dividend model).

References:
- Haug, Haug & Lewis (2003), "Back to basics: a new approach to the
  discrete dividend problem"
- Hull, "Options, Futures, and Other Derivatives", escrowed model
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.engines.analytic.black_formula import black_scholes_price


def cash_dividend_european_price(
    S, K, T, r, q, sigma, option_type: int,
    dividend_times=None, dividend_amounts=None,
):
    """European option price with discrete cash dividends.

    Uses the escrowed-dividend model: the spot is adjusted by subtracting
    the PV of future dividends, then standard BS is applied.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : continuous dividend yield (can be 0 if all divs are discrete)
    sigma : volatility
    option_type : OptionType.Call (1) or OptionType.Put (-1)
    dividend_times : array of dividend times (fractions of a year)
    dividend_amounts : array of cash dividend amounts

    Returns
    -------
    price : European option price
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))

    if dividend_times is not None and dividend_amounts is not None:
        div_times = jnp.asarray(dividend_times, dtype=jnp.float64)
        div_amounts = jnp.asarray(dividend_amounts, dtype=jnp.float64)

        # PV of dividends paid before expiry
        mask = div_times < T
        pv_divs = jnp.sum(jnp.where(mask, div_amounts * jnp.exp(-r * div_times), 0.0))
    else:
        pv_divs = jnp.float64(0.0)

    # Adjusted spot
    S_adj = S - pv_divs
    S_adj = jnp.maximum(S_adj, 1e-10)

    return black_scholes_price(S_adj, K, T, r, q, sigma, option_type)
