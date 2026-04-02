"""Heston stochastic volatility model with calibration support.

Wraps the Heston engine (Phase 4) and adds calibration to market option prices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.engines.analytic.heston import heston_price
from ql_jax.models.calibration import calibrate_least_squares, CalibrationResult


def heston_model_prices(params, S, r, q, strikes, maturities, option_types):
    """Compute Heston option prices for a set of instruments.

    Parameters
    ----------
    params : (v0, kappa, theta, xi, rho)
    S : spot price
    r : risk-free rate
    q : dividend yield
    strikes : array of strikes
    maturities : array of maturities
    option_types : array of option types

    Returns
    -------
    prices : array of option prices
    """
    v0, kappa, theta, xi, rho = params

    def price_one(K, T, otype):
        return heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, otype)

    prices = jax.vmap(price_one)(strikes, maturities, option_types)
    return prices


def calibrate_heston(
    S, r, q,
    market_strikes, market_maturities, market_prices,
    option_types=None,
    initial_params=None,
    max_iter=100,
):
    """Calibrate Heston model to market option prices.

    Parameters
    ----------
    S : spot
    r : risk-free rate
    q : dividend yield
    market_strikes : array of strikes
    market_maturities : array of maturities
    market_prices : array of market prices
    option_types : array of OptionType (default: all calls)
    initial_params : (v0, kappa, theta, xi, rho) initial guess
    max_iter : max calibration iterations

    Returns
    -------
    CalibrationResult with params = (v0, kappa, theta, xi, rho)
    """
    strikes = jnp.asarray(market_strikes, dtype=jnp.float64)
    maturities = jnp.asarray(market_maturities, dtype=jnp.float64)
    prices = jnp.asarray(market_prices, dtype=jnp.float64)

    if option_types is None:
        option_types = jnp.full(strikes.shape, OptionType.Call, dtype=jnp.int32)
    else:
        option_types = jnp.asarray(option_types, dtype=jnp.int32)

    if initial_params is None:
        initial_params = jnp.array([0.04, 2.0, 0.04, 0.5, -0.5])

    def model_fn(params):
        return heston_model_prices(params, S, r, q, strikes, maturities, option_types)

    bounds = [
        (1e-4, 1.0),    # v0
        (1e-4, 10.0),   # kappa
        (1e-4, 1.0),    # theta
        (1e-4, 3.0),    # xi
        (-0.99, 0.99),  # rho
    ]

    return calibrate_least_squares(
        model_fn, prices, None, initial_params,
        param_bounds=bounds, max_iter=max_iter,
    )
