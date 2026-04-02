"""Historical volatility estimators."""

from __future__ import annotations

import jax.numpy as jnp


def constant_estimator(prices, window=252):
    """Constant (close-to-close) historical volatility estimator.

    Parameters
    ----------
    prices : array of prices
    window : lookback window (default 252 trading days)

    Returns
    -------
    float : annualised volatility
    """
    prices = jnp.asarray(prices, dtype=jnp.float64)
    log_returns = jnp.diff(jnp.log(prices))
    n = min(len(log_returns), window)
    return jnp.std(log_returns[-n:]) * jnp.sqrt(252.0)


def garman_klass_estimator(opens, highs, lows, closes, window=252):
    """Garman-Klass historical volatility estimator.

    Uses open, high, low, close data for more efficient vol estimation.

    Parameters
    ----------
    opens, highs, lows, closes : arrays of OHLC prices
    window : lookback window

    Returns
    -------
    float : annualized volatility
    """
    opens = jnp.asarray(opens, dtype=jnp.float64)
    highs = jnp.asarray(highs, dtype=jnp.float64)
    lows = jnp.asarray(lows, dtype=jnp.float64)
    closes = jnp.asarray(closes, dtype=jnp.float64)

    n = min(len(opens), window)
    o, h, l, c = opens[-n:], highs[-n:], lows[-n:], closes[-n:]

    log_hl = jnp.log(h / l)
    log_co = jnp.log(c / o)

    # Garman-Klass formula
    gk = 0.5 * log_hl ** 2 - (2.0 * jnp.log(2.0) - 1.0) * log_co ** 2
    return jnp.sqrt(jnp.mean(gk) * 252.0)


def simple_local_estimator(prices, strikes, window=20):
    """Simple local volatility estimation from price data.

    Estimates local vol at each strike by computing realized vol
    conditioned on spot being near each strike level.

    Parameters
    ----------
    prices : array of spot prices
    strikes : array of strike levels
    window : smoothing window

    Returns
    -------
    array : local vol estimate at each strike
    """
    prices = jnp.asarray(prices, dtype=jnp.float64)
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    log_returns = jnp.diff(jnp.log(prices))

    local_vols = []
    for K in strikes:
        # Weight returns by proximity to strike
        weights = jnp.exp(-0.5 * ((prices[:-1] - K) / (0.1 * K)) ** 2)
        weights = weights / jnp.maximum(jnp.sum(weights), 1e-10)
        weighted_var = jnp.sum(weights * log_returns ** 2)
        local_vols.append(jnp.sqrt(weighted_var * 252.0))

    return jnp.array(local_vols)
