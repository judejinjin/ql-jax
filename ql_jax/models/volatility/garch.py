"""Volatility estimation models.

GARCH(1,1), constant, and simple local estimators.
"""

from __future__ import annotations

import jax.numpy as jnp


def garch11_loglikelihood(params, returns):
    """GARCH(1,1) log-likelihood.

    sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}

    Parameters
    ----------
    params : (omega, alpha, beta)
    returns : array of log returns

    Returns
    -------
    log_likelihood : scalar (negative for minimization)
    """
    omega, alpha, beta = params
    n = returns.shape[0]

    # Initialize variance at unconditional level
    var_uncond = omega / (1.0 - alpha - beta)
    var_uncond = jnp.maximum(var_uncond, 1e-10)

    def step(var_prev, r):
        var_new = omega + alpha * r**2 + beta * var_prev
        var_new = jnp.maximum(var_new, 1e-15)
        ll = -0.5 * (jnp.log(var_new) + r**2 / var_new)
        return var_new, ll

    import jax
    _, log_liks = jax.lax.scan(step, var_uncond, returns)
    return -jnp.sum(log_liks)  # negative for minimization


def garch11_forecast(params, returns, n_ahead=1):
    """Forecast variance n_ahead steps using GARCH(1,1).

    Parameters
    ----------
    params : (omega, alpha, beta)
    returns : historical returns
    n_ahead : forecast horizon

    Returns
    -------
    variance forecasts
    """
    omega, alpha, beta = params
    var_uncond = omega / (1.0 - alpha - beta)

    # Get last variance
    import jax
    def step(var_prev, r):
        var_new = omega + alpha * r**2 + beta * var_prev
        return var_new, var_new

    last_var, _ = jax.lax.scan(step, var_uncond, returns)

    # Multi-step forecast
    forecasts = jnp.zeros(n_ahead, dtype=jnp.float64)
    var_t = last_var
    for h in range(n_ahead):
        var_t = omega + (alpha + beta) * var_t
        forecasts = forecasts.at[h].set(var_t)

    return forecasts


def realized_volatility(returns, annualization=252):
    """Simple realized volatility estimator."""
    return jnp.std(returns) * jnp.sqrt(annualization)


def garman_klass_volatility(high, low, close, open_, annualization=252):
    """Garman-Klass volatility estimator using OHLC data.

    Parameters
    ----------
    high, low, close, open_ : arrays of daily OHLC prices

    Returns
    -------
    annualized volatility
    """
    u = jnp.log(high / open_)
    d = jnp.log(low / open_)
    c = jnp.log(close / open_)

    daily_var = 0.5 * (u - d)**2 - (2.0 * jnp.log(2.0) - 1.0) * c**2
    return jnp.sqrt(jnp.mean(daily_var) * annualization)
