"""Analytic variance swap pricing."""

import jax.numpy as jnp


def variance_swap_fair_strike(S, T, r, q, sigma, option_prices_fn=None):
    """Fair variance strike for a variance swap.

    K_var = (2/T) * [rT - (S*exp(rT)/S_0 - 1) + integral of OTM options]

    If no option prices function is provided, uses the simple
    Black-Scholes fair variance = sigma^2.

    Parameters
    ----------
    S : float – spot
    T : float – maturity
    r : float – risk-free rate
    q : float – dividend yield
    sigma : float – implied vol (for simple case)
    option_prices_fn : callable(K) -> (call_price, put_price) or None

    Returns
    -------
    K_var : float – fair variance strike
    """
    if option_prices_fn is None:
        # Under BS, fair variance = sigma^2
        return sigma**2

    # Model-free replication
    F = S * jnp.exp((r - q) * T)

    # Strip of OTM options
    n_strikes = 200
    K_low = F * 0.1
    K_high = F * 3.0
    strikes = jnp.linspace(K_low, K_high, n_strikes)
    dK = strikes[1] - strikes[0]

    integral = 0.0
    for i in range(n_strikes):
        K = strikes[i]
        call_p, put_p = option_prices_fn(K)
        # Use puts below forward, calls above
        price = jnp.where(K < F, put_p, call_p)
        integral += price / K**2 * dK

    K_var = (2.0 / T) * jnp.exp(r * T) * integral
    return K_var


def variance_swap_npv(notional, realized_var, strike_var, T, r):
    """NPV of a variance swap.

    Parameters
    ----------
    notional : float – variance notional
    realized_var : float – realized variance (annualized)
    strike_var : float – variance strike K_var
    T : float – remaining time to maturity
    r : float – risk-free rate

    Returns
    -------
    npv : float
    """
    return notional * (realized_var - strike_var) * jnp.exp(-r * T)


def discrete_realized_variance(log_returns, T, annualize=True):
    """Compute discrete realized variance from log returns.

    Parameters
    ----------
    log_returns : array – daily log returns
    T : float – total period in years
    annualize : bool

    Returns
    -------
    variance : float
    """
    n = len(log_returns)
    rv = jnp.sum(log_returns**2)
    if annualize:
        rv = rv / T
    else:
        rv = rv / n
    return rv


def vol_swap_fair_strike(S, T, r, q, sigma):
    """Fair volatility strike (approximate).

    Approximate: K_vol ≈ sqrt(K_var) - 1/(8 * K_var^(3/2)) * var_of_var

    For BS: K_vol = sigma * (1 - sigma^2 * T / (8 * n)) approximately.
    Simple approximation: K_vol ≈ sigma.

    Parameters
    ----------
    S, T, r, q, sigma : float

    Returns
    -------
    K_vol : float
    """
    K_var = sigma**2
    # Convexity adjustment (approximate)
    return jnp.sqrt(K_var)
