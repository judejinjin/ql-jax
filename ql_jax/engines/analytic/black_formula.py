"""Black-Scholes formula and calculator — the core of option pricing.

All functions are pure, JIT-compatible, and differentiable via jax.grad.
Greeks are obtained automatically via AD rather than closed-form derivations.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType


# ---------------------------------------------------------------------------
# Black-Scholes-Merton closed-form
# ---------------------------------------------------------------------------

def black_scholes_price(S, K, T, r, q, sigma, option_type: int):
    """Black-Scholes-Merton price for European options.

    Parameters
    ----------
    S : float or array — spot price
    K : float or array — strike price
    T : float or array — time to expiry (years)
    r : float or array — risk-free rate (cc)
    q : float or array — dividend yield (cc)
    sigma : float or array — volatility
    option_type : int — OptionType.Call (1) or OptionType.Put (-1)

    Returns
    -------
    price : same shape as inputs
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    price = phi * (S * jnp.exp(-q * T) * norm.cdf(phi * d1)
                   - K * jnp.exp(-r * T) * norm.cdf(phi * d2))
    return price


def black_price(F, K, T, sigma, df, option_type: int):
    """Black's formula for options on forwards/futures.

    Parameters
    ----------
    F : forward price
    K : strike
    T : time to expiry
    sigma : volatility
    df : discount factor exp(-r*T)
    option_type : int
    """
    F, K, T, sigma, df = (jnp.asarray(x, dtype=jnp.float64)
                           for x in (F, K, T, sigma, df))
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    return df * phi * (F * norm.cdf(phi * d1) - K * norm.cdf(phi * d2))


def bachelier_price(F, K, T, sigma_n, df, option_type: int):
    """Bachelier (normal) model price.

    Parameters
    ----------
    sigma_n : normal volatility (absolute, not relative)
    """
    F, K, T, sigma_n, df = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (F, K, T, sigma_n, df))
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    stddev = sigma_n * jnp.sqrt(T)
    d = phi * (F - K) / stddev
    return df * (phi * (F - K) * norm.cdf(d) + stddev * norm.pdf(d))


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def implied_volatility_black_scholes(
    price, S, K, T, r, q, option_type: int,
    guess: float = 0.2,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> float:
    """Solve for BS implied volatility using Newton's method with AD.

    Parameters
    ----------
    price : observed market price
    Returns
    -------
    implied vol (float)
    """
    import jax

    def _price_error(sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type) - price

    _vega = jax.grad(_price_error)

    sigma = jnp.float64(guess)
    for _ in range(max_iter):
        f = _price_error(sigma)
        if jnp.abs(f) < tol:
            break
        v = _vega(sigma)
        if jnp.abs(v) < 1e-20:
            break
        sigma = sigma - f / v
        sigma = jnp.clip(sigma, 1e-6, 10.0)
    return float(sigma)


# ---------------------------------------------------------------------------
# Greeks via AD (convenience wrappers)
# ---------------------------------------------------------------------------

def bs_greeks(S, K, T, r, q, sigma, option_type: int) -> dict:
    """Compute all Black-Scholes Greeks via automatic differentiation.

    Returns dict with: price, delta, gamma, vega, theta, rho.
    """
    import jax

    S, K, T, r, q, sigma = (jnp.float64(x) for x in (S, K, T, r, q, sigma))

    def _price(s, k, t, rate, div, vol):
        return black_scholes_price(s, k, t, rate, div, vol, option_type)

    price = _price(S, K, T, r, q, sigma)
    delta = jax.grad(_price, argnums=0)(S, K, T, r, q, sigma)
    gamma = jax.grad(jax.grad(_price, argnums=0), argnums=0)(S, K, T, r, q, sigma)
    vega = jax.grad(_price, argnums=5)(S, K, T, r, q, sigma)
    theta = -jax.grad(_price, argnums=2)(S, K, T, r, q, sigma)
    rho = jax.grad(_price, argnums=3)(S, K, T, r, q, sigma)

    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
