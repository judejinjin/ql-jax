"""Swaption pricing via Bachelier (normal) model."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def bachelier_swaption_price(notional, fixed_rate, swap_rate, annuity,
                               normal_vol, expiry, is_payer=True):
    """Price a European swaption using the Bachelier (normal) model.

    Parameters
    ----------
    notional : float
    fixed_rate : float – swap fixed rate
    swap_rate : float – forward swap rate
    annuity : float – present value of swap annuity
    normal_vol : float – normal (basis point) volatility
    expiry : float – swaption expiry in years
    is_payer : bool

    Returns
    -------
    price : float
    """
    d = (swap_rate - fixed_rate) / (normal_vol * jnp.sqrt(expiry))

    phi = 1.0 if is_payer else -1.0

    price = notional * annuity * normal_vol * jnp.sqrt(expiry) * (
        phi * d * norm.cdf(phi * d) + norm.pdf(d)
    )
    return price


def bachelier_implied_normal_vol(price, notional, fixed_rate, swap_rate,
                                   annuity, expiry, is_payer=True):
    """Implied normal vol from swaption price via Newton's method.

    Parameters
    ----------
    price : float – market swaption price
    notional, fixed_rate, swap_rate, annuity, expiry : float
    is_payer : bool

    Returns
    -------
    normal_vol : float
    """
    # Initial guess from at-the-money approximation
    sigma = price / (notional * annuity * jnp.sqrt(expiry / (2.0 * jnp.pi)))
    sigma = jnp.maximum(sigma, 1e-6)

    for _ in range(20):
        model_price = bachelier_swaption_price(
            notional, fixed_rate, swap_rate, annuity, sigma, expiry, is_payer
        )
        # Vega
        vega = notional * annuity * jnp.sqrt(expiry) * norm.pdf(
            (swap_rate - fixed_rate) / (sigma * jnp.sqrt(expiry))
        )
        sigma = sigma - (model_price - price) / jnp.maximum(vega, 1e-12)
        sigma = jnp.maximum(sigma, 1e-8)

    return sigma
