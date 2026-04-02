"""Analytic quanto option pricing."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def quanto_vanilla_price(S, K, T, r_d, r_f, q, sigma_s, sigma_fx, rho,
                          fx_rate, option_type='call'):
    """Quanto vanilla option price.

    Option on foreign asset S, struck and settled in domestic currency.
    The FX rate is fixed (quanto adjustment).

    Parameters
    ----------
    S : float – foreign asset spot price
    K : float – strike (in foreign currency)
    T : float – maturity
    r_d : float – domestic risk-free rate
    r_f : float – foreign risk-free rate
    q : float – dividend yield on foreign asset
    sigma_s : float – volatility of foreign asset
    sigma_fx : float – FX volatility
    rho : float – correlation between asset and FX
    fx_rate : float – fixed (quanto) FX rate
    option_type : 'call' or 'put'

    Returns
    -------
    price : float (in domestic currency)
    """
    S, K, T = (jnp.asarray(x, dtype=jnp.float64) for x in (S, K, T))

    # Quanto drift adjustment
    r_quanto = r_f - q - rho * sigma_s * sigma_fx

    d1 = (jnp.log(S / K) + (r_quanto + 0.5 * sigma_s**2) * T) / (sigma_s * jnp.sqrt(T))
    d2 = d1 - sigma_s * jnp.sqrt(T)

    phi = 1.0 if option_type == 'call' else -1.0
    price = fx_rate * jnp.exp(-r_d * T) * phi * (
        S * jnp.exp(r_quanto * T) * norm.cdf(phi * d1) -
        K * norm.cdf(phi * d2)
    )
    return price


def quanto_forward_price(S, T, r_d, r_f, q, sigma_s, sigma_fx, rho, fx_rate):
    """Quanto forward price of foreign asset.

    Parameters
    ----------
    Same as quanto_vanilla_price but without K and option_type.

    Returns
    -------
    price : float (in domestic currency)
    """
    r_quanto = r_f - q - rho * sigma_s * sigma_fx
    return fx_rate * S * jnp.exp((r_quanto - r_d) * T)
