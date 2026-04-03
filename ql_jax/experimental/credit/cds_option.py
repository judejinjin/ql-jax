"""CDS option — option on a credit default swap."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def black_cds_option_price(forward_spread, strike_spread, vol, T, annuity,
                           option_type=1):
    """Black-formula CDS option price.

    Parameters
    ----------
    forward_spread : forward CDS spread
    strike_spread : strike spread (fixed coupon)
    vol : spread volatility
    T : option expiry
    annuity : risky annuity (risky PV01)
    option_type : 1 for payer (right to buy protection), -1 for receiver

    Returns
    -------
    price : CDS option price
    """
    forward_spread = jnp.asarray(forward_spread, dtype=jnp.float64)
    strike_spread = jnp.asarray(strike_spread, dtype=jnp.float64)
    vol = jnp.asarray(vol, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    annuity = jnp.asarray(annuity, dtype=jnp.float64)
    phi = jnp.asarray(option_type, dtype=jnp.float64)

    vol_sqrt_T = vol * jnp.sqrt(T)
    d1 = (jnp.log(forward_spread / strike_spread) + 0.5 * vol**2 * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T

    price = phi * annuity * (forward_spread * norm.cdf(phi * d1) -
                              strike_spread * norm.cdf(phi * d2))
    return jnp.maximum(price, 0.0)
