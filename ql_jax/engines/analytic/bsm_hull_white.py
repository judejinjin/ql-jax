"""BSM-Hull-White hybrid engine: equity under Black-Scholes + Hull-White rates.

Prices European equity options when the short rate follows a Hull-White process
and the equity follows geometric Brownian motion with stochastic interest rates.

Reference: Brigo & Mercurio, "Interest Rate Models — Theory and Practice", Ch. 6.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType


def bsm_hull_white_price(
    S, K, T, r0, q, sigma_s, a_hw, sigma_hw, rho_sr,
    discount_curve_fn=None,
    option_type: int = 1,
):
    """European equity option under BSM + Hull-White short rates.

    Parameters
    ----------
    S : spot equity price
    K : strike
    T : time to expiry
    r0 : initial short rate
    q : continuous dividend yield
    sigma_s : equity volatility
    a_hw : HW mean reversion speed
    sigma_hw : HW volatility
    rho_sr : correlation between equity and rate
    discount_curve_fn : callable(t) -> P(0,t), market discount curve.
        If None, uses flat rate r0.
    option_type : OptionType.Call (1) or OptionType.Put (-1)

    Returns
    -------
    price : European option price
    """
    S, K, T, r0, q, sigma_s, a_hw, sigma_hw, rho_sr = (
        jnp.float64(x) for x in (S, K, T, r0, q, sigma_s, a_hw, sigma_hw, rho_sr)
    )
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    if discount_curve_fn is None:
        P_0_T = jnp.exp(-r0 * T)
    else:
        P_0_T = discount_curve_fn(T)

    F = S * jnp.exp(-q * T) / P_0_T  # Forward price under T-forward measure

    # HW bond volatility contribution
    B_hw = (1.0 - jnp.exp(-a_hw * T)) / a_hw

    # Integrated variance of log(F/P) under hybrid
    var_hw = sigma_hw**2 * (T - 2.0 * B_hw + (1.0 - jnp.exp(-2.0 * a_hw * T)) / (2.0 * a_hw))
    cov_sr = rho_sr * sigma_s * sigma_hw * (T - B_hw)
    total_var = sigma_s**2 * T + var_hw - 2.0 * cov_sr
    total_var = jnp.maximum(total_var, 1e-20)
    sigma_eff = jnp.sqrt(total_var / T)

    # Black formula under T-forward measure
    d1 = (jnp.log(S * jnp.exp(-q * T) / (K * P_0_T)) + 0.5 * total_var) / jnp.sqrt(total_var)
    d2 = d1 - jnp.sqrt(total_var)

    price = phi * P_0_T * (
        F * norm.cdf(phi * d1) - K * norm.cdf(phi * d2)
    )
    return price
