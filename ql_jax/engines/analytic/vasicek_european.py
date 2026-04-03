"""European option pricing under the Vasicek short-rate model.

The equity option is priced assuming the underlying follows a lognormal
process with stochastic interest rates given by the Vasicek/Hull-White model.

Reference: Rabinovitch (1989), "Pricing stock and bond options when the
default-free rate is stochastic".
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType


def vasicek_european_price(
    S, K, T, r0, q, sigma_s,
    a, b, sigma_r, rho,
    option_type: int = 1,
):
    """European equity option with Vasicek stochastic rates.

    dr = a(b - r) dt + sigma_r dW_r
    dS/S = (r - q) dt + sigma_s dW_s
    dW_s dW_r = rho dt

    Parameters
    ----------
    S : spot equity price
    K : strike
    T : time to expiry
    r0 : initial short rate
    q : continuous dividend yield
    sigma_s : equity volatility
    a : Vasicek mean reversion speed
    b : Vasicek long-run mean rate
    sigma_r : Vasicek rate volatility
    rho : correlation(dS, dr)
    option_type : OptionType.Call (1) or OptionType.Put (-1)

    Returns
    -------
    price : option price
    """
    S, K, T, r0, q, sigma_s = (jnp.float64(x) for x in (S, K, T, r0, q, sigma_s))
    a_v, b_v, sigma_r_v, rho_v = (jnp.float64(x) for x in (a, b, sigma_r, rho))
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    # Vasicek bond price P(0,T)
    B_val = (1.0 - jnp.exp(-a_v * T)) / a_v
    A_val = jnp.exp(
        (b_v - sigma_r_v**2 / (2.0 * a_v**2)) * (B_val - T)
        - sigma_r_v**2 / (4.0 * a_v) * B_val**2
    )
    P_0_T = A_val * jnp.exp(-B_val * r0)

    F = S * jnp.exp(-q * T) / P_0_T

    # Total variance
    var_r = sigma_r_v**2 * (T + (1.0 - jnp.exp(-2.0 * a_v * T)) / (2.0 * a_v)
                             - 2.0 * B_val)
    cov_sr = rho_v * sigma_s * sigma_r_v * (T - B_val)
    total_var = sigma_s**2 * T + var_r - 2.0 * cov_sr
    total_var = jnp.maximum(total_var, 1e-20)
    sigma_eff = jnp.sqrt(total_var)

    d1 = (jnp.log(F / K) + 0.5 * total_var) / sigma_eff
    d2 = d1 - sigma_eff

    price = phi * P_0_T * (F * norm.cdf(phi * d1) - K * norm.cdf(phi * d2))
    return price
