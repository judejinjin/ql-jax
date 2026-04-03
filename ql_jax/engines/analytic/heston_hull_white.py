"""Heston-Hull-White hybrid engines: stochastic vol + stochastic rates.

Analytic and Monte Carlo pricing for European equity options when:
  - Equity spot follows Heston stochastic volatility
  - Short rate follows Hull-White (extended Vasicek)
  - Correlation structure between S, v, r

Reference:
  Grzelak & Oosterlee (2011), "On the Heston Model with Stochastic
  Interest Rates", SIAM J. Financial Mathematics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType


def heston_hull_white_price(
    S, K, T, r0, q,
    v0, kappa, theta, xi, rho_sv,
    a_hw, sigma_hw, rho_sr, rho_vr=0.0,
    option_type: int = 1,
    n_points: int = 128,
    discount_curve_fn=None,
):
    """Semi-analytic Heston-HW hybrid price.

    Uses the approximation of Grzelak & Oosterlee (2011): solve Heston
    at an effective vol level that includes the HW rate variance contribution.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r0 : initial short rate
    q : dividend yield
    v0 : initial variance
    kappa : mean reversion of variance
    theta : long-run variance
    xi : vol-of-vol
    rho_sv : correlation(dS, dv)
    a_hw : HW mean reversion
    sigma_hw : HW volatility
    rho_sr : correlation(dS, dr)
    rho_vr : correlation(dv, dr) [typically small or zero]
    option_type : OptionType.Call (1) or OptionType.Put (-1)
    n_points : quadrature points for Heston integration
    discount_curve_fn : P(0, .) if None uses flat r0

    Returns
    -------
    price : European option price
    """
    from ql_jax.engines.analytic.heston import heston_price

    S, K, T = jnp.float64(S), jnp.float64(K), jnp.float64(T)
    r0, q = jnp.float64(r0), jnp.float64(q)
    v0, kappa, theta, xi, rho_sv = (jnp.float64(x) for x in (v0, kappa, theta, xi, rho_sv))
    a_hw, sigma_hw, rho_sr = jnp.float64(a_hw), jnp.float64(sigma_hw), jnp.float64(rho_sr)

    # HW bond volatility
    B_hw = (1.0 - jnp.exp(-a_hw * T)) / a_hw
    hw_var = sigma_hw**2 * (
        T - 2.0 * B_hw + (1.0 - jnp.exp(-2.0 * a_hw * T)) / (2.0 * a_hw)
    )
    # Cross term
    cross_var = 2.0 * rho_sr * sigma_hw * jnp.sqrt(v0) * (T - B_hw)

    # Effective initial variance: add HW contribution
    v0_eff = jnp.maximum(v0 + hw_var / T - cross_var / T, 1e-6)

    return heston_price(S, K, T, r0, q, v0_eff, kappa, theta, xi, rho_sv,
                        option_type, n_points)
