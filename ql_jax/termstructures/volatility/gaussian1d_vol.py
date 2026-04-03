"""Gaussian 1-factor swaption volatility surface.

Derives swaption implied vols from a calibrated Gaussian 1-factor
(Hull-White) short-rate model.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm


def gaussian1d_swaption_vol(exercise_time, swap_tenor, mean_reversion,
                            sigma_r, discount_curve_fn):
    """Swaption implied (normal) volatility from Gaussian 1-factor model.

    Parameters
    ----------
    exercise_time : option expiry
    swap_tenor : swap tenor (years)
    mean_reversion : HW mean reversion
    sigma_r : HW volatility
    discount_curve_fn : P(0,t) function

    Returns
    -------
    normal_vol : normal (Bachelier) implied vol of the swap rate
    """
    T = float(exercise_time)
    a = float(mean_reversion)
    sig = float(sigma_r)

    # Approximate: normal vol of swap rate ≈ sigma_r * dS/dr * annuity-weighted
    # Bond duration factor
    swap_end = T + swap_tenor
    B_T = (1.0 - jnp.exp(-a * swap_tenor)) / a

    # Variance of short rate
    var_r = sig**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * T))
    std_r = jnp.sqrt(var_r)

    # Swap rate sensitivity to r: approximately B(T, T+tenor) / annuity
    n_coupons = max(int(swap_tenor), 1)
    P_vals = [float(discount_curve_fn(T + (i + 1))) for i in range(n_coupons)]
    annuity = sum(P_vals)

    # Normal vol ≈ sigma_r * B / annuity * some factor
    # Simplified: use the variance of the swap rate
    normal_vol = std_r * float(B_T) / max(annuity / n_coupons, 1e-10)

    return normal_vol


def gaussian1d_smile_section(strike, forward, exercise_time, swap_tenor,
                              mean_reversion, sigma_r, discount_curve_fn):
    """Lognormal (Black) implied vol from Gaussian 1-factor.

    Parameters
    ----------
    strike : swaption strike
    forward : forward swap rate
    exercise_time, swap_tenor : maturity structure
    mean_reversion, sigma_r : HW params
    discount_curve_fn : discount function

    Returns
    -------
    black_vol : Black (lognormal) implied vol
    """
    nv = gaussian1d_swaption_vol(exercise_time, swap_tenor, mean_reversion,
                                  sigma_r, discount_curve_fn)
    # Convert normal vol to lognormal: sigma_black ≈ sigma_normal / forward
    return float(nv) / max(float(forward), 1e-10)
