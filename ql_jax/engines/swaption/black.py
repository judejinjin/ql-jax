"""Black swaption pricing engine.

Prices European swaptions using Black's formula on the swap rate.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm


def black_swaption_price(swaption, discount_curve_fn, forward_swap_rate, volatility):
    """Price a European swaption using Black's formula.

    Parameters
    ----------
    swaption : Swaption instrument
    discount_curve_fn : callable(t) -> P(0,t)
    forward_swap_rate : forward par swap rate
    volatility : Black swaption volatility

    Returns
    -------
    price
    """
    K = swaption.fixed_rate
    S = jnp.asarray(forward_swap_rate, dtype=jnp.float64)
    sigma = jnp.asarray(volatility, dtype=jnp.float64)
    T = swaption.exercise_time
    N = swaption.notional
    omega = 1.0 if swaption.payer else -1.0

    # Annuity (PV01)
    annuity = 0.0
    for i in range(swaption.n_periods):
        tau_i = float(swaption.swap_accrual_fractions[i])
        T_i = float(swaption.swap_payment_dates[i])
        annuity = annuity + tau_i * discount_curve_fn(T_i)

    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = N * annuity * omega * (S * norm.cdf(omega * d1) - K * norm.cdf(omega * d2))
    return price


def bachelier_swaption_price(swaption, discount_curve_fn, forward_swap_rate, normal_vol):
    """Price a European swaption using Bachelier (normal) model.

    Parameters
    ----------
    swaption : Swaption instrument
    discount_curve_fn : callable(t) -> P(0,t)
    forward_swap_rate : forward par swap rate
    normal_vol : normal (basis point) volatility

    Returns
    -------
    price
    """
    K = swaption.fixed_rate
    S = jnp.asarray(forward_swap_rate, dtype=jnp.float64)
    sigma_n = jnp.asarray(normal_vol, dtype=jnp.float64)
    T = swaption.exercise_time
    N = swaption.notional
    omega = 1.0 if swaption.payer else -1.0

    annuity = 0.0
    for i in range(swaption.n_periods):
        tau_i = float(swaption.swap_accrual_fractions[i])
        T_i = float(swaption.swap_payment_dates[i])
        annuity = annuity + tau_i * discount_curve_fn(T_i)

    sqrt_T = jnp.sqrt(T)
    d = omega * (S - K) / (sigma_n * sqrt_T)

    price = N * annuity * sigma_n * sqrt_T * (d * norm.cdf(d) + norm.pdf(d))
    return price


def par_swap_rate(discount_curve_fn, exercise_time, payment_dates, accrual_fractions):
    """Compute the forward par swap rate.

    S = (P(0,T0) - P(0,Tn)) / sum(tau_i * P(0,T_i))

    Parameters
    ----------
    discount_curve_fn : callable(t) -> P(0,t)
    exercise_time : swap start date
    payment_dates : fixed leg payment dates
    accrual_fractions : accrual fractions

    Returns
    -------
    par swap rate
    """
    P_start = discount_curve_fn(exercise_time)
    P_end = discount_curve_fn(float(payment_dates[-1]))

    annuity = 0.0
    n = payment_dates.shape[0]
    for i in range(n):
        tau_i = float(accrual_fractions[i])
        T_i = float(payment_dates[i])
        annuity = annuity + tau_i * discount_curve_fn(T_i)

    return (P_start - P_end) / annuity
