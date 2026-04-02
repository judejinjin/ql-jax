"""Analytic cap/floor engine under Hull-White model."""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.models.shortrate.hull_white import hull_white_caplet_price


def hull_white_capfloor_price(cap_floor, r0, a, sigma, discount_curve_fn):
    """Price a cap/floor using Hull-White analytic caplet formula.

    Parameters
    ----------
    cap_floor : CapFloor instrument
    r0 : current short rate
    a : HW mean reversion
    sigma : HW volatility
    discount_curve_fn : P(0,.)

    Returns
    -------
    price
    """
    n = cap_floor.n_periods
    K = cap_floor.strike
    N = cap_floor.notional
    omega = 1.0 if cap_floor.is_cap else -1.0

    total = 0.0
    for i in range(n):
        T_reset = float(cap_floor.reset_dates[i])
        T_pay = float(cap_floor.payment_dates[i])

        if cap_floor.is_cap:
            caplet = hull_white_caplet_price(r0, a, sigma, K, T_reset, T_pay, discount_curve_fn, N)
        else:
            # Floor = Cap - (Forward - Strike) by parity; or price put directly
            # Floorlet = call on ZCB
            tau = T_pay - T_reset
            X = 1.0 + K * tau
            P_reset = discount_curve_fn(T_reset)
            P_pay = discount_curve_fn(T_pay)
            B_val = (1.0 - jnp.exp(-a * tau)) / a
            sigma_p = sigma * B_val * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T_reset)) / (2.0 * a))

            from jax.scipy.stats import norm
            d1 = jnp.log(P_pay / (X * P_reset)) / sigma_p + 0.5 * sigma_p
            d2 = d1 - sigma_p
            # Call on ZCB = floorlet
            call = P_pay * norm.cdf(d1) - X * P_reset * norm.cdf(d2)
            caplet = N * call

        total = total + caplet

    return total
