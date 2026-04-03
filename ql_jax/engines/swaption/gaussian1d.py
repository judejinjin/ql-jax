"""Gaussian 1-factor swaption and cap/floor engine.

Prices European swaptions via Jamshidian decomposition and
caplets/floorlets via ZCB option formula under Hull-White 1-factor.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm


def _zcb_option(P_0_T, P_0_ex, sigma_p, X, is_put):
    """Zero-coupon bond option price (Black-like formula).

    Parameters
    ----------
    P_0_T : P(0, maturity)
    P_0_ex : P(0, exercise)
    sigma_p : bond price volatility
    X : strike
    is_put : True for put, False for call
    """
    d1 = jnp.log(P_0_T / (X * P_0_ex)) / sigma_p + 0.5 * sigma_p
    d2 = d1 - sigma_p
    if is_put:
        return X * P_0_ex * norm.cdf(-d2) - P_0_T * norm.cdf(-d1)
    else:
        return P_0_T * norm.cdf(d1) - X * P_0_ex * norm.cdf(d2)


def gaussian1d_swaption_price(
    exercise_time, swap_payment_times, swap_accrual_fractions,
    fixed_rate, notional, payer,
    mean_reversion, sigma_r, discount_curve_fn,
):
    """European swaption price under Gaussian 1-factor (Hull-White) model.

    Uses Jamshidian decomposition: find critical rate r* such that the
    swap value is zero, then decompose into ZCB options.

    Parameters
    ----------
    exercise_time : float
    swap_payment_times : array of payment dates (T_1, ..., T_n)
    swap_accrual_fractions : array of accrual fractions
    fixed_rate : strike rate
    notional : notional amount
    payer : True for payer swaption
    mean_reversion : short-rate mean reversion speed (a)
    sigma_r : short-rate volatility
    discount_curve_fn : callable(t) -> P(0,t)

    Returns
    -------
    price : swaption price
    """
    T_ex = float(exercise_time)
    K = float(fixed_rate)
    N = float(notional)
    a = float(mean_reversion)
    sigma = float(sigma_r)

    t_pay = [float(t) for t in swap_payment_times]
    tau = [float(t) for t in swap_accrual_fractions]
    n = len(t_pay)

    # Coupon amounts: c_i = K*tau_i, c_n += 1 (principal)
    coupons = [K * tau[i] for i in range(n)]
    coupons[-1] += 1.0

    # Market discount factors
    P_0_ex = float(discount_curve_fn(T_ex))
    P_0_Ti = [float(discount_curve_fn(t)) for t in t_pay]

    # B(T_ex, T_i) = (1 - exp(-a*(T_i - T_ex))) / a
    B_vals = [(1.0 - jnp.exp(-a * (t_pay[i] - T_ex))) / a for i in range(n)]
    B_vals = [float(b) for b in B_vals]

    # Bond price volatility: sigma_p_i = sigma * B_i * sqrt((1-exp(-2*a*T_ex))/(2*a))
    vol_factor = float(jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T_ex)) / (2.0 * a)))
    sigma_p_vals = [sigma * B_vals[i] * vol_factor for i in range(n)]

    # Instantaneous forward rate at T_ex (numerical derivative)
    eps = 1e-4
    f_inst = -(jnp.log(discount_curve_fn(T_ex + eps)) - jnp.log(P_0_ex)) / eps
    f_inst = float(f_inst)

    # A(T_ex, T_i) for HW bond price: P_hw(T_ex, T_i; r) = A_i * exp(-B_i * r)
    # ln A_i = ln(P(0,T_i)/P(0,T_ex)) + B_i*f(0,T_ex) - (sigma^2/(4a))*(1-exp(-2aT_ex))*B_i^2
    var_factor = (sigma**2 / (4.0 * a)) * (1.0 - jnp.exp(-2.0 * a * T_ex))
    var_factor = float(var_factor)
    A_vals = []
    for i in range(n):
        lnA = jnp.log(P_0_Ti[i] / P_0_ex) + B_vals[i] * f_inst - var_factor * B_vals[i]**2
        A_vals.append(float(jnp.exp(lnA)))

    # Find r* via Newton: g(r) = sum(c_i * A_i * exp(-B_i * r)) - 1 = 0
    r_star = f_inst
    for _ in range(50):
        g = sum(coupons[i] * A_vals[i] * jnp.exp(-B_vals[i] * r_star) for i in range(n)) - 1.0
        gp = -sum(coupons[i] * A_vals[i] * B_vals[i] * jnp.exp(-B_vals[i] * r_star) for i in range(n))
        g, gp = float(g), float(gp)
        if abs(gp) < 1e-15:
            break
        r_star = r_star - g / gp
        if abs(g) < 1e-12:
            break

    # ZCB strikes: X_i = P_hw(T_ex, T_i; r*) = A_i * exp(-B_i * r*)
    X_vals = [A_vals[i] * float(jnp.exp(-B_vals[i] * r_star)) for i in range(n)]

    # Sum ZCB options via Jamshidian
    is_put = payer  # payer swaption -> puts on ZCBs
    price = 0.0
    for i in range(n):
        zbo = _zcb_option(P_0_Ti[i], P_0_ex, sigma_p_vals[i], X_vals[i], is_put)
        price += coupons[i] * float(zbo)

    return N * price


def gaussian1d_capfloor_price(
    reset_times, payment_times, accrual_fractions,
    strike, notional, is_cap,
    mean_reversion, sigma_r, discount_curve_fn,
):
    """Cap/floor price under Gaussian 1-factor (Hull-White) model.

    Each caplet = (1+K*tau) * ZBP(0, t_fix, t_pay, 1/(1+K*tau))
    Each floorlet = (1+K*tau) * ZBC(0, t_fix, t_pay, 1/(1+K*tau))

    Parameters
    ----------
    reset_times : array of reset (fixing) dates
    payment_times : array of payment dates
    accrual_fractions : array of accrual fractions
    strike : cap/floor strike
    notional : notional
    is_cap : True for cap, False for floor
    mean_reversion : short-rate mean reversion
    sigma_r : rate vol
    discount_curve_fn : P(0,.)

    Returns
    -------
    price : cap/floor price
    """
    a = float(mean_reversion)
    sigma = float(sigma_r)
    K = float(strike)
    N = float(notional)

    price = 0.0
    for i in range(len(reset_times)):
        t_fix = float(reset_times[i])
        t_pay = float(payment_times[i])
        tau_i = float(accrual_fractions[i])

        X = 1.0 / (1.0 + K * tau_i)  # ZCB strike

        P_0_fix = float(discount_curve_fn(t_fix))
        P_0_pay = float(discount_curve_fn(t_pay))

        B_val = float((1.0 - jnp.exp(-a * (t_pay - t_fix))) / a)
        sigma_p = sigma * B_val * float(jnp.sqrt(
            (1.0 - jnp.exp(-2.0 * a * t_fix)) / (2.0 * a)
        ))

        # Caplet = (1+K*tau) * Put on ZCB
        # Floorlet = (1+K*tau) * Call on ZCB
        zbo = _zcb_option(P_0_pay, P_0_fix, sigma_p, X, is_put=is_cap)
        price += N * (1.0 + K * tau_i) * float(zbo)

    return price
