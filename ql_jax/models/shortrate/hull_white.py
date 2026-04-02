"""Hull-White (extended Vasicek) short-rate model.

dr = [theta(t) - a*r] dt + sigma dW

where theta(t) is chosen to fit the initial term structure.
"""

from __future__ import annotations

import jax.numpy as jnp


def hull_white_bond_price(r, a, sigma, t, T, discount_curve_fn):
    """Zero-coupon bond price P(t,T) under Hull-White.

    Parameters
    ----------
    r : current short rate at time t
    a : mean reversion speed
    sigma : volatility
    t : current time
    T : maturity time
    discount_curve_fn : callable(tau) -> P_market(0, tau)
        The initial market discount curve.

    Returns
    -------
    P(t,T)
    """
    tau = T - t
    B = (1.0 - jnp.exp(-a * tau)) / a

    # Market discount factors
    P_0_T = discount_curve_fn(T)
    P_0_t = discount_curve_fn(t)

    # Forward rate at time t
    dt_bump = 1e-4
    P_0_t_plus = discount_curve_fn(t + dt_bump)
    f_0_t = -jnp.log(P_0_t_plus / P_0_t) / dt_bump

    ln_A = jnp.log(P_0_T / P_0_t) + B * f_0_t - (sigma**2 / (4.0 * a)) * B**2 * (1.0 - jnp.exp(-2.0 * a * t))
    A = jnp.exp(ln_A)

    return A * jnp.exp(-B * r)


def hull_white_short_rate_mean(a, sigma, t, T, discount_curve_fn):
    """Expected short rate E[r(T) | r(t)] under Hull-White (risk-neutral).

    Parameters
    ----------
    a : mean reversion speed
    sigma : volatility
    t : current time
    T : future time
    discount_curve_fn : initial discount curve P(0, .)

    Returns
    -------
    E[r(T)]
    """
    dt_bump = 1e-4
    P_0_T = discount_curve_fn(T)
    P_0_T_plus = discount_curve_fn(T + dt_bump)
    f_0_T = -jnp.log(P_0_T_plus / P_0_T) / dt_bump

    return f_0_T + (sigma**2 / (2.0 * a**2)) * (1.0 - jnp.exp(-a * (T - t)))**2


def hull_white_caplet_price(r0, a, sigma, K, T_reset, T_pay, discount_curve_fn, notional=1.0):
    """Caplet price under Hull-White model.

    Parameters
    ----------
    r0 : current short rate
    a : mean reversion
    sigma : vol
    K : strike
    T_reset, T_pay : dates
    discount_curve_fn : P(0, .)
    notional : notional

    Returns
    -------
    caplet price
    """
    tau = T_pay - T_reset
    X = 1.0 + K * tau  # ZCB strike

    P_0_reset = discount_curve_fn(T_reset)
    P_0_pay = discount_curve_fn(T_pay)

    B_val = (1.0 - jnp.exp(-a * tau)) / a
    sigma_p = sigma * B_val * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T_reset)) / (2.0 * a))

    from jax.scipy.stats import norm
    d1 = jnp.log(P_0_pay / (X * P_0_reset)) / sigma_p + 0.5 * sigma_p
    d2 = d1 - sigma_p

    put = X * P_0_reset * norm.cdf(-d2) - P_0_pay * norm.cdf(-d1)
    return notional * put


def hull_white_swaption_price_jamshidian(
    a, sigma, discount_curve_fn,
    exercise_time, swap_tenors, fixed_rate, notional=1.0,
):
    """European swaption price via Jamshidian decomposition under Hull-White.

    A payer swaption (right to pay fixed, receive floating) is decomposed
    into a portfolio of put options on zero-coupon bonds.

    Parameters
    ----------
    a : mean reversion speed
    sigma : volatility
    discount_curve_fn : P(0, .)
    exercise_time : swaption exercise date
    swap_tenors : array of swap payment dates (T_1, T_2, ..., T_n)
    fixed_rate : fixed leg rate
    notional : notional

    Returns
    -------
    swaption price
    """
    swap_tenors = jnp.asarray(swap_tenors, dtype=jnp.float64)
    n = swap_tenors.shape[0]

    # Period lengths
    taus = jnp.diff(jnp.concatenate([jnp.array([exercise_time]), swap_tenors]))

    # Coupon amounts: c_i = fixed_rate * tau_i, with c_n += 1 (principal)
    coupons = fixed_rate * taus
    coupons = coupons.at[-1].add(1.0)

    # Find r* such that sum_i c_i * P(T_ex, T_i; r*) = 1
    # Use bisection
    def swap_value(r_star):
        total = 0.0
        for i in range(n):
            B_i = (1.0 - jnp.exp(-a * (swap_tenors[i] - exercise_time))) / a
            P_i = hull_white_bond_price(r_star, a, sigma, exercise_time, swap_tenors[i], discount_curve_fn)
            total = total + coupons[i] * P_i
        return total - 1.0

    # Simple bisection for r*
    from ql_jax.math.solvers.brent import solve as brent_solve
    r_star = brent_solve(swap_value, 1e-10, 0.03, x_min=-0.1, x_max=0.5)

    # Bond option volatilities
    P_0_ex = discount_curve_fn(exercise_time)

    from jax.scipy.stats import norm

    total_price = 0.0
    for i in range(n):
        B_i = (1.0 - jnp.exp(-a * (swap_tenors[i] - exercise_time))) / a
        sigma_p_i = sigma * B_i * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * exercise_time)) / (2.0 * a))

        X_i = hull_white_bond_price(r_star, a, sigma, exercise_time, swap_tenors[i], discount_curve_fn)
        P_0_Ti = discount_curve_fn(swap_tenors[i])

        d1 = jnp.log(P_0_Ti / (X_i * P_0_ex)) / sigma_p_i + 0.5 * sigma_p_i
        d2 = d1 - sigma_p_i

        # Put on ZCB
        put_i = X_i * P_0_ex * norm.cdf(-d2) - P_0_Ti * norm.cdf(-d1)
        total_price = total_price + coupons[i] * put_i

    return notional * total_price
