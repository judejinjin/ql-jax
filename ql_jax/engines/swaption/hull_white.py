"""Hull-White analytic swaption engine."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def hw_swaption_price(notional, fixed_rate, payment_dates, day_fractions,
                       a, sigma, discount_fn, expiry, is_payer=True):
    """Analytic European swaption price under Hull-White.

    Uses Jamshidian's decomposition: the swaption is decomposed into
    a portfolio of zero-coupon bond options.

    Parameters
    ----------
    notional : float
    fixed_rate : float
    payment_dates : array – swap payment dates
    day_fractions : array – accrual fractions
    a : float – mean reversion
    sigma : float – short-rate vol
    discount_fn : callable(t) -> DF
    expiry : float – swaption expiry
    is_payer : bool

    Returns
    -------
    price : float
    """
    n = len(payment_dates)

    # Find the critical short rate r* where swap value = 0
    # Use Newton's method
    r_star = _find_critical_rate(fixed_rate, payment_dates, day_fractions,
                                   a, sigma, discount_fn, expiry)

    # Jamshidian: swaption = sum of ZCB options
    price = 0.0
    for i in range(n):
        T_i = payment_dates[i]
        tau_i = day_fractions[i]

        # Bond option strike
        X_i = _hw_zcb_price_at_rate(r_star, expiry, T_i, a, sigma, discount_fn)

        # Coupon
        c_i = fixed_rate * tau_i * notional
        if i == n - 1:
            c_i += notional  # Principal at last payment

        # ZCB option price
        zcb_opt = hw_bond_option_price(
            expiry, T_i, X_i, a, sigma, discount_fn,
            is_call=is_payer
        )
        price += c_i * zcb_opt

    return price


def hw_bond_option_price(T_opt, T_bond, X, a, sigma, discount_fn,
                          is_call=True):
    """Hull-White zero-coupon bond option price.

    Parameters
    ----------
    T_opt : float – option expiry
    T_bond : float – bond maturity
    X : float – strike price
    a : float – mean reversion
    sigma : float – short-rate vol
    discount_fn : callable(t) -> DF
    is_call : bool

    Returns
    -------
    price : float
    """
    P_T = discount_fn(T_opt)
    P_S = discount_fn(T_bond)

    B = (1.0 - jnp.exp(-a * (T_bond - T_opt))) / a
    sigma_p = sigma * B * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T_opt)) / (2.0 * a))

    d1 = jnp.log(P_S / (P_T * X)) / sigma_p + 0.5 * sigma_p
    d2 = d1 - sigma_p

    if is_call:
        return P_S * norm.cdf(d1) - X * P_T * norm.cdf(d2)
    else:
        return X * P_T * norm.cdf(-d2) - P_S * norm.cdf(-d1)


def _hw_zcb_price_at_rate(r, t, T, a, sigma, discount_fn):
    """HW ZCB price P(t,T) given short rate r at time t."""
    B = (1.0 - jnp.exp(-a * (T - t))) / a
    P_0t = discount_fn(t)
    P_0T = discount_fn(T)
    A = P_0T / P_0t * jnp.exp(
        B * (-jnp.log(P_0t) / t if t > 0 else 0.0) -
        sigma**2 / (4.0 * a) * B**2 * (1.0 - jnp.exp(-2.0 * a * t))
    )
    return A * jnp.exp(-B * r)


def _find_critical_rate(fixed_rate, payment_dates, day_fractions,
                          a, sigma, discount_fn, expiry):
    """Find r* where swap value = 0 at swaption expiry."""
    r_star = 0.03  # initial guess
    n = len(payment_dates)

    for _ in range(20):
        swap_val = 0.0
        swap_deriv = 0.0
        for i in range(n):
            T_i = payment_dates[i]
            tau_i = day_fractions[i]
            B_i = (1.0 - jnp.exp(-a * (T_i - expiry))) / a

            P_i = _hw_zcb_price_at_rate(r_star, expiry, T_i, a, sigma, discount_fn)

            c_i = fixed_rate * tau_i
            if i == n - 1:
                c_i += 1.0

            swap_val += c_i * P_i
            swap_deriv += c_i * (-B_i) * P_i

        swap_val -= 1.0  # Floating leg = 1 at par

        r_star = r_star - swap_val / jnp.where(jnp.abs(swap_deriv) > 1e-12,
                                                  swap_deriv, 1e-12)

    return r_star
