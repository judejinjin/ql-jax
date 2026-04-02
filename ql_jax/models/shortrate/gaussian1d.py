"""Gaussian 1D short-rate model base.

A mean-reverting Gaussian model with time-dependent volatility:
  dx = -a*x dt + sigma(t) dW
  r(t) = x(t) + phi(t)
where phi(t) is deterministic and calibrated to the initial term structure.

This module provides the base Gaussian1D functionality used by GSR
and Markov Functional models.
"""

import jax.numpy as jnp
from jax.scipy.stats import norm


def gaussian1d_bond_price(x, a, sigma, t, T, discount_curve_fn):
    """Zero-coupon bond price P(t,T) in Gaussian 1D model.

    Parameters
    ----------
    x : current state variable
    a : mean reversion
    sigma : volatility (scalar or callable(t)->float)
    t : current time
    T : bond maturity
    discount_curve_fn : P(0, .) initial discount curve
    """
    tau = T - t
    sig = sigma if not callable(sigma) else sigma(t)

    B = (1.0 - jnp.exp(-a * tau)) / a

    P_0_T = discount_curve_fn(T)
    P_0_t = discount_curve_fn(t) if t > 0 else 1.0

    # Forward rate at t
    dt_bump = 1e-4
    P_bump = discount_curve_fn(t + dt_bump)
    f_0_t = -jnp.log(P_bump / P_0_t) / dt_bump if t > 0 else -jnp.log(discount_curve_fn(dt_bump)) / dt_bump

    # Variance of x(t)
    V_t = sig**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t))

    ln_A = jnp.log(P_0_T / P_0_t) + B * f_0_t - 0.5 * B**2 * V_t
    return jnp.exp(ln_A - B * x)


def gaussian1d_swap_rate(x, a, sigma, t, swap_dates, discount_curve_fn):
    """Par swap rate in Gaussian 1D model.

    Parameters
    ----------
    x : state variable
    swap_dates : array of payment dates [T0, T1, ..., Tn]
    """
    n = len(swap_dates) - 1
    P = jnp.array([gaussian1d_bond_price(x, a, sigma, t, Ti, discount_curve_fn)
                    for Ti in swap_dates])

    # Annuity
    annuity = sum((swap_dates[i + 1] - swap_dates[i]) * P[i + 1] for i in range(n))
    return (P[0] - P[n]) / annuity


def gaussian1d_swaption_price(x, a, sigma, t, swap_dates, strike,
                               discount_curve_fn, is_payer=True):
    """Swaption price in Gaussian 1D model using Jamshidian decomposition.

    Parameters
    ----------
    x : state variable
    a, sigma : model parameters
    t : current time (typically 0)
    swap_dates : [T0, ..., Tn] schedule
    strike : fixed rate
    discount_curve_fn : P(0, .)
    is_payer : True for payer swaption
    """
    T_expiry = swap_dates[0]
    n = len(swap_dates) - 1

    # Variance of x at expiry
    sig = sigma if not callable(sigma) else sigma(0.0)
    V_T = sig**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * T_expiry))
    sigma_x = jnp.sqrt(V_T)

    # Find x* such that swap rate = strike (Jamshidian trick)
    from ql_jax.math.solvers.brent import solve

    def swap_rate_minus_strike(x_val):
        return gaussian1d_swap_rate(x_val, a, sigma, T_expiry, swap_dates, discount_curve_fn) - strike

    x_star = solve(swap_rate_minus_strike, -0.5, 0.5)

    # Decompose into bond options
    price = 0.0
    sign = 1.0 if is_payer else -1.0

    for i in range(n):
        tau_i = swap_dates[i + 1] - swap_dates[i]
        B_i = (1.0 - jnp.exp(-a * (swap_dates[i + 1] - T_expiry))) / a
        P_star = gaussian1d_bond_price(x_star, a, sigma, T_expiry,
                                        swap_dates[i + 1], discount_curve_fn)
        K_i = P_star  # Strike for each bond option

        # Bond option via Black formula
        P_0_Ti = discount_curve_fn(swap_dates[i + 1])
        P_0_T0 = discount_curve_fn(T_expiry)
        sigma_bond = B_i * sigma_x

        d1 = jnp.log(P_0_Ti / (K_i * P_0_T0)) / sigma_bond + 0.5 * sigma_bond
        d2 = d1 - sigma_bond

        ci = tau_i * strike
        if i == 0:
            ci -= 1.0  # floating leg notional
        if i == n - 1:
            ci += 1.0 + tau_i * strike

        bond_opt = sign * (P_0_Ti * norm.cdf(sign * d1) - K_i * P_0_T0 * norm.cdf(sign * d2))
        price += ci * bond_opt

    return jnp.abs(price)


def y_grid(a, sigma, t, n_points=64, n_std=7.0):
    """Generate integration grid for state variable.

    Returns array of y values centered at 0 with n_std standard deviations.
    """
    sig = sigma if not callable(sigma) else sigma(0.0)
    V_t = sig**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t))
    std = jnp.sqrt(jnp.maximum(V_t, 1e-20))
    return jnp.linspace(-n_std * std, n_std * std, n_points)
