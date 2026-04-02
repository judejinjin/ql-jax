"""G2++ two-factor short-rate model.

r(t) = x(t) + y(t) + phi(t)
dx = -a*x*dt + sigma*dW1
dy = -b*y*dt + eta*dW2
<dW1, dW2> = rho*dt

phi(t) chosen to fit the initial term structure.
"""

from __future__ import annotations

import jax.numpy as jnp


def g2pp_bond_price(x, y, a, b_, sigma, eta, rho, t, T, discount_curve_fn):
    """Zero-coupon bond price P(t,T) under G2++ model.

    Parameters
    ----------
    x, y : current state variables
    a, b_ : mean reversion speeds for x, y
    sigma, eta : volatilities for x, y
    rho : correlation
    t : current time
    T : maturity
    discount_curve_fn : P_market(0, .)

    Returns
    -------
    P(t,T)
    """
    tau = T - t

    V = _V(a, b_, sigma, eta, rho, t, T)
    V_0 = _V(a, b_, sigma, eta, rho, 0.0, T)
    V_0t = _V(a, b_, sigma, eta, rho, 0.0, t)

    P_0_T = discount_curve_fn(T)
    P_0_t = discount_curve_fn(t)

    B_a = (1.0 - jnp.exp(-a * tau)) / a
    B_b = (1.0 - jnp.exp(-b_ * tau)) / b_

    ln_A = jnp.log(P_0_T / P_0_t) + 0.5 * (V - V_0 + V_0t)

    return jnp.exp(ln_A - B_a * x - B_b * y)


def _V(a, b_, sigma, eta, rho, t, T):
    """Helper: variance integral for G2++."""
    tau = T - t

    term1 = (sigma**2 / a**2) * (tau + (2.0 / a) * jnp.exp(-a * tau) - (1.0 / (2.0 * a)) * jnp.exp(-2.0 * a * tau) - 3.0 / (2.0 * a))
    term2 = (eta**2 / b_**2) * (tau + (2.0 / b_) * jnp.exp(-b_ * tau) - (1.0 / (2.0 * b_)) * jnp.exp(-2.0 * b_ * tau) - 3.0 / (2.0 * b_))
    term3 = 2.0 * rho * sigma * eta / (a * b_) * (
        tau
        + (jnp.exp(-a * tau) - 1.0) / a
        + (jnp.exp(-b_ * tau) - 1.0) / b_
        - (jnp.exp(-(a + b_) * tau) - 1.0) / (a + b_)
    )

    return term1 + term2 + term3


def g2pp_swaption_price(
    a, b_, sigma, eta, rho,
    discount_curve_fn,
    exercise_time, swap_tenors, fixed_rate, notional=1.0,
    n_grid=50,
):
    """European swaption price under G2++ via numerical integration.

    Uses the semi-analytic formula integrating over one state variable.

    Parameters
    ----------
    a, b_ : mean reversion speeds
    sigma, eta : volatilities
    rho : correlation
    discount_curve_fn : P(0, .)
    exercise_time : exercise date
    swap_tenors : swap payment dates
    fixed_rate : fixed leg rate
    notional : notional
    n_grid : number of integration points

    Returns
    -------
    swaption price
    """
    swap_tenors = jnp.asarray(swap_tenors, dtype=jnp.float64)
    n = swap_tenors.shape[0]
    taus = jnp.diff(jnp.concatenate([jnp.array([exercise_time]), swap_tenors]))
    coupons = fixed_rate * taus
    coupons = coupons.at[-1].add(1.0)

    T_ex = exercise_time

    # Variance of x(T_ex)
    var_x = sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * T_ex))
    std_x = jnp.sqrt(var_x)

    # Integration over x
    x_grid = jnp.linspace(-4 * std_x, 4 * std_x, n_grid)
    dx = x_grid[1] - x_grid[0]

    from jax.scipy.stats import norm as norm_dist

    def integrand(x_val):
        # For each x, find y* that makes swap value = 0
        # Then sum bond option values
        B_a_vals = (1.0 - jnp.exp(-a * (swap_tenors - T_ex))) / a
        B_b_vals = (1.0 - jnp.exp(-b_ * (swap_tenors - T_ex))) / b_

        # Market-implied A values
        P_0_T_vals = jnp.array([discount_curve_fn(swap_tenors[i]) for i in range(n)])
        P_0_ex = discount_curve_fn(T_ex)

        V_T = jnp.array([_V(a, b_, sigma, eta, rho, T_ex, swap_tenors[i]) for i in range(n)])

        ln_A_vals = jnp.log(P_0_T_vals / P_0_ex) + 0.5 * V_T

        # Swap value = sum c_i * A_i * exp(-B_a_i * x - B_b_i * y) - 1
        # Find y* where this = 0 via Newton iteration
        def swap_val(y_val):
            bond_prices = jnp.exp(ln_A_vals - B_a_vals * x_val - B_b_vals * y_val)
            return jnp.sum(coupons * bond_prices) - 1.0

        # Simple Newton
        y_val = 0.0
        for _ in range(20):
            sv = swap_val(y_val)
            dsv = jnp.sum(-coupons * B_b_vals * jnp.exp(ln_A_vals - B_a_vals * x_val - B_b_vals * y_val))
            y_val = y_val - sv / dsv

        # Mean and variance of y conditional on x
        mu_y = rho * sigma * eta / (a * b_) * (1.0 - jnp.exp(-a * T_ex)) * (1.0 - jnp.exp(-b_ * T_ex))  # approximate
        var_y = eta**2 / (2.0 * b_) * (1.0 - jnp.exp(-2.0 * b_ * T_ex))
        std_y = jnp.sqrt(var_y)

        # Swaption payoff = max(swap_value, 0) integrated over y given x
        # Each bond option in the decomposition
        total = 0.0
        for i in range(n):
            kappa_i = -B_b_vals[i]
            # Strike y for bond i
            X_i = jnp.exp(ln_A_vals[i] - B_a_vals[i] * x_val - B_b_vals[i] * y_val)
            P_i = jnp.exp(ln_A_vals[i] - B_a_vals[i] * x_val)
            # This is a call/put on exp(-B_b * y)
            d_i = (y_val - 0.0) / std_y  # using mean 0 for simplicity
            total = total + coupons[i] * (
                P_i * norm_dist.cdf(B_b_vals[i] * std_y - d_i)
                - X_i * norm_dist.cdf(-d_i)
            )

        # Weight by normal density of x
        weight = norm_dist.pdf(x_val / std_x) / std_x
        return total * weight

    # Numerical integration via trapezoidal rule
    values = jnp.array([integrand(x_grid[i]) for i in range(n_grid)])
    price = jnp.trapezoid(values, x_grid)

    return notional * price
