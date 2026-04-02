"""GSR (Gaussian Short Rate) model — piecewise-constant volatility extension.

dx = -a*x dt + sigma(t) dW
r(t) = x(t) + phi(t)

The GSR model is a Gaussian 1-factor model with piecewise-constant
volatility sigma(t). It is used for pricing Bermudan swaptions and
callable bonds via backward induction on a numerical grid.
"""

import jax.numpy as jnp
from jax.scipy.stats import norm


def make_sigma_fn(vol_times, vol_values):
    """Create piecewise-constant volatility function.

    Parameters
    ----------
    vol_times : array – breakpoint times [t1, t2, ...]
    vol_values : array – volatility values [sigma1, sigma2, ...]
        sigma_i applies for t in [t_{i-1}, t_i)
    """
    vol_times = jnp.asarray(vol_times, dtype=jnp.float64)
    vol_values = jnp.asarray(vol_values, dtype=jnp.float64)

    def sigma(t):
        idx = jnp.searchsorted(vol_times, t, side='right')
        idx = jnp.clip(idx, 0, len(vol_values) - 1)
        return vol_values[idx]

    return sigma


def gsr_variance(a, sigma_fn, t, vol_times):
    """Variance of x(t): integral of sigma(s)^2 * e^{-2a(t-s)} ds.

    Computed piecewise for piecewise-constant sigma.
    """
    V = 0.0
    s_prev = 0.0
    for s_next in vol_times:
        s_next = jnp.minimum(s_next, t)
        if s_prev >= t:
            break
        sig = sigma_fn(s_prev)
        V += sig**2 / (2.0 * a) * (jnp.exp(-2.0 * a * (t - s_next)) -
                                     jnp.exp(-2.0 * a * (t - s_prev)))
        s_prev = s_next

    # Last segment
    if s_prev < t:
        sig = sigma_fn(s_prev)
        V += sig**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * (t - s_prev)))

    return V


def gsr_bond_price(x, a, sigma_fn, vol_times, t, T, discount_curve_fn):
    """ZCB price P(t,T) in GSR model."""
    tau = T - t
    B = (1.0 - jnp.exp(-a * tau)) / a

    P_0_T = discount_curve_fn(T)
    P_0_t = discount_curve_fn(t) if t > 0 else 1.0

    dt_bump = 1e-4
    P_bump = discount_curve_fn(t + dt_bump)
    f_0_t = -jnp.log(P_bump / P_0_t) / dt_bump if t > 0 else -jnp.log(discount_curve_fn(dt_bump)) / dt_bump

    V_t = gsr_variance(a, sigma_fn, t, vol_times)
    ln_A = jnp.log(P_0_T / P_0_t) + B * f_0_t - 0.5 * B**2 * V_t

    return jnp.exp(ln_A - B * x)


def gsr_swaption_npv(a, sigma_fn, vol_times, swap_dates, strike,
                      discount_curve_fn, is_payer=True, n_grid=64, n_std=7.0):
    """Price a European swaption using GSR model via numerical integration.

    Parameters
    ----------
    a : mean reversion
    sigma_fn : piecewise-constant vol function
    vol_times : array of vol breakpoints
    swap_dates : [T0, T1, ..., Tn]
    strike : swaption strike
    discount_curve_fn : P(0, .)
    is_payer : True for payer
    n_grid : number of integration points
    n_std : number of std deviations for grid
    """
    T_expiry = swap_dates[0]
    V_T = gsr_variance(a, sigma_fn, T_expiry, vol_times)
    sigma_x = jnp.sqrt(jnp.maximum(V_T, 1e-20))

    # Integration grid
    xs = jnp.linspace(-n_std * sigma_x, n_std * sigma_x, n_grid)
    dx = xs[1] - xs[0]

    npv = 0.0
    for x_val in xs:
        # Swap NPV at this node
        n = len(swap_dates) - 1
        P = jnp.array([gsr_bond_price(x_val, a, sigma_fn, vol_times, T_expiry,
                                        Ti, discount_curve_fn) for Ti in swap_dates])
        floating = P[0] - P[n]
        fixed = strike * sum((swap_dates[i + 1] - swap_dates[i]) * P[i + 1] for i in range(n))

        swap_npv = floating - fixed if is_payer else fixed - floating
        exercise_value = jnp.maximum(swap_npv, 0.0)

        # Probability weight (normal density)
        weight = jnp.exp(-0.5 * (x_val / sigma_x)**2) / (sigma_x * jnp.sqrt(2.0 * jnp.pi))
        npv += exercise_value * weight * dx

    P_0_T0 = discount_curve_fn(T_expiry)
    return npv * P_0_T0


def calibrate_to_swaptions(a, swap_dates_list, strikes, market_prices,
                            vol_times, discount_curve_fn, is_payer=True):
    """Calibrate piecewise-constant sigma to swaption prices.

    Parameters
    ----------
    a : mean reversion (fixed)
    swap_dates_list : list of swap date arrays
    strikes : array of swaption strikes
    market_prices : array of market swaption prices
    vol_times : breakpoint times for sigma
    discount_curve_fn : P(0, .)

    Returns array of calibrated vol values.
    """
    from ql_jax.math.optimization.levenberg_marquardt import minimize

    n_vols = len(vol_times)

    def residuals(log_vols):
        vols = jnp.exp(log_vols)
        sigma_fn = make_sigma_fn(vol_times, vols)
        model_prices = []
        for i in range(len(market_prices)):
            p = gsr_swaption_npv(a, sigma_fn, vol_times, swap_dates_list[i],
                                  strikes[i], discount_curve_fn, is_payer)
            model_prices.append(p)
        return jnp.array(model_prices) - market_prices

    x0 = jnp.log(jnp.full(n_vols, 0.01))
    result = minimize(residuals, x0, max_iterations=100)
    return jnp.exp(result['x'])
