"""Markov Functional short-rate model.

A 1-factor model where the numeraire and rates are expressed as
functions of a single Markov process (typically Gaussian):
  dx = -a*x dt + sigma dW

The mapping from x to rates/numeraire is calibrated to match a set of
market prices (typically swaptions at each exercise date).
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def build_numeraire_surface(a, sigma, exercise_dates, swap_dates_list,
                             market_vols, discount_curve_fn,
                             n_grid=64, n_std=7.0):
    """Build the numeraire N(t, x) mapping on a grid.

    For each exercise date, the model calibrates:
      N(t_i, x) such that swaption prices match Black vols.

    Parameters
    ----------
    a : mean reversion
    sigma : volatility of x
    exercise_dates : array of swaption expiry dates
    swap_dates_list : list of swap date arrays (one per exercise)
    market_vols : array of Black swaption vols
    discount_curve_fn : P(0, .)
    n_grid : grid size for x
    n_std : grid extent in std deviations

    Returns dict with: x_grid, numeraire_values (2D), exercise_dates.
    """
    n_exercises = len(exercise_dates)
    numeraire_grid = []

    for i in range(n_exercises):
        t_i = exercise_dates[i]
        V_t = sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t_i))
        sigma_x = jnp.sqrt(jnp.maximum(V_t, 1e-20))

        x_grid = jnp.linspace(-n_std * sigma_x, n_std * sigma_x, n_grid)

        # For each grid point, find N(t_i, x) from market swaption vol
        swap_dates = swap_dates_list[i]
        n = len(swap_dates) - 1
        vol = market_vols[i]
        strike = _atm_swap_rate(swap_dates, discount_curve_fn)

        # N(t_i, x) = P(t_i, T_n)/annuity at each x
        # Calibrated iteratively; for now use Gaussian 1D as approximation
        B_vals = jnp.array([(1.0 - jnp.exp(-a * (Tj - t_i))) / a for Tj in swap_dates])
        P_0_Ti = discount_curve_fn(t_i)

        N_vals = jnp.zeros(n_grid)
        for j in range(n_grid):
            x = x_grid[j]
            # Approximate bond prices
            P_bonds = jnp.array([
                discount_curve_fn(Tk) / P_0_Ti * jnp.exp(-B_vals[k] * x - 0.5 * B_vals[k]**2 * V_t)
                for k, Tk in enumerate(swap_dates)
            ])
            annuity = sum((swap_dates[k + 1] - swap_dates[k]) * P_bonds[k + 1] for k in range(n))
            N_vals = N_vals.at[j].set(P_bonds[n] / (annuity + 1e-30))

        numeraire_grid.append(N_vals)

    return {
        'x_grid': jnp.linspace(-n_std * sigma, n_std * sigma, n_grid),
        'numeraire_values': jnp.array(numeraire_grid),
        'exercise_dates': jnp.array(exercise_dates),
        'a': a,
        'sigma': sigma,
    }


def _atm_swap_rate(swap_dates, discount_curve_fn):
    """ATM par swap rate."""
    n = len(swap_dates) - 1
    P = jnp.array([discount_curve_fn(T) for T in swap_dates])
    annuity = sum((swap_dates[i + 1] - swap_dates[i]) * P[i + 1] for i in range(n))
    return (P[0] - P[n]) / annuity


def price_bermudan(numeraire_data, discount_curve_fn,
                    swap_dates, strike, is_payer=True, n_grid=64, n_std=7.0):
    """Price a Bermudan swaption by backward induction on the MF model.

    Parameters
    ----------
    numeraire_data : dict from build_numeraire_surface
    discount_curve_fn : P(0, .)
    swap_dates : full swap schedule
    strike : fixed rate
    is_payer : True for payer

    Returns float – Bermudan swaption price.
    """
    a = numeraire_data['a']
    sigma = numeraire_data['sigma']
    exercise_dates = numeraire_data['exercise_dates']
    n_ex = len(exercise_dates)

    # Build grid at final date
    t_final = exercise_dates[-1]
    V_final = sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t_final))
    sigma_x = jnp.sqrt(jnp.maximum(V_final, 1e-20))
    x_grid = jnp.linspace(-n_std * sigma_x, n_std * sigma_x, n_grid)

    # Terminal values
    values = jnp.zeros(n_grid)

    # Backward induction
    for i in range(n_ex - 1, -1, -1):
        t_i = exercise_dates[i]
        V_i = sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t_i))
        sig_i = jnp.sqrt(jnp.maximum(V_i, 1e-20))

        # Exercise value at t_i
        exercise = jnp.zeros(n_grid)
        B_vals = jnp.array([(1.0 - jnp.exp(-a * (Tj - t_i))) / a for Tj in swap_dates])
        P_0_ti = discount_curve_fn(t_i)

        for j in range(n_grid):
            x = x_grid[j]
            P_bonds = jnp.array([
                discount_curve_fn(Tk) / P_0_ti * jnp.exp(-B_vals[k] * x - 0.5 * B_vals[k]**2 * V_i)
                for k, Tk in enumerate(swap_dates)
            ])
            n_swap = len(swap_dates) - 1
            floating = P_bonds[0] - P_bonds[n_swap]
            fixed = strike * sum((swap_dates[k + 1] - swap_dates[k]) * P_bonds[k + 1] for k in range(n_swap))
            swap_npv = floating - fixed if is_payer else fixed - floating
            exercise = exercise.at[j].set(jnp.maximum(swap_npv, 0.0))

        # Continuation: rollback
        if i < n_ex - 1:
            dt = exercise_dates[i + 1] - t_i
            decay = jnp.exp(-a * dt)
            V_cond = sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * dt))
            sig_cond = jnp.sqrt(jnp.maximum(V_cond, 1e-20))

            cont = jnp.zeros(n_grid)
            for j in range(n_grid):
                # Expected continuation value E[V(t_{i+1}, x') | x(t_i) = x_grid[j]]
                mean_x = x_grid[j] * decay
                weights = norm.pdf(x_grid, mean_x, sig_cond)
                weights = weights / (jnp.sum(weights) + 1e-30)
                cont = cont.at[j].set(jnp.dot(weights, values))

            values = jnp.maximum(exercise, cont)
        else:
            values = exercise

    # Price at t=0: integrate over initial distribution
    V_0 = sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * exercise_dates[0]))
    sig_0 = jnp.sqrt(jnp.maximum(V_0, 1e-20))
    weights = norm.pdf(x_grid, 0.0, sig_0)
    weights = weights / (jnp.sum(weights) + 1e-30)
    return jnp.dot(weights, values) * discount_curve_fn(exercise_dates[0])
