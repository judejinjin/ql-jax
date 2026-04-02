"""Longstaff-Schwartz method for American option pricing via Monte Carlo.

Backward induction with polynomial regression at each exercise date.
Uses jax.lax.fori_loop for the backward sweep.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def lsm_american_put(paths, K, T, r, n_basis=3):
    """Price an American put using Longstaff-Schwartz.

    Parameters
    ----------
    paths : (n_paths, n_steps+1) array of spot prices (NOT log-spot)
    K : strike
    T : time to expiry
    r : risk-free rate
    n_basis : number of polynomial basis functions

    Returns
    -------
    price : estimated price
    """
    n_paths, n_times = paths.shape
    n_steps = n_times - 1
    dt = T / n_steps
    df = jnp.exp(-r * dt)

    # Payoff at each time for put
    payoff = jnp.maximum(K - paths, 0.0)

    # Cash flows: at each step, what the holder receives if exercising optimally
    # from that step onward
    cashflow = payoff[:, -1]  # terminal payoff
    exercise_time = jnp.full(n_paths, n_steps, dtype=jnp.float64)

    def backward_step(i, state):
        cashflow, exercise_time = state
        step = n_steps - 1 - i  # goes from n_steps-1 down to 1

        S_t = paths[:, step]
        exercise_value = payoff[:, step]
        itm = exercise_value > 0.0

        # Discount cashflow one step
        continuation = cashflow * df

        # Regression: estimated continuation value for ITM paths
        basis = _polynomial_basis(S_t, n_basis)

        # Weighted least squares without forming N×N diagonal matrix:
        # weight basis and continuation by sqrt(itm indicator)
        w = jnp.where(itm, 1.0, 0.0)[:, None]  # (n_paths, 1)
        basisW = basis * w        # zero out non-ITM rows
        contW = continuation * w[:, 0]

        # Solve: (basisW^T basisW) coeffs = basisW^T contW
        coeffs = jnp.linalg.lstsq(basisW, contW, rcond=None)[0]
        est_continuation = basis @ coeffs

        # Exercise if exercise value > estimated continuation and ITM
        do_exercise = itm & (exercise_value >= est_continuation)

        # Update cashflow and exercise time
        cashflow = jnp.where(do_exercise, exercise_value, continuation)
        exercise_time = jnp.where(do_exercise, jnp.float64(step), exercise_time)

        return (cashflow, exercise_time)

    # Run backward from step n_steps-1 to step 1
    n_backward = n_steps - 1
    cashflow, exercise_time = jax.lax.fori_loop(
        0, n_backward, backward_step, (cashflow, exercise_time)
    )

    # Discount to time 0
    discount_steps = exercise_time
    discounted = cashflow * jnp.exp(-r * discount_steps * dt)

    return jnp.mean(discounted)


def lsm_american_option(paths, K, T, r, option_type, n_basis=3):
    """Price an American option using Longstaff-Schwartz.

    Parameters
    ----------
    paths : (n_paths, n_steps+1) array of spot prices
    K : strike
    T : time to expiry
    r : risk-free rate
    option_type : +1 for call, -1 for put
    n_basis : number of polynomial basis functions

    Returns
    -------
    price : estimated price
    """
    phi = jnp.asarray(option_type, dtype=jnp.float64)
    n_paths, n_times = paths.shape
    n_steps = n_times - 1
    dt = T / n_steps
    df = jnp.exp(-r * dt)

    payoff = jnp.maximum(phi * (paths - K), 0.0)

    cashflow = payoff[:, -1]
    exercise_time = jnp.full(n_paths, n_steps, dtype=jnp.float64)

    def backward_step(i, state):
        cashflow, exercise_time = state
        step = n_steps - 1 - i

        S_t = paths[:, step]
        exercise_value = payoff[:, step]
        itm = exercise_value > 0.0

        continuation = cashflow * df

        basis = _polynomial_basis(S_t, n_basis)
        w = jnp.where(itm, 1.0, 0.0)[:, None]
        basisW = basis * w
        contW = continuation * w[:, 0]
        coeffs = jnp.linalg.lstsq(basisW, contW, rcond=None)[0]
        est_continuation = basis @ coeffs

        do_exercise = itm & (exercise_value >= est_continuation)
        cashflow = jnp.where(do_exercise, exercise_value, continuation)
        exercise_time = jnp.where(do_exercise, jnp.float64(step), exercise_time)

        return (cashflow, exercise_time)

    n_backward = n_steps - 1
    cashflow, exercise_time = jax.lax.fori_loop(
        0, n_backward, backward_step, (cashflow, exercise_time)
    )

    discounted = cashflow * jnp.exp(-r * exercise_time * dt)
    return jnp.mean(discounted)


def _polynomial_basis(S, n_basis):
    """Build polynomial basis matrix [1, S, S^2, ..., S^(n_basis-1)].

    Parameters
    ----------
    S : (n_paths,) spot prices
    n_basis : degree + 1

    Returns
    -------
    basis : (n_paths, n_basis) matrix
    """
    powers = jnp.arange(n_basis, dtype=jnp.float64)
    return S[:, None] ** powers[None, :]


def _laguerre_basis(S, n_basis):
    """Build Laguerre polynomial basis (weighted).

    L_0(x) = 1
    L_1(x) = 1 - x
    L_2(x) = 1 - 2x + x^2/2

    Parameters
    ----------
    S : (n_paths,) spot prices (normalized)
    n_basis : number of basis functions

    Returns
    -------
    basis : (n_paths, n_basis) matrix
    """
    x = S
    basis = jnp.ones((S.shape[0], n_basis))
    if n_basis >= 2:
        basis = basis.at[:, 1].set(1.0 - x)
    if n_basis >= 3:
        basis = basis.at[:, 2].set(1.0 - 2.0 * x + x ** 2 / 2.0)
    return basis
