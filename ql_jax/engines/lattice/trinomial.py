"""Trinomial tree pricing for vanilla and exotic options.

Supports CRR-like trinomial, Kamrad-Ritchken, and Tian trinomial.
Uses jax.lax.fori_loop for efficient backward induction.
"""

import jax.numpy as jnp
import jax

from ql_jax._util.types import OptionType


def trinomial_price(
    S, K, T, r, q, sigma, option_type: int,
    n_steps: int = 100,
    american: bool = False,
    stretch: float = jnp.sqrt(3.0),
):
    """Price a vanilla option using a trinomial tree.

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : OptionType.Call or OptionType.Put
    n_steps : number of time steps
    american : allow early exercise
    stretch : stretch parameter (√3 for standard, adjustable)

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    dt = T / n_steps
    n = n_steps

    # Trinomial parameters
    dx = sigma * jnp.sqrt(stretch * dt)
    nu = r - q - 0.5 * sigma**2
    disc = jnp.exp(-r * dt)

    # Transition probabilities
    p_u = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    p_d = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    p_m = 1.0 - (sigma**2 * dt + nu**2 * dt**2) / dx**2

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    # Terminal nodes: at step n, nodes range from -n to +n (2n+1 nodes)
    j = jnp.arange(-n, n + 1, dtype=jnp.float64)
    spots = S * jnp.exp(j * dx)
    values = jnp.maximum(phi * (spots - K), 0.0)

    # Pre-compute index array (static)
    all_j = jnp.arange(-n, n + 1)  # shape (2n+1,)

    def step_fn(i, values):
        step = n - 1 - i  # current step (going backwards)
        # Mask: active nodes at this step are |j| <= step
        mask = jnp.abs(all_j) <= step

        # Backward induction using shifted full arrays
        v_up = jnp.roll(values, -1)
        v_mid = values
        v_down = jnp.roll(values, 1)

        hold = disc * (p_u * v_up + p_m * v_mid + p_d * v_down)

        if american:
            j_vals = all_j.astype(jnp.float64)
            spots_step = S * jnp.exp(j_vals * dx)
            exercise = jnp.maximum(phi * (spots_step - K), 0.0)
            hold = jnp.maximum(hold, exercise)

        return jnp.where(mask, hold, 0.0)

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return values[n]  # Center node


def trinomial_barrier_price(
    S, K, T, r, q, sigma, barrier, option_type: int,
    is_up_and_out: bool = True,
    rebate: float = 0.0,
    n_steps: int = 100,
    stretch: float = jnp.sqrt(3.0),
):
    """Price a barrier option using a trinomial tree.

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    barrier : float – barrier level
    option_type : int
    is_up_and_out : bool
    rebate : float
    n_steps : int
    stretch : float

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    dt = T / n_steps
    n = n_steps

    dx = sigma * jnp.sqrt(stretch * dt)
    nu = r - q - 0.5 * sigma**2
    disc = jnp.exp(-r * dt)

    p_u = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    p_d = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    p_m = 1.0 - (sigma**2 * dt + nu**2 * dt**2) / dx**2

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)

    # Terminal payoff with barrier
    j = jnp.arange(-n, n + 1, dtype=jnp.float64)
    spots = S * jnp.exp(j * dx)
    payoff = jnp.maximum(phi * (spots - K), 0.0)

    # Apply barrier at terminal
    if is_up_and_out:
        knocked = spots >= barrier
    else:
        knocked = spots <= barrier
    values = jnp.where(knocked, rebate * disc**(0), payoff)

    # Pre-compute index array (static)
    all_j = jnp.arange(-n, n + 1)  # shape (2n+1,)
    all_spots = S * jnp.exp(all_j.astype(jnp.float64) * dx)

    def step_fn(i, values):
        step = n - 1 - i
        mask = jnp.abs(all_j) <= step

        v_up = jnp.roll(values, -1)
        v_mid = values
        v_down = jnp.roll(values, 1)

        hold = disc * (p_u * v_up + p_m * v_mid + p_d * v_down)

        # Apply barrier
        if is_up_and_out:
            knocked_step = all_spots >= barrier
        else:
            knocked_step = all_spots <= barrier
        hold = jnp.where(knocked_step, rebate, hold)

        return jnp.where(mask, hold, 0.0)

    values = jax.lax.fori_loop(0, n, step_fn, values)
    return values[n]
