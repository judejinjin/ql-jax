"""Callable bond instrument and pricing."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class CallableBond:
    """Callable fixed-rate bond.

    Parameters
    ----------
    face_value : par/face value
    coupon_rate : annual coupon rate
    coupon_dates : array of coupon payment dates (year fractions)
    call_dates : dates at which bond can be called
    call_prices : call prices at each call date (typically par)
    maturity : final maturity
    """
    face_value: float
    coupon_rate: float
    coupon_dates: jnp.ndarray
    call_dates: jnp.ndarray
    call_prices: jnp.ndarray
    maturity: float


def callable_bond_price_hw(bond, a, sigma, discount_curve_fn, n_steps=100):
    """Price a callable bond under the Hull-White model using a trinomial tree.

    Simplified implementation using backward induction on a 1D grid.

    Parameters
    ----------
    bond : CallableBond
    a : HW mean reversion
    sigma : HW volatility
    discount_curve_fn : P(0,.)
    n_steps : number of time steps in the tree

    Returns
    -------
    price : callable bond price
    """
    T = bond.maturity
    dt = T / n_steps

    # Build short rate grid
    dr = sigma * jnp.sqrt(3.0 * dt)
    j_max = int(jnp.ceil(0.184 / (a * dt)).item())
    n_nodes = 2 * j_max + 1
    j_range = jnp.arange(-j_max, j_max + 1, dtype=jnp.float64)

    # Short rate at each node: r_j = j * dr + alpha(t)
    # alpha(t) calibrated to fit initial term structure

    # Time grid
    times = jnp.linspace(0, T, n_steps + 1)

    # Initialize terminal values: face + final coupon
    last_coupon = bond.coupon_rate * bond.face_value * dt  # approximate
    values = jnp.full(n_nodes, bond.face_value + last_coupon, dtype=jnp.float64)

    # Transition probabilities for trinomial tree
    def get_probs(j):
        """Return (p_up, p_mid, p_down) for node j."""
        aj = a * j * dt
        p_up = 1.0 / 6.0 + (aj**2 - aj) / 2.0
        p_mid = 2.0 / 3.0 - aj**2
        p_down = 1.0 / 6.0 + (aj**2 + aj) / 2.0
        return p_up, p_mid, p_down

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        t_curr = float(times[step])

        # Compute alpha(t) from market discount factors
        P_market = discount_curve_fn(t_curr + dt)
        P_market_curr = discount_curve_fn(t_curr) if t_curr > 0 else 1.0

        # Approximate alpha
        alpha = -jnp.log(P_market / P_market_curr) / dt

        # Short rates at this time step
        r_nodes = alpha + j_range * dr

        # Discount factors
        df = jnp.exp(-r_nodes * dt)

        # Expected continuation values (trinomial weighting)
        new_values = jnp.zeros(n_nodes, dtype=jnp.float64)
        for j_idx in range(n_nodes):
            j = j_idx - j_max
            p_up, p_mid, p_down = get_probs(j)

            # Clamp neighbor indices
            up_idx = jnp.clip(j_idx + 1, 0, n_nodes - 1)
            mid_idx = j_idx
            down_idx = jnp.clip(j_idx - 1, 0, n_nodes - 1)

            cont = p_up * values[up_idx] + p_mid * values[mid_idx] + p_down * values[down_idx]
            new_values = new_values.at[j_idx].set(df[j_idx] * cont)

        # Add coupons at coupon dates
        for c_date in bond.coupon_dates:
            if abs(float(c_date) - t_curr) < dt / 2:
                coupon = bond.coupon_rate * bond.face_value * dt  # simplified
                new_values = new_values + coupon

        # Apply call constraint
        for i, c_date in enumerate(bond.call_dates):
            if abs(float(c_date) - t_curr) < dt / 2:
                call_price = float(bond.call_prices[i])
                new_values = jnp.minimum(new_values, call_price)

        values = new_values

    # Price at root node
    return values[j_max]
