"""Brownian generators: Sobol Brownian bridge and standard pseudo-random."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def pseudo_random_brownian(key, n_paths, n_steps, n_factors):
    """Generate pseudo-random Brownian increments.

    Parameters
    ----------
    key : JAX PRNG key
    n_paths : int
    n_steps : int
    n_factors : int

    Returns
    -------
    array [n_paths, n_steps, n_factors] of standard normal increments
    """
    return jax.random.normal(key, shape=(n_paths, n_steps, n_factors))


def sobol_brownian_bridge(n_paths, n_steps, n_factors, skip=0):
    """Generate Sobol+Brownian-bridge Brownian increments.

    Uses Sobol low-discrepancy sequence with Brownian bridge construction
    for variance reduction in MC simulation.

    Parameters
    ----------
    n_paths : int (should be power of 2)
    n_steps : int
    n_factors : int
    skip : int (Sobol skip dimension)

    Returns
    -------
    array [n_paths, n_steps, n_factors]
    """
    from ql_jax.math.random.quasi import sobol_sequence
    from jax.scipy.stats import norm as jnorm

    total_dim = n_steps * n_factors
    # Generate Sobol sequence in [0,1]^total_dim
    u = sobol_sequence(total_dim, n_paths, skip=skip)
    # Inverse normal transform
    z = jnorm.ppf(jnp.clip(u, 1e-10, 1.0 - 1e-10))
    z = z.reshape(n_paths, n_steps, n_factors)

    # Brownian bridge reordering
    # Process steps in bridge order: mid-points first, then fill in
    result = jnp.zeros_like(z)
    bridge_order = _brownian_bridge_order(n_steps)

    for idx, (step, left, right, left_weight, right_weight, std_dev) in enumerate(bridge_order):
        if left == -1:
            # First step: just the increment
            result = result.at[:, step, :].set(z[:, idx, :] * jnp.sqrt(float(step + 1)))
        else:
            # Bridge: interpolate and add noise
            w_left = result[:, left, :] if left >= 0 else jnp.zeros((n_paths, n_factors))
            w_right = result[:, right, :] if right < n_steps else jnp.zeros((n_paths, n_factors))
            bridge_mean = left_weight * w_left + right_weight * w_right
            result = result.at[:, step, :].set(bridge_mean + std_dev * z[:, idx, :])

    # Convert cumulative to increments
    increments = jnp.diff(
        jnp.concatenate([jnp.zeros((n_paths, 1, n_factors)), result], axis=1),
        axis=1,
    )
    return increments


def _brownian_bridge_order(n_steps):
    """Compute the order of bridge construction.

    Returns list of (step, left_step, right_step, left_weight, right_weight, std_dev).
    """
    if n_steps <= 1:
        return [(0, -1, -1, 0.0, 0.0, 1.0)]

    order = []
    # Start with endpoint
    order.append((n_steps - 1, -1, -1, 0.0, 0.0, float(n_steps) ** 0.5))

    queue = [(0, n_steps - 1)]
    while queue:
        left, right = queue.pop(0)
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        dt_total = float(right - left)
        dt_left = float(mid - left)
        dt_right = float(right - mid)
        lw = dt_right / dt_total
        rw = dt_left / dt_total
        sd = (dt_left * dt_right / dt_total) ** 0.5
        order.append((mid, left, right, lw, rw, sd))
        queue.append((left, mid))
        queue.append((mid, right))

    return order
