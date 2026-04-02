"""Sobol sequence with Brownian bridge construction for path generation."""

import jax
import jax.numpy as jnp
from ql_jax.math.random.quasi import sobol_sequence


def generate_paths(n_paths, n_steps, key=None):
    """Generate correlated Brownian increments using Sobol + Brownian bridge.

    The Brownian bridge reorders the construction of path points to
    improve the low-discrepancy properties of Sobol sequences for
    path-dependent simulations.

    Parameters
    ----------
    n_paths : int – number of paths
    n_steps : int – number of time steps per path

    Returns array of shape (n_paths, n_steps) of standard normal increments.
    """
    # Generate Sobol sequence in [0,1)^n_steps
    uniforms = sobol_sequence(n_paths, n_steps)

    # Transform to normal via inverse CDF
    from jax.scipy.stats import norm
    normals = norm.ppf(jnp.clip(uniforms, 1e-10, 1.0 - 1e-10))

    # Apply Brownian bridge construction
    return _brownian_bridge(normals, n_steps)


def _brownian_bridge(z, n_steps):
    """Apply Brownian bridge ordering to normal variates.

    Parameters
    ----------
    z : array (n_paths, n_steps) – iid standard normal draws
    n_steps : int – number of steps

    Returns array (n_paths, n_steps) of bridged increments.
    """
    n_paths = z.shape[0]
    # Build bridge ordering: first set endpoint, then midpoint, recursively
    order = _bridge_ordering(n_steps)

    # Reconstruct path using bridge
    dt = 1.0 / n_steps
    W = jnp.zeros((n_paths, n_steps + 1))

    # Set the final value first (uses first Sobol dimension)
    W = W.at[:, n_steps].set(z[:, 0] * jnp.sqrt(n_steps * dt))

    # Bridge fill
    bridge_idx = 1
    for left, right in order:
        if bridge_idx >= n_steps:
            break
        mid = (left + right) // 2
        t_l = left * dt
        t_m = mid * dt
        t_r = right * dt

        # Bridge formula: W(t_m) = ((t_r-t_m)*W(t_l) + (t_m-t_l)*W(t_r))/(t_r-t_l)
        #                          + sqrt((t_m-t_l)*(t_r-t_m)/(t_r-t_l)) * z
        mu = ((t_r - t_m) * W[:, left] + (t_m - t_l) * W[:, right]) / (t_r - t_l + 1e-30)
        sigma = jnp.sqrt(jnp.maximum((t_m - t_l) * (t_r - t_m) / (t_r - t_l + 1e-30), 0))
        W = W.at[:, mid].set(mu + sigma * z[:, bridge_idx])
        bridge_idx += 1

    # Convert cumulative to increments
    increments = jnp.diff(W, axis=1)
    return increments / jnp.sqrt(dt)


def _bridge_ordering(n):
    """Generate Brownian bridge bisection ordering.

    Returns list of (left, right) interval pairs to bisect.
    """
    from collections import deque
    pairs = []
    queue = deque([(0, n)])
    while queue:
        left, right = queue.popleft()
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        pairs.append((left, right))
        queue.append((left, mid))
        queue.append((mid, right))
    return pairs
