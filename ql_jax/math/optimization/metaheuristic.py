"""Advanced optimization: PSO, firefly, hybrid SA."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


def particle_swarm_optimization(objective, bounds, n_particles=30,
                                 n_iterations=100, w=0.7, c1=1.5, c2=1.5,
                                 seed=42):
    """Particle Swarm Optimization (PSO).

    Parameters
    ----------
    objective : callable, f(x) -> scalar to minimize
    bounds : (n_dim, 2) array of [lower, upper] bounds
    n_particles : swarm size
    n_iterations : max iterations
    w : inertia weight
    c1, c2 : cognitive and social coefficients
    seed : RNG seed

    Returns
    -------
    best_position : (n_dim,) best found position
    best_value : best objective value
    """
    bounds = jnp.asarray(bounds)
    n_dim = bounds.shape[0]
    key = jax.random.PRNGKey(seed)

    # Initialize positions and velocities
    key, k1, k2 = jax.random.split(key, 3)
    positions = bounds[:, 0] + jax.random.uniform(k1, (n_particles, n_dim)) * (bounds[:, 1] - bounds[:, 0])
    velocities = jax.random.uniform(k2, (n_particles, n_dim)) * 0.1 * (bounds[:, 1] - bounds[:, 0])

    # Evaluate initial
    values = jnp.array([objective(positions[i]) for i in range(n_particles)])
    p_best = positions.copy()
    p_best_val = values.copy()
    g_best_idx = jnp.argmin(values)
    g_best = positions[g_best_idx]
    g_best_val = values[g_best_idx]

    for it in range(n_iterations):
        key, k1, k2 = jax.random.split(key, 3)
        r1 = jax.random.uniform(k1, (n_particles, n_dim))
        r2 = jax.random.uniform(k2, (n_particles, n_dim))

        velocities = (w * velocities +
                      c1 * r1 * (p_best - positions) +
                      c2 * r2 * (g_best[None, :] - positions))
        positions = positions + velocities
        positions = jnp.clip(positions, bounds[:, 0], bounds[:, 1])

        values = jnp.array([objective(positions[i]) for i in range(n_particles)])
        improved = values < p_best_val
        p_best = jnp.where(improved[:, None], positions, p_best)
        p_best_val = jnp.where(improved, values, p_best_val)

        new_g_idx = jnp.argmin(p_best_val)
        if p_best_val[new_g_idx] < g_best_val:
            g_best = p_best[new_g_idx]
            g_best_val = p_best_val[new_g_idx]

    return g_best, float(g_best_val)


def firefly_algorithm(objective, bounds, n_fireflies=20, n_iterations=50,
                       alpha=0.5, beta0=1.0, gamma=1.0, seed=42):
    """Firefly Algorithm (nature-inspired metaheuristic).

    Parameters
    ----------
    objective : callable, f(x) -> scalar to minimize
    bounds : (n_dim, 2) array of [lower, upper] bounds
    n_fireflies : population size
    n_iterations : max iterations
    alpha : randomization parameter
    beta0 : attractiveness at distance 0
    gamma : absorption coefficient
    seed : RNG seed

    Returns
    -------
    best_position : (n_dim,) best found position
    best_value : best objective value
    """
    bounds = jnp.asarray(bounds)
    n_dim = bounds.shape[0]
    key = jax.random.PRNGKey(seed)

    key, k1 = jax.random.split(key)
    positions = bounds[:, 0] + jax.random.uniform(k1, (n_fireflies, n_dim)) * (bounds[:, 1] - bounds[:, 0])

    intensities = jnp.array([objective(positions[i]) for i in range(n_fireflies)])

    for it in range(n_iterations):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if intensities[j] < intensities[i]:
                    r = jnp.sqrt(jnp.sum((positions[i] - positions[j])**2))
                    beta = beta0 * jnp.exp(-gamma * r**2)
                    key, k1 = jax.random.split(key)
                    eps = jax.random.uniform(k1, (n_dim,)) - 0.5
                    new_pos = positions[i] + beta * (positions[j] - positions[i]) + alpha * eps
                    new_pos = jnp.clip(new_pos, bounds[:, 0], bounds[:, 1])
                    new_val = objective(new_pos)
                    if new_val < intensities[i]:
                        positions = positions.at[i].set(new_pos)
                        intensities = intensities.at[i].set(new_val)

    best_idx = jnp.argmin(intensities)
    return positions[best_idx], float(intensities[best_idx])


def hybrid_simulated_annealing(objective, x0, bounds, T_init=1.0,
                                T_min=1e-6, cooling=0.95, n_local=10,
                                n_iterations=200, seed=42):
    """Hybrid Simulated Annealing with local search.

    Parameters
    ----------
    objective : callable, f(x) -> scalar to minimize
    x0 : initial position
    bounds : (n_dim, 2) bounds
    T_init : initial temperature
    T_min : minimum temperature
    cooling : cooling factor
    n_local : local search steps per temperature
    n_iterations : max iterations
    seed : RNG seed

    Returns
    -------
    best_position : (n_dim,) best found position
    best_value : best objective value
    """
    bounds = jnp.asarray(bounds)
    n_dim = len(x0)
    x = jnp.asarray(x0)
    fx = objective(x)
    best_x, best_fx = x, fx
    T = T_init
    key = jax.random.PRNGKey(seed)

    for it in range(n_iterations):
        if T < T_min:
            break
        for _ in range(n_local):
            key, k1, k2 = jax.random.split(key, 3)
            step = jax.random.normal(k1, (n_dim,)) * T * (bounds[:, 1] - bounds[:, 0]) * 0.1
            x_new = jnp.clip(x + step, bounds[:, 0], bounds[:, 1])
            fx_new = objective(x_new)
            delta = fx_new - fx
            if delta < 0:
                x, fx = x_new, fx_new
            else:
                p = jnp.exp(-delta / (T + 1e-15))
                if jax.random.uniform(k2) < p:
                    x, fx = x_new, fx_new
            if fx < best_fx:
                best_x, best_fx = x, fx
        T *= cooling

    return best_x, float(best_fx)


def multidim_quadrature(f, bounds, n_points=5):
    """N-dimensional Gauss-Legendre quadrature.

    Parameters
    ----------
    f : callable, f(x) -> scalar where x is (n_dim,)
    bounds : (n_dim, 2) integration bounds
    n_points : quadrature points per dimension

    Returns
    -------
    integral : approximate integral value
    """
    import numpy as np
    bounds = jnp.asarray(bounds)
    n_dim = bounds.shape[0]

    pts, wts = np.polynomial.legendre.leggauss(n_points)
    pts, wts = jnp.array(pts), jnp.array(wts)

    # Tensor product quadrature
    import itertools
    total = 0.0
    for indices in itertools.product(range(n_points), repeat=n_dim):
        x = jnp.array([
            0.5 * (bounds[d, 1] - bounds[d, 0]) * pts[indices[d]] +
            0.5 * (bounds[d, 1] + bounds[d, 0])
            for d in range(n_dim)
        ])
        w = 1.0
        for d in range(n_dim):
            w *= wts[indices[d]] * 0.5 * (bounds[d, 1] - bounds[d, 0])
        total += float(w) * float(f(x))

    return total
