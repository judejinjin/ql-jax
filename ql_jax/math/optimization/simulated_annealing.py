"""Simulated Annealing global optimizer."""

import jax
import jax.numpy as jnp


def minimize(objective, x0, bounds=None, max_iterations=10000,
             T_init=1.0, T_min=1e-8, cooling_rate=0.995,
             step_size=0.1, key=None):
    """Minimize objective using Simulated Annealing.

    Parameters
    ----------
    objective : callable(x) -> float
    x0 : array – initial guess
    bounds : optional array (n, 2) – lower/upper bounds
    max_iterations : int
    T_init : float – initial temperature
    T_min : float – minimum temperature (stopping criterion)
    cooling_rate : float – multiplicative cooling factor
    step_size : float – std dev of Gaussian perturbation
    key : jax PRNGKey

    Returns dict with: x, fun, n_iterations, converged.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    x = jnp.asarray(x0, dtype=jnp.float64)
    n_dim = len(x)
    f_current = objective(x)
    best_x = x
    best_f = f_current
    T = T_init

    for iteration in range(max_iterations):
        if T < T_min:
            break

        # Propose new candidate
        key, sk1, sk2 = jax.random.split(key, 3)
        perturbation = step_size * jax.random.normal(sk1, (n_dim,))
        x_new = x + perturbation

        if bounds is not None:
            bounds_arr = jnp.asarray(bounds)
            x_new = jnp.clip(x_new, bounds_arr[:, 0], bounds_arr[:, 1])

        f_new = objective(x_new)
        delta = f_new - f_current

        # Metropolis acceptance
        accept = delta < 0
        if not accept:
            p_accept = jnp.exp(-delta / T)
            u = jax.random.uniform(sk2)
            accept = u < p_accept

        if accept:
            x = x_new
            f_current = f_new
            if f_current < best_f:
                best_x = x
                best_f = f_current

        T *= cooling_rate

    return {
        'x': best_x,
        'fun': best_f,
        'n_iterations': iteration + 1,
        'converged': T < T_min,
    }
