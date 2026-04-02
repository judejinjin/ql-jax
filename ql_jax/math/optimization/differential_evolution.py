"""Differential Evolution global optimizer.

A population-based stochastic optimizer for non-convex problems.
"""

import jax
import jax.numpy as jnp


def minimize(objective, bounds, population_size=50, max_iterations=1000,
             tol=1e-8, F=0.8, CR=0.9, key=None, strategy='best1bin'):
    """Minimize objective using Differential Evolution.

    Parameters
    ----------
    objective : callable(x) -> float
    bounds : array of shape (n, 2) – lower/upper bounds per dimension
    population_size : int
    max_iterations : int
    tol : float – convergence tolerance on objective improvement
    F : float – mutation factor in [0, 2]
    CR : float – crossover probability in [0, 1]
    key : jax PRNGKey (auto-created if None)
    strategy : str – 'best1bin' or 'rand1bin'

    Returns dict with: x, fun, n_iterations, converged.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    bounds = jnp.asarray(bounds, dtype=jnp.float64)
    n_dim = bounds.shape[0]
    lo = bounds[:, 0]
    hi = bounds[:, 1]

    # Initialize population uniformly in bounds
    key, subkey = jax.random.split(key)
    pop = lo + jax.random.uniform(subkey, (population_size, n_dim)) * (hi - lo)

    # Evaluate initial fitness
    fitness = jnp.array([objective(pop[i]) for i in range(population_size)])
    best_idx = jnp.argmin(fitness)
    best_x = pop[best_idx]
    best_f = fitness[best_idx]

    converged = False
    for iteration in range(max_iterations):
        key, *subkeys = jax.random.split(key, 4)

        for i in range(population_size):
            # Select 3 distinct random indices != i
            key, sk = jax.random.split(key)
            idxs = jax.random.choice(sk, population_size, shape=(3,), replace=False)
            # Simple rejection (rare collision)
            r0, r1, r2 = int(idxs[0]), int(idxs[1]), int(idxs[2])

            if strategy == 'best1bin':
                mutant = best_x + F * (pop[r0] - pop[r1])
            else:  # rand1bin
                mutant = pop[r0] + F * (pop[r1] - pop[r2])

            # Crossover
            key, sk = jax.random.split(key)
            cross_mask = jax.random.uniform(sk, (n_dim,)) < CR
            key, sk = jax.random.split(key)
            j_rand = jax.random.randint(sk, (), 0, n_dim)
            cross_mask = cross_mask.at[j_rand].set(True)

            trial = jnp.where(cross_mask, mutant, pop[i])
            # Clip to bounds
            trial = jnp.clip(trial, lo, hi)

            trial_f = objective(trial)
            if trial_f <= fitness[i]:
                pop = pop.at[i].set(trial)
                fitness = fitness.at[i].set(trial_f)
                if trial_f < best_f:
                    best_x = trial
                    best_f = trial_f

        # Check convergence
        spread = jnp.max(fitness) - jnp.min(fitness)
        if spread < tol:
            converged = True
            break

    return {
        'x': best_x,
        'fun': best_f,
        'n_iterations': iteration + 1,
        'converged': converged,
    }
