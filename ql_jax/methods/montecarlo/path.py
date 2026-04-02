"""Monte Carlo path generation for stochastic processes.

Generates vectorized paths using JAX random number generation and vmap.
Supports single-factor (BS) and multi-factor (Heston) processes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def generate_paths_bs(process, T, n_steps, n_paths, key, antithetic=False):
    """Generate log-spot paths for a BlackScholesProcess.

    Parameters
    ----------
    process : BlackScholesProcess
    T : time horizon
    n_steps : number of time steps
    n_paths : number of paths
    key : jax.random.PRNGKey
    antithetic : if True, generate antithetic paths (doubles effective paths)

    Returns
    -------
    paths : (n_eff_paths, n_steps+1) array of log-spot values
        If antithetic, n_eff_paths = 2 * n_paths
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)

    if antithetic:
        z = jax.random.normal(key, shape=(n_paths, n_steps))
        z = jnp.concatenate([z, -z], axis=0)
    else:
        z = jax.random.normal(key, shape=(n_paths, n_steps))

    dw = z * sqrt_dt

    log_s0 = jnp.log(jnp.asarray(process.spot, dtype=jnp.float64))

    def step_fn(log_s, dw_i):
        log_s_new = process.evolve(0.0, log_s, dt, dw_i)
        return log_s_new, log_s_new

    def simulate_one(dw_path):
        _, log_path = jax.lax.scan(step_fn, log_s0, dw_path)
        return jnp.concatenate([jnp.array([log_s0]), log_path])

    paths = jax.vmap(simulate_one)(dw)
    return paths


def generate_paths_heston(process, T, n_steps, n_paths, key, antithetic=False):
    """Generate (log-spot, variance) paths for a HestonProcess.

    Parameters
    ----------
    process : HestonProcess
    T : time horizon
    n_steps : number of time steps
    n_paths : number of paths
    key : jax.random.PRNGKey
    antithetic : if True, generate antithetic paths

    Returns
    -------
    log_spot_paths : (n_eff_paths, n_steps+1) array
    var_paths : (n_eff_paths, n_steps+1) array
    """
    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)

    key1, key2 = jax.random.split(key)

    if antithetic:
        z1 = jax.random.normal(key1, shape=(n_paths, n_steps))
        z2 = jax.random.normal(key2, shape=(n_paths, n_steps))
        z1 = jnp.concatenate([z1, -z1], axis=0)
        z2 = jnp.concatenate([z2, -z2], axis=0)
    else:
        z1 = jax.random.normal(key1, shape=(n_paths, n_steps))
        z2 = jax.random.normal(key2, shape=(n_paths, n_steps))

    dw_s = z1 * sqrt_dt
    dw_v = z2 * sqrt_dt

    log_s0 = jnp.log(jnp.asarray(process.spot, dtype=jnp.float64))
    v0 = jnp.asarray(process.v0, dtype=jnp.float64)

    def step_fn(state, noise):
        dw_s_i, dw_v_i = noise
        new_state = process.evolve(0.0, state, dt, (dw_s_i, dw_v_i))
        return new_state, new_state

    def simulate_one(dw_s_path, dw_v_path):
        _, states = jax.lax.scan(
            step_fn, (log_s0, v0),
            (dw_s_path, dw_v_path),
        )
        log_s_traj, v_traj = states
        log_s_full = jnp.concatenate([jnp.array([log_s0]), log_s_traj])
        v_full = jnp.concatenate([jnp.array([v0]), v_traj])
        return log_s_full, v_full

    log_spot_paths, var_paths = jax.vmap(simulate_one)(dw_s, dw_v)
    return log_spot_paths, var_paths


def time_grid(T, n_steps):
    """Create a uniform time grid [0, dt, 2*dt, ..., T]."""
    return jnp.linspace(0.0, T, n_steps + 1)
