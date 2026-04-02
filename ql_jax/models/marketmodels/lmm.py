"""LIBOR Market Model (BGM/LMM) implementation.

Simulates forward LIBOR rates under the spot measure using
vectorized JAX operations for efficient path generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class LMMConfig:
    """LMM configuration.

    Parameters
    ----------
    n_rates : number of forward rates
    tenors : array of tenor start times (T_0, T_1, ..., T_n)
    tau : accrual fraction for each period
    initial_forwards : initial forward rates F_i(0)
    volatilities : instantaneous vol sigma_i(t) — simplified as constant per rate
    correlation : (n_rates, n_rates) correlation matrix
    """
    n_rates: int
    tenors: jnp.ndarray
    tau: jnp.ndarray
    initial_forwards: jnp.ndarray
    volatilities: jnp.ndarray
    correlation: jnp.ndarray


def simulate_lmm_paths(config, n_paths, n_steps_per_period=10, key=None):
    """Simulate forward rate paths under the LMM spot measure.

    Uses log-Euler discretization with drift correction.

    Parameters
    ----------
    config : LMMConfig
    n_paths : number of paths
    n_steps_per_period : time steps per tenor period
    key : PRNG key

    Returns
    -------
    forward_rates : (n_paths, n_rates, n_total_steps+1) array
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n = config.n_rates
    total_steps = n * n_steps_per_period
    dt = float(config.tau[0]) / n_steps_per_period  # assume uniform tau

    # Cholesky decomposition of correlation
    L = jnp.linalg.cholesky(config.correlation)

    # Generate correlated normals: (n_paths, total_steps, n_rates)
    z = jax.random.normal(key, shape=(n_paths, total_steps, n))
    dW = jnp.einsum('ij,ptj->pti', L, z) * jnp.sqrt(dt)

    # Initialize forward rates
    log_f = jnp.log(config.initial_forwards)  # (n_rates,)
    log_f = jnp.broadcast_to(log_f, (n_paths, n))  # (n_paths, n_rates)

    # Store paths
    all_log_f = [log_f]

    sigma = config.volatilities
    tau = config.tau

    for step in range(total_steps):
        # Current time determines which rates are still alive
        t = step * dt

        # Drift correction under spot measure:
        # d log F_i = (-sigma_i^2/2 + sum_{j=eta(t)}^{i} tau_j * sigma_i * sigma_j * rho_ij * F_j / (1 + tau_j * F_j)) dt + sigma_i dW_i
        f = jnp.exp(log_f)

        # Compute drift for each forward rate
        drift = jnp.zeros_like(log_f)
        for i in range(n):
            # Which rates are in the drift sum depends on the numeraire
            # Under spot measure, sum from eta(t) to i
            eta = int(t / float(tau[0]))  # index of current period
            d_i = -0.5 * sigma[i]**2
            for j in range(min(eta, n), i + 1):
                d_i = d_i + tau[j] * sigma[i] * sigma[j] * config.correlation[i, j] * f[:, j] / (1.0 + tau[j] * f[:, j])
            drift = drift.at[:, i].set(d_i)

        # Euler step in log-space
        log_f = log_f + drift * dt + sigma[None, :] * dW[:, step, :]

        all_log_f.append(log_f)

    return jnp.exp(jnp.stack(all_log_f, axis=2))  # (n_paths, n_rates, total_steps+1)


def lmm_swap_rate(forward_rates, tau):
    """Compute par swap rate from forward rates.

    S = (1 - P(0,T_n)) / sum(tau_i * P(0,T_i))
    where P(0,T_i) = prod_{j=0}^{i-1} 1/(1 + tau_j * F_j)

    Parameters
    ----------
    forward_rates : array of forward rates F_i
    tau : accrual fractions

    Returns
    -------
    par swap rate
    """
    n = forward_rates.shape[0]
    discount = 1.0
    annuity = 0.0

    for i in range(n):
        discount = discount / (1.0 + tau[i] * forward_rates[i])
        annuity = annuity + tau[i] * discount

    return (1.0 - discount) / annuity


def lmm_caplet_price(config, T_i, K, n_paths=50000, key=None):
    """Price a caplet on forward rate F_i via LMM simulation.

    Parameters
    ----------
    config : LMMConfig
    T_i : caplet expiry (index)
    K : strike
    n_paths : MC paths
    key : PRNG key

    Returns
    -------
    price, stderr
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    i = T_i
    paths = simulate_lmm_paths(config, n_paths, key=key)

    # Forward rate at expiry
    step_at_expiry = i * 10  # n_steps_per_period=10
    F_i_T = paths[:, i, step_at_expiry]

    tau_i = config.tau[i]
    payoff = jnp.maximum(F_i_T - K, 0.0) * tau_i

    # Discount using realized forwards
    discount = jnp.ones(n_paths)
    for j in range(i + 1):
        F_j = paths[:, j, j * 10]  # forward rate j at its reset
        discount = discount / (1.0 + config.tau[j] * F_j)

    discounted = discount * payoff
    price = jnp.mean(discounted)
    stderr = jnp.std(discounted) / jnp.sqrt(jnp.float64(n_paths))

    return price, stderr
