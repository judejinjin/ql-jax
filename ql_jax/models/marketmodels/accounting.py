"""Market model accounting engine and pathwise accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp


@dataclass
class MarketModelProduct:
    """Describes a product in the LMM framework.

    Parameters
    ----------
    n_rates : int — number of forward rates
    evolution_times : array of evolution times
    cashflow_times : array of cashflow payment times
    cashflow_fn : callable (forward_rates_at_t, t_idx) -> array of cashflows
    """
    n_rates: int
    evolution_times: jnp.ndarray
    cashflow_times: jnp.ndarray
    cashflow_fn: Callable


@dataclass
class AccountingEngineResult:
    """Result from the accounting engine."""
    npv_mean: float
    npv_std: float
    npvs: jnp.ndarray  # per-path NPVs


def accounting_engine(
    product: MarketModelProduct,
    initial_rates,
    taus,
    volatilities,
    correlation,
    n_paths=10000,
    key=None,
):
    """Run the accounting engine for a market model product.

    Simulates forward rates under the terminal measure and accumulates
    discounted cashflows.

    Parameters
    ----------
    product : MarketModelProduct
    initial_rates : array of initial forward rates
    taus : array of accrual fractions
    volatilities : 2D array [n_steps x n_rates] of instantaneous vols
    correlation : 2D array [n_rates x n_rates]
    n_paths : int
    key : JAX PRNG key

    Returns
    -------
    AccountingEngineResult
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    n_rates = product.n_rates
    n_steps = len(product.evolution_times) - 1
    initial_rates = jnp.asarray(initial_rates, dtype=jnp.float64)
    taus = jnp.asarray(taus, dtype=jnp.float64)
    volatilities = jnp.asarray(volatilities, dtype=jnp.float64)
    correlation = jnp.asarray(correlation, dtype=jnp.float64)

    # Cholesky for correlated normals
    L = jnp.linalg.cholesky(correlation)

    # Simulate
    rates = jnp.tile(initial_rates, (n_paths, 1))  # [n_paths, n_rates]
    npvs = jnp.zeros(n_paths, dtype=jnp.float64)

    # Numeraire: terminal bond
    numeraire = jnp.ones(n_paths, dtype=jnp.float64)

    for step in range(n_steps):
        dt = product.evolution_times[step + 1] - product.evolution_times[step]
        sqrt_dt = jnp.sqrt(dt)

        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(n_paths, n_rates))
        z_corr = z @ L.T

        # Terminal measure drift
        # volatilities can be 1D (per rate) or 2D (per step x rate)
        if volatilities.ndim == 1:
            vols = volatilities
        else:
            vols = volatilities[step]
        alive = jnp.arange(n_rates) >= step  # rates still alive

        # Drift adjustment (terminal measure)
        drift = jnp.zeros(n_rates, dtype=jnp.float64)
        for j in range(n_rates):
            for k in range(j + 1, n_rates):
                drift = drift.at[j].add(
                    -taus[k] * rates[:, k].mean() * vols[k] * correlation[j, k] /
                    (1.0 + taus[k] * rates[:, k].mean())
                )

        # Euler step for log-rates
        log_rates = jnp.log(jnp.maximum(rates, 1e-10))
        log_rates = log_rates + (drift[None, :] - 0.5 * vols[None, :] ** 2) * dt + vols[None, :] * sqrt_dt * z_corr
        rates = jnp.exp(log_rates) * alive[None, :]

        # Update numeraire
        if step < n_rates:
            numeraire = numeraire * (1.0 + taus[step] * rates[:, step])

        # Compute cashflows at this step
        cashflows = product.cashflow_fn(rates, step)
        if cashflows is not None:
            npvs = npvs + cashflows / numeraire

    return AccountingEngineResult(
        npv_mean=float(jnp.mean(npvs)),
        npv_std=float(jnp.std(npvs) / jnp.sqrt(n_paths)),
        npvs=npvs,
    )


def pathwise_accounting_engine(
    product: MarketModelProduct,
    initial_rates,
    taus,
    volatilities,
    correlation,
    n_paths=10000,
    key=None,
):
    """Pathwise accounting engine with pathwise sensitivities.

    Same as accounting_engine but also computes delta via pathwise method.

    Returns
    -------
    dict with 'npv_mean', 'npv_std', 'deltas' (per-rate sensitivities)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    initial_rates = jnp.asarray(initial_rates, dtype=jnp.float64)

    def npv_fn(rates0):
        result = accounting_engine(product, rates0, taus, volatilities, correlation, n_paths, key)
        return result.npv_mean

    npv = npv_fn(initial_rates)
    deltas = jax.grad(npv_fn)(initial_rates)

    return {
        'npv_mean': float(npv),
        'deltas': deltas,
    }
