"""Correlation models for market models.

Parametric correlation structures for forward rates.
"""

import jax.numpy as jnp


def exponential_correlation(n, beta):
    """Exponential decay correlation: rho(i,j) = exp(-beta * |T_i - T_j|).

    Parameters
    ----------
    n : int – number of rates
    beta : float – decay parameter (>= 0)

    Returns (n, n) correlation matrix.
    """
    idx = jnp.arange(n)
    return jnp.exp(-beta * jnp.abs(idx[:, None] - idx[None, :]))


def time_homogeneous_correlation(n, long_corr, beta, gamma):
    """Time-homogeneous forward correlation (Rebonato):

    rho(i,j) = long_corr + (1-long_corr) * exp(-beta * |T_i - T_j|)

    Parameters
    ----------
    n : number of rates
    long_corr : asymptotic correlation
    beta : decay speed
    gamma : not used (placeholder for extensions)
    """
    idx = jnp.arange(n, dtype=jnp.float64)
    diff = jnp.abs(idx[:, None] - idx[None, :])
    return long_corr + (1.0 - long_corr) * jnp.exp(-beta * diff)


def historical_correlation(returns_matrix):
    """Estimate correlation from historical data.

    Parameters
    ----------
    returns_matrix : array (n_obs, n_rates) – historical return observations

    Returns (n_rates, n_rates) sample correlation matrix.
    """
    # Demean
    centered = returns_matrix - jnp.mean(returns_matrix, axis=0)
    cov = centered.T @ centered / (returns_matrix.shape[0] - 1)
    stds = jnp.sqrt(jnp.diag(cov))
    outer_stds = stds[:, None] * stds[None, :]
    return cov / (outer_stds + 1e-30)


def rank_reduced_correlation(corr_matrix, rank):
    """Rank-reduce a correlation matrix via eigendecomposition.

    Keeps top `rank` eigenvalues and renormalizes.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(corr_matrix)
    # Keep top `rank` eigenvalues
    idx = jnp.argsort(eigenvalues)[::-1][:rank]
    L = eigenvectors[:, idx] * jnp.sqrt(eigenvalues[idx])
    approx = L @ L.T

    # Renormalize diagonal to 1
    d = jnp.sqrt(jnp.diag(approx))
    return approx / (d[:, None] * d[None, :])
