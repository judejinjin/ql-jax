"""Drift computation for LIBOR and Swap Market Models.

In the T_N-forward measure, LIBOR forward rate drifts are:
  mu_i(t) = sigma_i(t) * sum_{j=q(t)}^{i} tau_j * f_j * rho_{ij} * sigma_j / (1 + tau_j * f_j)

where q(t) is the index of the next rate to fix after t.
"""

import jax.numpy as jnp


def lmm_drift(forward_rates, volatilities, correlations, accruals, alive_index):
    """Compute LMM drifts in the terminal measure.

    Parameters
    ----------
    forward_rates : array (n,) – current forward rates
    volatilities : array (n,) – instantaneous volatilities sigma_i
    correlations : array (n, n) – instantaneous correlation matrix rho
    accruals : array (n,) – year fractions tau_i
    alive_index : int – index of the first rate that hasn't expired

    Returns array (n,) of drift terms mu_i.
    """
    n = len(forward_rates)
    drifts = jnp.zeros(n)

    for i in range(alive_index, n):
        drift_i = 0.0
        for j in range(alive_index, i + 1):
            tau_f = accruals[j] * forward_rates[j]
            drift_i += tau_f / (1.0 + tau_f) * correlations[i, j] * volatilities[j]
        drifts = drifts.at[i].set(volatilities[i] * drift_i)

    return drifts


def smm_drift(swap_rates, volatilities, correlations, accruals, alive_index):
    """Compute swap market model drifts (approximate).

    In the annuity measure for coterminal swaps, drifts are more complex;
    this uses the frozen-coefficient approximation.
    """
    n = len(swap_rates)
    drifts = jnp.zeros(n)

    for i in range(alive_index, n):
        drift_i = 0.0
        for j in range(alive_index, n):
            if j != i:
                drift_i += correlations[i, j] * volatilities[j] * swap_rates[j]
        drifts = drifts.at[i].set(volatilities[i] * drift_i * accruals[i])

    return drifts


def predictor_corrector_drift(forward_rates_start, forward_rates_end,
                               volatilities, correlations, accruals, alive_index):
    """Predictor-corrector drift: average of drift at start and end rates.

    More accurate than pure Euler — reduces discretization bias.
    """
    drift_start = lmm_drift(forward_rates_start, volatilities, correlations,
                             accruals, alive_index)
    drift_end = lmm_drift(forward_rates_end, volatilities, correlations,
                           accruals, alive_index)
    return 0.5 * (drift_start + drift_end)
