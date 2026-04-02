"""Market model evolvers: Euler and Predictor-Corrector for LMM/SMM.

An evolver advances forward rates one time step in a Monte Carlo simulation.
"""

import jax
import jax.numpy as jnp
from ql_jax.models.marketmodels.drift import lmm_drift, predictor_corrector_drift


def euler_evolve(forward_rates, volatilities, correlations, accruals,
                  alive_index, dt, dw):
    """Euler step for log-normal LMM.

    d(ln f_i) = (mu_i - 0.5*sigma_i^2) dt + sigma_i dW_i

    Parameters
    ----------
    forward_rates : array (n,) – current rates
    volatilities : array (n,) – sigma_i
    correlations : array (n, n) – rho matrix
    accruals : array (n,) – tau_i
    alive_index : int – first live rate
    dt : float – time step
    dw : array (n,) – correlated Brownian increments

    Returns array (n,) of new forward rates.
    """
    n = len(forward_rates)
    drifts = lmm_drift(forward_rates, volatilities, correlations, accruals, alive_index)

    log_f = jnp.log(jnp.maximum(forward_rates, 1e-10))
    for i in range(alive_index, n):
        log_f = log_f.at[i].set(
            log_f[i] + (drifts[i] - 0.5 * volatilities[i]**2) * dt +
            volatilities[i] * dw[i]
        )

    return jnp.exp(log_f)


def predictor_corrector_evolve(forward_rates, volatilities, correlations,
                                 accruals, alive_index, dt, dw):
    """Predictor-corrector step for log-normal LMM.

    Step 1 (predict): Euler with drifts at current rates
    Step 2 (correct): Use average of drifts at current and predicted rates
    """
    n = len(forward_rates)

    # Predict
    f_predicted = euler_evolve(forward_rates, volatilities, correlations,
                                accruals, alive_index, dt, dw)

    # Correct: average drifts
    drifts_corrected = predictor_corrector_drift(
        forward_rates, f_predicted, volatilities, correlations, accruals, alive_index
    )

    log_f = jnp.log(jnp.maximum(forward_rates, 1e-10))
    for i in range(alive_index, n):
        log_f = log_f.at[i].set(
            log_f[i] + (drifts_corrected[i] - 0.5 * volatilities[i]**2) * dt +
            volatilities[i] * dw[i]
        )

    return jnp.exp(log_f)


def generate_correlated_normals(key, n_rates, correlation_matrix, dt):
    """Generate correlated Brownian increments sqrt(dt) * L * z.

    Parameters
    ----------
    key : JAX PRNGKey
    n_rates : number of rates
    correlation_matrix : (n, n) correlation matrix
    dt : time step

    Returns array (n,) of correlated increments.
    """
    L = jnp.linalg.cholesky(correlation_matrix)
    z = jax.random.normal(key, (n_rates,))
    return jnp.sqrt(dt) * L @ z


def simulate_lmm(initial_rates, volatilities, correlations, rate_times,
                   n_paths=10000, method='predictor_corrector', key=None):
    """Full LMM simulation generating paths of forward rates.

    Parameters
    ----------
    initial_rates : array (n,) – initial forward rates
    volatilities : array (n,) – constant vols (or callable(t, i))
    correlations : array (n, n) – correlation matrix
    rate_times : array (n+1,) – [T0, T1, ..., Tn]
    n_paths : number of simulation paths
    method : 'euler' or 'predictor_corrector'
    key : JAX PRNGKey

    Returns array (n_paths, n_steps, n_rates) of evolved forward rates.
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    n = len(initial_rates)
    accruals = jnp.diff(rate_times)
    evolve_fn = predictor_corrector_evolve if method == 'predictor_corrector' else euler_evolve

    # Time steps at rate fixing dates
    n_steps = n
    all_paths = []

    for path in range(n_paths):
        rates = initial_rates.copy()
        path_rates = [rates]

        for step in range(n_steps):
            key, sk = jax.random.split(key)
            dt = accruals[step]
            dw = generate_correlated_normals(sk, n, correlations, dt)

            rates = evolve_fn(rates, volatilities, correlations, accruals,
                               step, dt, dw)
            path_rates.append(rates)

        all_paths.append(jnp.stack(path_rates))

    return jnp.stack(all_paths)
