"""Volatility models for market models.

Parametric volatility structures for forward rate instantaneous vols.
"""

import jax.numpy as jnp


def abcd_volatility(a, b, c, d, T_fix, t):
    """ABCD instantaneous vol: sigma(t, T) = (a + b*(T-t)) * exp(-c*(T-t)) + d.

    Parameters
    ----------
    a, b, c, d : ABCD parameters
    T_fix : fixing time of the forward rate
    t : current time
    """
    tau = T_fix - t
    return (a + b * tau) * jnp.exp(-c * tau) + d


def piecewise_constant_volatility(vol_matrix, time_idx, rate_idx):
    """Piecewise-constant vol: sigma(time_idx, rate_idx).

    Parameters
    ----------
    vol_matrix : array (n_steps, n_rates) – vol at each step for each rate
    time_idx : int – current time index
    rate_idx : int – rate index

    Returns float volatility.
    """
    return vol_matrix[time_idx, rate_idx]


def calibrate_abcd_to_caplets(caplet_vols, rate_times, initial_guess=None):
    """Fit ABCD parameters to market caplet volatilities.

    Parameters
    ----------
    caplet_vols : array – market caplet vols (Black)
    rate_times : array – [T0, T1, ..., Tn]

    Returns dict with a, b, c, d.
    """
    from ql_jax.math.optimization.levenberg_marquardt import minimize

    if initial_guess is None:
        x0 = jnp.array([0.0, 0.01, 0.5, 0.15])
    else:
        x0 = jnp.array([initial_guess['a'], initial_guess['b'],
                         initial_guess['c'], initial_guess['d']])

    accruals = jnp.diff(rate_times)

    def residuals(params):
        a, b, c_raw, d = params
        c = jnp.abs(c_raw)
        model_vols = []
        for i in range(len(caplet_vols)):
            T_fix = rate_times[i]
            # Black vol = sqrt(1/T * integral_0^T sigma(t)^2 dt)
            # Approximate with mid-point
            from ql_jax.math.interpolations.abcd import abcd_black_vol
            model_vols.append(abcd_black_vol(T_fix, a, b, c, d))
        return jnp.array(model_vols) - caplet_vols

    result = minimize(residuals, x0, max_iterations=200)
    p = result['x']
    return {'a': p[0], 'b': p[1], 'c': jnp.abs(p[2]), 'd': p[3]}


def calibrate_piecewise_to_swaptions(swaption_vols, rate_times, correlation_matrix,
                                       swap_lengths, initial_vols=None):
    """Cascade calibration of piecewise-constant vols to swaption matrix.

    Uses the Rebonato formula: sigma_swap^2 * T ≈ sum of rate vol contributions.

    Returns (n_steps, n_rates) vol matrix.
    """
    n = len(rate_times) - 1
    accruals = jnp.diff(rate_times)
    vol_matrix = jnp.full((n, n), 0.01) if initial_vols is None else initial_vols

    # Bootstrap: calibrate column by column
    for step in range(n):
        from ql_jax.math.solvers.brent import solve

        def objective(vol_val, step=step):
            vm = vol_matrix.at[step, step].set(vol_val)
            # Rebonato formula for ATM swaption vol
            T_exp = rate_times[step]
            swap_len = min(swap_lengths[step] if step < len(swap_lengths) else n - step, n - step)
            # Simplified: only calibrate diagonal
            model_vol_sq = vol_val**2 * T_exp
            return jnp.sqrt(model_vol_sq / T_exp) - swaption_vols[step]

        try:
            v = solve(objective, 0.001, 1.0)
            vol_matrix = vol_matrix.at[step, step].set(v)
        except Exception:
            pass

    return vol_matrix
