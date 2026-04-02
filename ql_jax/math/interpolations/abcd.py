"""ABCD interpolation for volatility term structures.

The ABCD parameterization for instantaneous vol:
  sigma(t) = (a + b*t) * exp(-c*t) + d

Used in market models for caplet volatility fitting.
"""

import jax.numpy as jnp


def abcd_vol(t, a, b, c, d):
    """Instantaneous vol: sigma(t) = (a + b*t) * exp(-c*t) + d."""
    return (a + b * t) * jnp.exp(-c * t) + d


def abcd_black_vol(T, a, b, c, d):
    """Time-averaged (Black) vol for maturity T.

    BlackVol^2 * T = integral_0^T sigma(t)^2 dt
    """
    eps = 1e-12
    T = jnp.maximum(T, eps)
    c2 = c * c
    exp_2cT = jnp.exp(-2.0 * c * T)

    # Compute integral of ((a+bt)e^{-ct}+d)^2 dt from 0 to T
    # = integral of (a+bt)^2 e^{-2ct} + 2d(a+bt)e^{-ct} + d^2  dt
    i1 = (a * a + a * b / c + b * b / (2.0 * c2)) / (2.0 * c) * (1.0 - exp_2cT)
    i2 = -b / c2 * (a + b / (2.0 * c)) * T * exp_2cT
    i3 = -b * b / (4.0 * c * c2) * T * T * exp_2cT  # correction term

    exp_cT = jnp.exp(-c * T)
    j1 = 2.0 * d * (a / c * (1.0 - exp_cT) + b / c2 * (1.0 - (1.0 + c * T) * exp_cT))
    j2 = d * d * T

    var = (i1 + i2 + i3 + j1 + j2) / T
    return jnp.sqrt(jnp.maximum(var, 0.0))


def build(times, vols, initial_guess=None):
    """Calibrate ABCD parameters to market vols by least-squares.

    Parameters
    ----------
    times : array – option maturities
    vols : array – market Black vols
    initial_guess : optional dict with a, b, c, d

    Returns dict with keys: a, b, c, d.
    """
    from ql_jax.math.optimization.levenberg_marquardt import minimize

    if initial_guess is None:
        x0 = jnp.array([0.0, 0.01, 0.5, 0.15])
    else:
        x0 = jnp.array([initial_guess['a'], initial_guess['b'],
                         initial_guess['c'], initial_guess['d']])

    def residuals(params):
        a, b_raw, c_raw, d_raw = params
        c_val = jnp.exp(c_raw)  # ensure c > 0
        d_val = jnp.abs(d_raw)  # ensure d >= 0
        model = jnp.array([abcd_black_vol(t, a, b_raw, c_val, d_val) for t in times])
        return model - vols

    result = minimize(residuals, x0, max_iterations=200)
    p = result['x']
    return {'a': p[0], 'b': p[1], 'c': jnp.exp(p[2]), 'd': jnp.abs(p[3])}


def evaluate(params, t):
    """Evaluate ABCD Black vol at maturity t."""
    return abcd_black_vol(t, params['a'], params['b'], params['c'], params['d'])
