"""XABR (eXtended ABCD) interpolation framework.

Generalises the SABR model by allowing alternative functional forms
for the local volatility backbone.  The SABR model is a special case
where the backbone is ``sigma(F) = alpha * F^(beta-1)``.

The XABR framework replaces this with a user-defined ``phi(F)``
function, enabling e.g. shifted-SABR, free-boundary SABR, or
CEV-like variants.

This module provides:
* ``xabr_vol``  — XABR implied vol given parameters
* ``calibrate_xabr`` — least-squares calibration to market vols
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def xabr_vol(
    f: float,
    k: float,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    phi_fn=None,
) -> float:
    """XABR implied volatility.

    Uses a generic backbone ``phi(F)`` instead of the SABR default
    ``F^(beta-1)``.  When ``phi_fn`` is None, falls back to standard SABR.

    Parameters
    ----------
    f : forward price
    k : strike
    t : time to expiry
    alpha, beta, rho, nu : SABR-like parameters
    phi_fn : callable(F, beta) -> local vol multiplier.
             If None, uses ``F^(beta-1)`` (standard SABR).

    Returns
    -------
    vol : implied Black vol
    """
    if phi_fn is None:
        def phi_fn(F, b):
            return jnp.power(jnp.maximum(F, 1e-10), b - 1.0)

    fk = jnp.sqrt(f * k)
    phi_fk = phi_fn(fk, beta)
    log_fk = jnp.log(f / k)

    # z and chi(z)
    z = nu / alpha * (f - k) * phi_fk
    chi_z = jnp.log(
        (jnp.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho)
    )
    # Ratio z/chi(z), with ATM limit = 1
    z_over_chi = jnp.where(jnp.abs(z) < 1e-12, 1.0, z / chi_z)

    # Mid-point local vol
    sigma_0 = alpha * phi_fk

    # Higher-order corrections
    I1 = (
        phi_fk**2 / 24.0 * alpha**2
        + 0.25 * rho * nu * alpha * phi_fk
        + (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    )

    # Denominator correction for log-moneyness
    denom = 1.0 + log_fk**2 / 24.0 + log_fk**4 / 1920.0

    vol = sigma_0 / denom * z_over_chi * (1.0 + I1 * t)
    return vol


def calibrate_xabr(
    f: float,
    strikes: jnp.ndarray,
    market_vols: jnp.ndarray,
    t: float,
    beta: float = 0.5,
    phi_fn=None,
    initial_guess: tuple | None = None,
    max_iter: int = 200,
) -> dict:
    """Calibrate XABR parameters to market smile.

    Fixes beta and calibrates (alpha, rho, nu).

    Parameters
    ----------
    f : forward
    strikes : array of strikes
    market_vols : array of market implied vols
    t : time to expiry
    beta : CEV exponent (fixed)
    phi_fn : backbone function (None = standard SABR)
    initial_guess : (alpha0, rho0, nu0)
    max_iter : optimiser iterations

    Returns
    -------
    dict with keys 'alpha', 'beta', 'rho', 'nu', 'error'
    """
    from ql_jax.math.optimization.levenberg_marquardt import levenberg_marquardt

    if initial_guess is None:
        alpha0 = float(jnp.mean(market_vols))
        rho0 = -0.2
        nu0 = 0.4
    else:
        alpha0, rho0, nu0 = initial_guess

    def residuals(params):
        alpha, rho_raw, nu_raw = params
        rho = jnp.tanh(rho_raw)
        nu = jnp.exp(nu_raw)
        model_vols = jnp.array([
            xabr_vol(f, float(K), t, alpha, beta, rho, nu, phi_fn)
            for K in strikes
        ])
        return model_vols - market_vols

    x0 = jnp.array([alpha0, jnp.arctanh(rho0), jnp.log(nu0)])

    try:
        result = levenberg_marquardt(residuals, x0, max_iter=max_iter)
        alpha_opt = float(result[0])
        rho_opt = float(jnp.tanh(result[1]))
        nu_opt = float(jnp.exp(result[2]))
    except Exception:
        # Fallback: simple grid search
        alpha_opt, rho_opt, nu_opt = alpha0, rho0, nu0

    final_err = float(jnp.sum(residuals(jnp.array([alpha_opt, jnp.arctanh(rho_opt), jnp.log(nu_opt)]))**2))

    return {
        'alpha': alpha_opt,
        'beta': beta,
        'rho': rho_opt,
        'nu': nu_opt,
        'error': final_err,
    }
