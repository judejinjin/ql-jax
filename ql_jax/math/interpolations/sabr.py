"""SABR interpolation for volatility smile.

Calibrates SABR parameters (alpha, beta, nu, rho) to market vols
and evaluates the smile.
"""

import jax
import jax.numpy as jnp
from functools import partial


def sabr_vol(f, k, t, alpha, beta, rho, nu):
    """Hagan's SABR implied volatility formula.

    Parameters
    ----------
    f : forward price
    k : strike
    t : time to expiry
    alpha : initial vol level
    beta : CEV exponent (0=normal, 1=lognormal)
    rho : correlation between spot and vol
    nu : vol-of-vol
    """
    fk = f * k
    fk_beta = jnp.power(fk, (1.0 - beta) / 2.0)
    log_fk = jnp.log(f / k)

    # Handle ATM case
    is_atm = jnp.abs(log_fk) < 1e-12

    # z and x(z)
    z = nu / alpha * fk_beta * log_fk
    x_z = jnp.log((jnp.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
    x_z = jnp.where(jnp.abs(z) < 1e-12, 1.0, z / x_z)

    # Pre-factors
    one_minus_beta = 1.0 - beta

    # Denominator
    denom = fk_beta * (1.0 + one_minus_beta**2 / 24.0 * log_fk**2
                        + one_minus_beta**4 / 1920.0 * log_fk**4)

    # Correction term
    corr = (1.0 + (one_minus_beta**2 / 24.0 * alpha**2 / jnp.power(fk, one_minus_beta)
                    + 0.25 * rho * beta * nu * alpha / fk_beta
                    + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * t)

    vol = alpha / denom * x_z * corr

    # ATM formula
    vol_atm = (alpha / jnp.power(f, one_minus_beta)
               * (1.0 + (one_minus_beta**2 / 24.0 * alpha**2 / jnp.power(f, 2.0 * one_minus_beta)
                          + 0.25 * rho * beta * nu * alpha / jnp.power(f, one_minus_beta)
                          + (2.0 - 3.0 * rho**2) / 24.0 * nu**2) * t))

    return jnp.where(is_atm, vol_atm, vol)


def sabr_vol_normal(f, k, t, alpha, beta, rho, nu):
    """SABR normal (Bachelier) volatility approximation."""
    fk = f * k
    fk_beta = jnp.power(fk, beta / 2.0)

    log_fk = jnp.log(f / k)
    is_atm = jnp.abs(log_fk) < 1e-12

    z = nu / alpha * (f - k) / fk_beta
    x_z = jnp.log((jnp.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
    x_z = jnp.where(jnp.abs(z) < 1e-12, 1.0, z / x_z)

    vol_n = alpha * fk_beta * x_z
    return vol_n


def build(strikes, vols, forward, expiry, beta=0.5):
    """Calibrate SABR parameters to market volatility smile.

    Parameters
    ----------
    strikes : array of strikes
    vols : array of market implied vols
    forward : forward price
    expiry : time to expiry
    beta : CEV exponent (fixed)

    Returns
    -------
    dict with calibrated parameters and input data
    """
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    vols = jnp.asarray(vols, dtype=jnp.float64)

    # Initial guess
    atm_idx = jnp.argmin(jnp.abs(strikes - forward))
    alpha0 = float(vols[atm_idx]) * forward**(1.0 - beta)
    rho0 = -0.2
    nu0 = 0.4

    # Simple gradient descent calibration
    params = jnp.array([alpha0, rho0, nu0])

    def loss(params):
        alpha, rho, nu = params
        rho = jnp.tanh(rho)  # constrain to (-1, 1)
        nu = jnp.abs(nu)
        alpha = jnp.abs(alpha)
        model_vols = jax.vmap(lambda k: sabr_vol(forward, k, expiry, alpha, beta, rho, nu))(strikes)
        return jnp.mean((model_vols - vols)**2)

    grad_fn = jax.grad(loss)

    # Simple gradient descent
    lr = 0.01
    for _ in range(200):
        g = grad_fn(params)
        params = params - lr * g

    alpha, rho, nu = params
    rho = jnp.tanh(rho)
    nu = jnp.abs(nu)
    alpha = jnp.abs(alpha)

    return {
        "strikes": strikes, "vols": vols, "forward": forward,
        "expiry": expiry, "beta": beta,
        "alpha": alpha, "rho": rho, "nu": nu,
    }


def evaluate(state, k):
    """Evaluate calibrated SABR vol at strike k."""
    return sabr_vol(
        state["forward"], k, state["expiry"],
        state["alpha"], state["beta"], state["rho"], state["nu"]
    )
