"""SABR model functions.

The SABR stochastic volatility model:
    dF = sigma * F^beta * dW1
    d(sigma) = nu * sigma * dW2
    dW1 * dW2 = rho * dt

Hagan et al. (2002) implied volatility approximation.
"""

import jax.numpy as jnp


def sabr_vol(strike, forward, t, alpha, beta, rho, nu):
    """SABR implied volatility using Hagan's formula.

    Parameters
    ----------
    strike : float or array
        Strike price(s).
    forward : float
        Forward price.
    t : float
        Time to expiry.
    alpha : float
        Vol-of-vol initial level.
    beta : float
        CEV exponent (0 = normal, 1 = lognormal).
    rho : float
        Correlation between forward and vol.
    nu : float
        Vol-of-vol.

    Returns
    -------
    Implied Black volatility.
    """
    strike = jnp.asarray(strike, dtype=jnp.float64)
    forward = jnp.asarray(forward, dtype=jnp.float64)
    t = jnp.asarray(t, dtype=jnp.float64)

    # Handle ATM case for numerical stability
    eps = 1e-12
    fk = forward * strike
    fk = jnp.maximum(fk, eps)

    one_minus_beta = 1.0 - beta
    fk_beta = jnp.power(fk, one_minus_beta / 2.0)
    log_fk = jnp.log(forward / jnp.maximum(strike, eps))

    # z = nu / alpha * fk^((1-beta)/2) * log(F/K)
    z = nu / jnp.maximum(alpha, eps) * fk_beta * log_fk

    # x(z) = log((sqrt(1 - 2*rho*z + z^2) + z - rho) / (1 - rho))
    sqrt_term = jnp.sqrt(1.0 - 2.0 * rho * z + z * z)
    x_z = jnp.log((sqrt_term + z - rho) / (1.0 - rho + eps))

    # ATM correction
    z_over_xz = jnp.where(jnp.abs(z) < eps, 1.0, z / (x_z + eps))

    # Leading term
    a = alpha / (fk_beta * (1.0 + one_minus_beta**2 / 24.0 * log_fk**2
                             + one_minus_beta**4 / 1920.0 * log_fk**4))

    # Correction terms
    b1 = one_minus_beta**2 / 24.0 * alpha**2 / jnp.maximum(fk**one_minus_beta, eps)
    b2 = 0.25 * rho * beta * nu * alpha / jnp.maximum(fk_beta, eps)
    b3 = (2.0 - 3.0 * rho**2) / 24.0 * nu**2

    correction = 1.0 + (b1 + b2 + b3) * t

    return a * z_over_xz * correction


def sabr_vol_normal(strike, forward, t, alpha, beta, rho, nu):
    """SABR normal (Bachelier) implied volatility."""
    black_vol = sabr_vol(strike, forward, t, alpha, beta, rho, nu)
    # Convert Black vol to normal vol (approximate)
    fk = jnp.sqrt(forward * jnp.maximum(strike, 1e-12))
    return black_vol * fk
