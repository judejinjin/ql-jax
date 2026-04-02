"""ZABR stochastic volatility interpolation (Andreasen-Huge extension of SABR).

The ZABR model generalizes SABR by allowing the exponent beta to be stochastic:
  dF = sigma * F^beta * dW_1
  dsigma = nu * sigma^gamma * dW_2
  <dW_1, dW_2> = rho * dt
"""

import jax.numpy as jnp
from functools import partial


def zabr_local_vol(f, k, t, alpha, beta, nu, rho, gamma=1.0):
    """ZABR implied vol using short-maturity asymptotic expansion.

    Parameters
    ----------
    f : float – forward
    k : float – strike
    t : float – time to expiry
    alpha : float – initial vol level
    beta : float – CEV exponent
    nu : float – vol-of-vol
    rho : float – correlation
    gamma : float – vol process exponent (1.0 = lognormal vol, 0.0 = normal vol)
    """
    eps = 1e-10
    fk = jnp.maximum(f * k, eps)
    fk_mid = jnp.sqrt(fk)
    log_fk = jnp.log(jnp.maximum(f / k, eps))

    one_minus_beta = 1.0 - beta
    fk_pow = fk_mid ** one_minus_beta

    # Leading-order SABR-like vol
    sigma0 = alpha / fk_pow
    zeta = nu / alpha * fk_pow * log_fk
    x_zeta = _chi(zeta, rho)
    vol0 = sigma0 * zeta / jnp.where(jnp.abs(zeta) < eps, 1.0, x_zeta)

    # Higher-order corrections
    c1 = one_minus_beta ** 2 / 24.0 * alpha ** 2 / (fk_pow ** 2)
    c2 = 0.25 * rho * beta * nu * alpha / fk_pow
    c3 = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    correction = 1.0 + (c1 + c2 + c3) * t

    # Gamma correction for ZABR (modifies vol-of-vol contribution)
    gamma_adj = 1.0 + (gamma - 1.0) * nu ** 2 * t / 24.0
    return vol0 * correction * gamma_adj


def _chi(zeta, rho):
    """Helper: chi(zeta) = log((sqrt(1-2*rho*zeta+zeta^2) + zeta - rho) / (1-rho))."""
    disc = jnp.sqrt(1.0 - 2.0 * rho * zeta + zeta ** 2)
    return jnp.log((disc + zeta - rho) / (1.0 - rho + 1e-15))


def build(strikes, vols, f, t, beta=0.5, gamma=1.0):
    """Calibrate ZABR parameters (alpha, nu, rho) by least-squares.

    Returns dict with keys: alpha, beta, nu, rho, gamma, strikes, f, t.
    """
    from ql_jax.math.optimization.levenberg_marquardt import minimize

    def residuals(params):
        alpha = jnp.exp(params[0])
        nu = jnp.exp(params[1])
        rho = jnp.tanh(params[2])
        model_vols = jnp.array([
            zabr_local_vol(f, k, t, alpha, beta, nu, rho, gamma) for k in strikes
        ])
        return model_vols - vols

    x0 = jnp.array([jnp.log(0.2), jnp.log(0.4), 0.0])
    result = minimize(residuals, x0, max_iterations=200)
    p = result['x']
    return {
        'alpha': jnp.exp(p[0]),
        'beta': beta,
        'nu': jnp.exp(p[1]),
        'rho': jnp.tanh(p[2]),
        'gamma': gamma,
        'f': f,
        't': t,
    }


def evaluate(params, strike):
    """Evaluate ZABR implied vol at a given strike."""
    return zabr_local_vol(
        params['f'], strike, params['t'],
        params['alpha'], params['beta'],
        params['nu'], params['rho'], params['gamma'],
    )
