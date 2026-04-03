"""Risk-neutral density (RND) calculators.

Provides closed-form or numerical methods for extracting risk-neutral
densities from various models.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def bsm_rnd(S, r, q, sigma, T, x):
    """Black-Scholes-Merton risk-neutral density at spot level x.

    p(x) = d²C/dK² * exp(rT) (the Q-measure density, integrates to 1).

    Parameters
    ----------
    S : current spot
    r, q : rates
    sigma : vol
    T : maturity
    x : array of spot levels

    Returns
    -------
    density : risk-neutral density values
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    F = S * jnp.exp((r - q) * T)
    sigma_sqrt_T = sigma * jnp.sqrt(T)

    d2 = (jnp.log(F / x) - 0.5 * sigma**2 * T) / sigma_sqrt_T

    # Risk-neutral density (integrates to 1): phi(d2) / (x * sigma * sqrt(T))
    return norm.pdf(d2) / (x * sigma_sqrt_T)


def cev_rnd(S, r, q, sigma, beta, T, x):
    """CEV model risk-neutral density (numerical differentiation).

    Uses second derivative of the call price with respect to strike.
    """
    from ql_jax.engines.analytic.cev import cev_price
    eps = x * 1e-4 + 1e-8
    c_up = jax.vmap(lambda k: cev_price(S, k + eps, T, r, q, sigma, beta, 1))(x)
    c_mid = jax.vmap(lambda k: cev_price(S, k, T, r, q, sigma, beta, 1))(x)
    c_dn = jax.vmap(lambda k: cev_price(S, k - eps, T, r, q, sigma, beta, 1))(x)
    return jnp.exp(r * T) * (c_up - 2.0 * c_mid + c_dn) / (eps**2)


def heston_rnd(S, r, q, v0, kappa, theta, xi, rho, T, x):
    """Heston model risk-neutral density (numerical second derivative of call).

    Parameters
    ----------
    S : spot
    r, q : rates
    v0, kappa, theta, xi, rho : Heston params
    T : maturity
    x : spot levels

    Returns
    -------
    density : RND values
    """
    from ql_jax.engines.analytic.heston import heston_price
    eps = x * 1e-4 + 1e-8
    c_up = jax.vmap(lambda k: heston_price(S, k + eps, T, r, q, v0, kappa, theta, xi, rho, 1))(x)
    c_mid = jax.vmap(lambda k: heston_price(S, k, T, r, q, v0, kappa, theta, xi, rho, 1))(x)
    c_dn = jax.vmap(lambda k: heston_price(S, k - eps, T, r, q, v0, kappa, theta, xi, rho, 1))(x)
    return jnp.exp(r * T) * (c_up - 2.0 * c_mid + c_dn) / (eps**2)


def local_vol_rnd(local_vol_fn, S, r, q, T, x, n_time=100):
    """Local volatility model RND via Fokker-Planck forward PDE.

    Solves dp/dt = -d/dx[(r-q-0.5*sigma_loc^2)*p] + 0.5*d^2/dx^2[sigma_loc^2*p]

    Parameters
    ----------
    local_vol_fn : callable(S, t) -> local vol
    S : initial spot
    r, q : rates
    T : maturity
    x : spot grid
    n_time : time steps

    Returns
    -------
    density : approximate RND at time T
    """
    from ql_jax.methods.finitedifferences.operators import d_zero, d_plus_d_minus

    dx_arr = jnp.diff(x)
    n = len(x)
    dt = T / n_time

    # Initial condition: delta at S (discrete approximation)
    idx = jnp.argmin(jnp.abs(x - S))
    p = jnp.zeros(n)
    p = p.at[idx].set(1.0 / (dx_arr[jnp.minimum(idx, n - 2)]))

    for step in range(n_time):
        t = step * dt
        sigma_loc = jnp.array([float(local_vol_fn(float(xi), t)) for xi in x])
        vol2 = sigma_loc**2

        # Explicit Euler step for Fokker-Planck
        D1 = d_zero(x)
        D2 = d_plus_d_minus(x)

        drift = (r - q - 0.5 * vol2)
        # dp/dt = -d/dx[drift*p] + 0.5*d^2/dx^2[vol2*p]
        advection = D1.apply(drift * p)
        diffusion = D2.apply(0.5 * vol2 * p)
        p = p + dt * (-advection + diffusion)
        p = jnp.maximum(p, 0.0)

    # Normalize
    integral = jnp.trapezoid(p, x)
    return jnp.where(integral > 0, p / integral, p)
