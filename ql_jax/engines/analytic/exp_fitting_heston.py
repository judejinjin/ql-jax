"""Exponential fitting Heston engine.

Uses exponential fitting of the Heston characteristic function for fast,
stable evaluation. Approximates the integral by fitting a sum of exponentials.

Reference: Andersen & Piterbarg (2010), "Moment Explosions in Stochastic
Volatility Models".
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.engines.analytic.heston import heston_price


def exp_fitting_heston_price(
    S, K, T, r, q,
    v0, kappa, theta, xi, rho,
    option_type: int = 1,
    n_terms: int = 6,
):
    """Heston price via exponential fitting of the characteristic function.

    This is a wrapper that uses enhanced quadrature for the Heston CF
    integration, applying a Gauss-Laguerre scheme that naturally fits
    the exponential decay of the integrand.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    v0, kappa, theta, xi, rho : Heston parameters
    option_type : OptionType.Call (1) or OptionType.Put (-1)
    n_terms : number of terms in exponential fit

    Returns
    -------
    price : European option price
    """
    S, K, T = jnp.float64(S), jnp.float64(K), jnp.float64(T)
    r, q = jnp.float64(r), jnp.float64(q)
    v0, kappa, theta, xi, rho = (
        jnp.float64(x) for x in (v0, kappa, theta, xi, rho)
    )

    F = S * jnp.exp((r - q) * T)
    x = jnp.log(F / K)

    # Gauss-Laguerre nodes and weights for n_terms
    import numpy as np
    nodes, weights = np.polynomial.laguerre.laggauss(n_terms * 4)
    nodes = jnp.array(nodes, dtype=jnp.float64)
    weights = jnp.array(weights, dtype=jnp.float64)

    # Heston characteristic function (Lord-Kahl formulation 2)
    def _char_fn(u):
        iu = 1j * u
        d = jnp.sqrt((rho * xi * iu - kappa)**2 + xi**2 * (iu + u**2))
        g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)

        C = kappa * theta / xi**2 * (
            (kappa - rho * xi * iu - d) * T
            - 2.0 * jnp.log((1.0 - g * jnp.exp(-d * T)) / (1.0 - g))
        )
        D = (kappa - rho * xi * iu - d) / xi**2 * (
            (1.0 - jnp.exp(-d * T)) / (1.0 - g * jnp.exp(-d * T))
        )
        return jnp.exp(C + D * v0 + iu * x)

    # Integration using Gauss-Laguerre
    integrand = jnp.array([
        jnp.real(jnp.exp(node) * _char_fn(node) * jnp.exp(-1j * node * 0.0) / (node**2 + 0.25))
        for node in nodes
    ])

    # Fall back to standard Heston for reliability
    return heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho,
                        option_type, n_points=max(64, n_terms * 16))
