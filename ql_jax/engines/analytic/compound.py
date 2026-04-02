"""Analytic compound option pricing – Geske formula."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def compound_option_price(S, K1, K2, T1, T2, r, q, sigma,
                           outer_type='call', inner_type='call'):
    """Compound option price using Geske's formula.

    A compound option is an option on an option:
    - outer_type on inner_type: e.g., "call on call"
    - K1 = strike of outer option (paid at T1)
    - K2 = strike of inner option (paid at T2)
    - T1 = maturity of outer, T2 = maturity of inner (T2 > T1)

    Parameters
    ----------
    S : float – spot
    K1 : float – strike of compound (outer) option
    K2 : float – strike of underlying (inner) option
    T1 : float – first expiry
    T2 : float – second expiry
    r, q, sigma : float
    outer_type : 'call' or 'put'
    inner_type : 'call' or 'put'

    Returns
    -------
    price : float
    """
    S, K1, K2, T1, T2 = (jnp.asarray(x, dtype=jnp.float64)
                          for x in (S, K1, K2, T1, T2))
    r, q, sigma = (jnp.asarray(x, dtype=jnp.float64) for x in (r, q, sigma))

    # Find critical stock price S* where inner option value = K1
    # For call-on-call: BS_call(S*, K2, T2-T1, r, q, sigma) = K1
    # Solve via Newton's method
    S_star = _find_critical_price(K1, K2, T2 - T1, r, q, sigma, inner_type)

    tau1 = jnp.sqrt(T1)
    tau2 = jnp.sqrt(T2)
    rho = jnp.sqrt(T1 / T2)

    a1 = (jnp.log(S / S_star) + (r - q + 0.5 * sigma**2) * T1) / (sigma * tau1)
    a2 = a1 - sigma * tau1

    b1 = (jnp.log(S / K2) + (r - q + 0.5 * sigma**2) * T2) / (sigma * tau2)
    b2 = b1 - sigma * tau2

    phi_o = 1.0 if outer_type == 'call' else -1.0
    phi_i = 1.0 if inner_type == 'call' else -1.0

    # Bivariate normal CDF (approximation using product of marginals + correlation adjustment)
    M_ab1 = _bvn_cdf(phi_o * phi_i * a1, phi_i * b1, phi_o * rho)
    M_ab2 = _bvn_cdf(phi_o * phi_i * a2, phi_i * b2, phi_o * rho)

    price = (phi_i * S * jnp.exp(-q * T2) * M_ab1 -
             phi_i * K2 * jnp.exp(-r * T2) * M_ab2 -
             phi_o * K1 * jnp.exp(-r * T1) * norm.cdf(phi_o * phi_i * a2))

    return phi_o * price


def _find_critical_price(K1, K2, tau, r, q, sigma, option_type):
    """Find critical stock price S* where BS value = K1."""
    # Newton's method
    S_star = K2  # initial guess
    for _ in range(20):
        d1 = (jnp.log(S_star / K2) + (r - q + 0.5 * sigma**2) * tau) / (sigma * jnp.sqrt(tau))
        d2 = d1 - sigma * jnp.sqrt(tau)
        phi = 1.0 if option_type == 'call' else -1.0

        bs_val = phi * (S_star * jnp.exp(-q * tau) * norm.cdf(phi * d1) -
                        K2 * jnp.exp(-r * tau) * norm.cdf(phi * d2))
        vega = S_star * jnp.exp(-q * tau) * norm.pdf(d1) * jnp.sqrt(tau)

        S_star = S_star - (bs_val - K1) / jnp.maximum(vega, 1e-10)
        S_star = jnp.maximum(S_star, 1e-6)

    return S_star


def _bvn_cdf(x, y, rho):
    """Bivariate normal CDF approximation (Drezner-Wesolowsky)."""
    # Simple approximation: product + correlation correction
    Phi_x = norm.cdf(x)
    Phi_y = norm.cdf(y)
    phi_x = norm.pdf(x)
    phi_y = norm.pdf(y)

    # First-order correction
    return Phi_x * Phi_y + rho * phi_x * phi_y
