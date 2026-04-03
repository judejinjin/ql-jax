"""Analytic CEV (Constant Elasticity of Variance) engine.

dS = mu*S*dt + sigma*S^beta*dW

Uses the non-central chi-squared distribution for European option pricing.
Reference: Schroder (1989), "Computing the CEV option pricing formula".
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType


def _ncx2_cdf(x, k, lam):
    """Non-central chi-squared CDF via series expansion.

    Parameters
    ----------
    x : argument
    k : degrees of freedom
    lam : non-centrality parameter

    Uses Poisson-weighted sum of central chi-squared CDFs.
    """
    from jax.scipy.stats import chi2 as chi2_dist
    max_terms = 80
    half_lam = 0.5 * lam
    log_half_lam = jnp.log(jnp.maximum(half_lam, 1e-300))

    def _body(carry, j):
        j_f = jnp.float64(j)
        log_weight = j_f * log_half_lam - half_lam - jax.scipy.special.gammaln(j_f + 1.0)
        weight = jnp.exp(log_weight)
        # chi2 CDF with k + 2*j degrees of freedom
        nu = k + 2.0 * j_f
        chi2_val = _regularized_gamma_p(0.5 * nu, 0.5 * x)
        term = weight * chi2_val
        return carry + term, None

    import jax.lax as lax
    result, _ = lax.scan(_body, jnp.float64(0.0), jnp.arange(max_terms))
    return result


def _regularized_gamma_p(a, x):
    """Lower regularized incomplete gamma function P(a, x) = gamma(a,x)/Gamma(a)."""
    from jax.scipy.special import gammainc
    return gammainc(a, x)


def cev_price(S, K, T, r, q, sigma, beta, option_type: int):
    """European option price under the CEV model.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    sigma : CEV volatility parameter
    beta : elasticity parameter (0 < beta < 1 for inverse leverage, beta > 1 for leverage)
    option_type : OptionType.Call (1) or OptionType.Put (-1)

    Returns
    -------
    price : European option price
    """
    S, K, T, r, q, sigma, beta = (
        jnp.asarray(x, dtype=jnp.float64) for x in (S, K, T, r, q, sigma, beta)
    )

    F = S * jnp.exp((r - q) * T)
    disc = jnp.exp(-r * T)

    # CEV parameters
    nu = 1.0 / (2.0 * (1.0 - beta))
    # Handle beta = 1 (Black-Scholes) separately
    tau = sigma**2 * T / (2.0 * (r - q) * (1.0 - beta)) * (
        jnp.exp(2.0 * (r - q) * (1.0 - beta) * T) - 1.0
    )
    # When r-q ~ 0
    tau = jnp.where(
        jnp.abs(r - q) < 1e-12,
        sigma**2 * T * S**(2.0 * (beta - 1.0)),
        sigma**2 / (2.0 * (r - q) * (1.0 - beta)) * (
            F**(2.0 * (1.0 - beta)) - S**(2.0 * (1.0 - beta))
        ),
    )
    tau = jnp.maximum(tau, 1e-50)

    x_val = K**(2.0 * (1.0 - beta)) / ((1.0 - beta)**2 * tau)
    y_val = F**(2.0 * (1.0 - beta)) / ((1.0 - beta)**2 * tau)

    # For beta < 1: Call = F * (1 - ncx2(x, 2+1/nu, y)) - K * ncx2(y, 1/nu, x)
    # For beta > 1: swap roles
    call_price = jnp.where(
        beta < 1.0,
        disc * (F * (1.0 - _ncx2_cdf(x_val, 2.0 + 1.0 / nu, y_val))
                - K * _ncx2_cdf(y_val, 1.0 / nu, x_val)),
        disc * (F * _ncx2_cdf(y_val, 2.0 + 1.0 / nu, x_val)
                - K * (1.0 - _ncx2_cdf(x_val, 1.0 / nu, y_val))),
    )

    return jnp.where(option_type == OptionType.Call, call_price, call_price - disc * (F - K))
