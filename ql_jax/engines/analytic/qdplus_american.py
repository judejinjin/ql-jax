"""QD+ American option engine (Andersen-Lake-Offengenden, 2016).

High-accuracy analytic approximation with iterative early-exercise
boundary refinement.  For American puts and calls with continuous dividends.

Reference: Andersen, L., Lake, M. & Offengenden, D. (2016),
"High Performance American Option Pricing", Journal of Computational Finance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType
from ql_jax.engines.analytic.black_formula import black_scholes_price


def qdplus_american_price(
    S, K, T, r, q, sigma, option_type: int, n_iter: int = 8,
):
    """QD+ American option price.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    option_type : 1 for call, -1 for put
    n_iter : iterations for boundary refinement

    Returns
    -------
    price : American option price
    """
    S, K, T, r, q, sigma = (jnp.float64(x) for x in (S, K, T, r, q, sigma))
    phi = jnp.float64(option_type)

    # European price
    euro = black_scholes_price(S, K, T, r, q, sigma, option_type)

    sigma2 = sigma * sigma
    sqrt_T = jnp.sqrt(jnp.maximum(T, 1e-20))

    # QD parameters (BAW-style)
    M = 2.0 * r / sigma2
    N_param = 2.0 * (r - q) / sigma2
    K_val = 1.0 - jnp.exp(-r * T)
    K_val = jnp.maximum(K_val, 1e-12)

    q2 = (-(N_param - 1.0) + phi * jnp.sqrt(
        (N_param - 1.0)**2 + 4.0 * M / K_val
    )) / 2.0

    # Initial exercise boundary estimate
    # For call: S* > K, typically S* = K * factor where factor > 1
    # For put: S* < K, typically S* = K * factor where factor < 1
    # Use BAW initial guess
    def _compute_s_star_init():
        # Seed from perpetual boundary
        perp_boundary = jnp.where(
            phi > 0,
            K * q2 / (q2 - 1.0),  # Call perpetual boundary
            K * q2 / (q2 - 1.0),  # Put perpetual boundary (q2 < 0 for puts)
        )
        # Clamp to reasonable range
        perp_boundary = jnp.where(phi > 0,
                                   jnp.maximum(perp_boundary, K * 1.01),
                                   jnp.minimum(perp_boundary, K * 0.99))
        perp_boundary = jnp.where(phi > 0,
                                   jnp.minimum(perp_boundary, K * 5.0),
                                   jnp.maximum(perp_boundary, K * 0.01))
        return perp_boundary

    s_star = _compute_s_star_init()

    # Newton refinement of exercise boundary
    def _refine(s_star, _):
        d1 = (jnp.log(s_star / K) + (r - q + 0.5 * sigma2) * T) / (sigma * sqrt_T)
        bs_val = black_scholes_price(s_star, K, T, r, q, sigma, option_type)
        intrinsic = jnp.maximum(phi * (s_star - K), 0.0)

        # At the boundary: intrinsic = bs + (1 - delta/q2) * (intrinsic - bs)
        # Rearranged: phi*S* - K = bs + A*(phi*S* - K - bs)
        delta = phi * jnp.exp(-q * T) * norm.cdf(phi * d1)

        # The boundary condition: C(S*) = phi*(S* - K)
        # With BAW: phi*(S* - K) = bs(S*) + A2 * (S*/S*_inf)^q2
        # Newton: f(S*) = bs(S*) + phi*S*/q2 * (1 - exp(-q*T)*N(phi*d1)) - phi*(S* - K) = 0
        f_val = bs_val + phi * s_star / q2 * (1.0 - jnp.exp(-q * T) * norm.cdf(phi * d1)) - intrinsic

        # Numerical derivative
        eps = s_star * 1e-5
        s_plus = s_star + eps
        d1_plus = (jnp.log(s_plus / K) + (r - q + 0.5 * sigma2) * T) / (sigma * sqrt_T)
        bs_plus = black_scholes_price(s_plus, K, T, r, q, sigma, option_type)
        intr_plus = jnp.maximum(phi * (s_plus - K), 0.0)
        f_plus = bs_plus + phi * s_plus / q2 * (1.0 - jnp.exp(-q * T) * norm.cdf(phi * d1_plus)) - intr_plus

        f_prime = (f_plus - f_val) / eps
        f_prime = jnp.where(jnp.abs(f_prime) < 1e-15, 1e-15, f_prime)

        s_star_new = s_star - f_val / f_prime

        # Clamp
        s_star_new = jnp.where(phi > 0,
                                jnp.clip(s_star_new, K * 1.001, K * 10.0),
                                jnp.clip(s_star_new, K * 0.01, K * 0.999))
        return s_star_new, None

    s_star, _ = jax.lax.scan(_refine, s_star, None, length=n_iter)

    # Compute the early exercise premium
    bs_star = black_scholes_price(s_star, K, T, r, q, sigma, option_type)
    intrinsic_star = jnp.maximum(phi * (s_star - K), 0.0)
    A2 = intrinsic_star - bs_star

    # American price = European + early exercise premium * (S/S*)^q2
    # For call (phi=1): if S < S*, premium = A2 * (S/S*)^q2; if S >= S*, use intrinsic
    # For put (phi=-1): if S > S*, premium = A2 * (S/S*)^q2_eff; if S <= S*, use intrinsic
    ratio = S / s_star
    power = jnp.where(phi > 0, q2, -q2)  # q2 > 0 for calls, q2 < 0 for puts
    ratio_safe = jnp.maximum(ratio, 1e-20)

    in_continuation = jnp.where(phi > 0, S < s_star, S > s_star)
    premium = jnp.where(in_continuation,
                         jnp.maximum(A2, 0.0) * jnp.power(ratio_safe, jnp.abs(q2)),
                         0.0)

    price = euro + premium

    # Ensure >= intrinsic
    intrinsic = jnp.maximum(phi * (S - K), 0.0)
    price = jnp.where(in_continuation, jnp.maximum(price, intrinsic), intrinsic)

    return price
