"""Heston model pricing variants.

Includes COS method, Bates model, and expansion approximations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# COS method for Heston (Fang & Oosterlee 2008)
# ---------------------------------------------------------------------------

def heston_price_cos(
    S: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    option_type: int = 1, N: int = 256,
) -> float:
    """Heston price via COS (Fourier-cosine series) method.

    Parameters
    ----------
    S, K, T, r, q : standard parameters
    v0 : initial variance
    kappa : mean-reversion speed
    theta : long-run variance
    sigma_v : vol of vol
    rho : correlation
    option_type : 1 for call, -1 for put
    N : number of Fourier terms
    """
    S, K, T, r, q = (jnp.float64(x) for x in (S, K, T, r, q))
    v0, kappa, theta, sigma_v, rho = (
        jnp.float64(x) for x in (v0, kappa, theta, sigma_v, rho)
    )

    x = jnp.log(S / K) + (r - q) * T

    # Truncation range
    c1 = (r - q) * T + (1.0 - jnp.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T
    c2 = (sigma_v ** 2 / (8.0 * kappa ** 3)) * (
        (1.0 - jnp.exp(-2.0 * kappa * T)) * (sigma_v ** 2 - 4.0 * kappa * theta)
        + 2.0 * kappa * T * (4.0 * kappa * theta - sigma_v ** 2)
    ) + theta * sigma_v ** 2 * T / (4.0 * kappa)
    c2 = jnp.maximum(c2, 1e-8)

    L = 12.0
    a = c1 - L * jnp.sqrt(c2)
    b = c1 + L * jnp.sqrt(c2)

    # Characteristic function of log(S_T/K) under Heston
    def cf_heston(u):
        """Characteristic function."""
        d = jnp.sqrt((rho * sigma_v * 1j * u - kappa) ** 2 + sigma_v ** 2 * (1j * u + u ** 2))
        g = (kappa - rho * sigma_v * 1j * u - d) / (kappa - rho * sigma_v * 1j * u + d)
        C = (r - q) * 1j * u * T + kappa * theta / sigma_v ** 2 * (
            (kappa - rho * sigma_v * 1j * u - d) * T - 2.0 * jnp.log((1.0 - g * jnp.exp(-d * T)) / (1.0 - g))
        )
        D = (kappa - rho * sigma_v * 1j * u - d) / sigma_v ** 2 * (
            (1.0 - jnp.exp(-d * T)) / (1.0 - g * jnp.exp(-d * T))
        )
        return jnp.exp(C + D * v0)

    # COS coefficients for put payoff (more stable)
    def chi(c, d, k):
        """Chi coefficients."""
        k_pi = k * jnp.pi / (b - a)
        return (1.0 / (1.0 + k_pi ** 2)) * (
            jnp.cos(k_pi * (d - a)) * jnp.exp(d) - jnp.cos(k_pi * (c - a)) * jnp.exp(c)
            + k_pi * jnp.sin(k_pi * (d - a)) * jnp.exp(d) - k_pi * jnp.sin(k_pi * (c - a)) * jnp.exp(c)
        )

    def psi(c, d, k):
        """Psi coefficients."""
        k_pi = k * jnp.pi / (b - a)
        return jnp.where(
            k == 0,
            d - c,
            (jnp.sin(k_pi * (d - a)) - jnp.sin(k_pi * (c - a))) / k_pi,
        )

    # Compute via put then use put-call parity if needed
    ks = jnp.arange(N, dtype=jnp.float64)

    # V_k for put
    V_k_put = (2.0 / (b - a)) * (-chi(a, 0.0, ks) + psi(a, 0.0, ks))

    # CF values
    u_k = ks * jnp.pi / (b - a)
    cf_vals = jax.vmap(cf_heston)(u_k)
    re_vals = jnp.real(cf_vals * jnp.exp(-1j * ks * a * jnp.pi / (b - a)))

    # 1/2 for k=0 term
    weights = jnp.where(ks == 0, 0.5, 1.0)

    put = K * jnp.exp(-r * T) * jnp.sum(weights * re_vals * V_k_put)
    put = jnp.maximum(put, 0.0)

    # Call via put-call parity
    call = put + S * jnp.exp(-q * T) - K * jnp.exp(-r * T)

    return jnp.where(option_type == 1, jnp.maximum(call, 0.0), jnp.maximum(put, 0.0))


# ---------------------------------------------------------------------------
# Bates model (Heston + jumps)
# ---------------------------------------------------------------------------

def bates_price(
    S: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    jump_intensity: float, jump_mean: float, jump_vol: float,
    option_type: int = 1, N: int = 64,
) -> float:
    """Bates (1996) stochastic volatility jump-diffusion European price.

    Extends Heston with Merton-style jumps via Fourier inversion.

    Parameters
    ----------
    S, K, T, r, q : standard parameters
    v0, kappa, theta, sigma_v, rho : Heston parameters
    jump_intensity : jump arrival rate (lambda)
    jump_mean : mean log jump size
    jump_vol : jump size volatility
    option_type : 1 for call, -1 for put
    N : quadrature points for Fourier integral
    """
    S, K, T = jnp.float64(S), jnp.float64(K), jnp.float64(T)
    r, q = jnp.float64(r), jnp.float64(q)

    mean_j = jnp.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0
    x = jnp.log(S / K)

    def cf_bates(u):
        """Log characteristic function of Bates model."""
        # Heston part
        alpha = -0.5 * u * (u + 1j)
        beta = kappa - rho * sigma_v * 1j * u
        gamma = 0.5 * sigma_v ** 2
        d = jnp.sqrt(beta ** 2 - 4.0 * alpha * gamma)
        r_plus = (beta + d) / (2.0 * gamma)
        r_minus = (beta - d) / (2.0 * gamma)
        g = r_minus / r_plus

        D = r_minus * (1.0 - jnp.exp(-d * T)) / (1.0 - g * jnp.exp(-d * T))
        C = kappa * (r_minus * T - 2.0 / sigma_v ** 2 * jnp.log(
            (1.0 - g * jnp.exp(-d * T)) / (1.0 - g)
        ))

        # Jump part
        jump_cf = jump_intensity * T * (
            jnp.exp(1j * u * jump_mean - 0.5 * jump_vol ** 2 * u ** 2) - 1.0
            - 1j * u * mean_j
        )

        return jnp.exp(
            C * theta + D * v0 + 1j * u * (x + (r - q - jump_intensity * mean_j) * T)
            + jump_cf
        )

    # Fourier integral (Gil-Pelaez inversion)
    du = 0.5
    us = jnp.arange(1, N + 1, dtype=jnp.float64) * du

    def integrand(u):
        cf_val = cf_bates(u - 0.5j)
        return jnp.real(cf_val * jnp.exp(-1j * u * x)) / (u ** 2 + 0.25)

    vals = jax.vmap(integrand)(us)
    integral = jnp.sum(vals) * du

    call = S * jnp.exp(-q * T) - jnp.sqrt(S * K) * jnp.exp(-r * T) / jnp.pi * integral
    call = jnp.maximum(call, 0.0)
    put = call - S * jnp.exp(-q * T) + K * jnp.exp(-r * T)

    return jnp.where(option_type == 1, call, jnp.maximum(put, 0.0))


# ---------------------------------------------------------------------------
# Heston expansion approximation (Lewis / Gatheral)
# ---------------------------------------------------------------------------

def heston_expansion_price(
    S: float, K: float, T: float, r: float, q: float,
    v0: float, kappa: float, theta: float, sigma_v: float, rho: float,
    option_type: int = 1,
) -> float:
    """Heston price via second-order expansion (Gatheral 2006).

    Fast approximation suitable for calibration inner loops.
    Uses implied vol expansion to second order in vol-of-vol.
    """
    # Effective variance
    v_bar = theta + (v0 - theta) * (1.0 - jnp.exp(-kappa * T)) / (kappa * T)

    # Second-order correction to implied vol
    sigma_bs = jnp.sqrt(v_bar)

    # Skew correction
    log_m = jnp.log(S * jnp.exp((r - q) * T) / K)
    skew_corr = rho * sigma_v / (2.0 * sigma_bs) * log_m

    # Convexity correction
    convex_corr = (sigma_v ** 2 / (12.0 * sigma_bs)) * (
        2.0 - 3.0 * rho ** 2
    ) * log_m ** 2 / (sigma_bs ** 2 * T)

    sigma_impl = sigma_bs + skew_corr + convex_corr
    sigma_impl = jnp.maximum(sigma_impl, 1e-6)

    from ql_jax.engines.analytic.black_formula import black_scholes_price
    return black_scholes_price(S, K, T, r, q, sigma_impl, option_type)
