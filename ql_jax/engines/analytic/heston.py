"""Analytic Heston pricing engine (semi-analytic via characteristic function).

Uses the Heston (1993) characteristic function with numerical integration.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax

from ql_jax._util.types import OptionType


def heston_price(
    S, K, T, r, q,
    v0, kappa, theta, xi, rho,
    option_type: int,
    n_points: int = 128,
):
    """Semi-analytic Heston price via characteristic function integration.

    Uses the Heston (1993) P1/P2 formulation with Lord-Kahl (Formulation 2)
    for numerical stability.

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    v0 : initial variance
    kappa : mean reversion speed
    theta : long-run variance
    xi : vol-of-vol
    rho : correlation
    option_type : OptionType.Call or OptionType.Put
    n_points : number of quadrature points
    """
    S, K, T, r, q = (jnp.asarray(x, dtype=jnp.float64) for x in (S, K, T, r, q))
    v0, kappa, theta, xi, rho = (jnp.asarray(x, dtype=jnp.float64)
                                  for x in (v0, kappa, theta, xi, rho))

    log_moneyness = jnp.log(S / K)
    df = jnp.exp(-r * T)

    def _char_func(u, b, uj):
        """Heston CF for probability P_j.

        Uses Formulation 2 (Lord-Kahl) with exp(-d*T) for stability.
        b = b_j, uj = u_j from Heston (1993).
        """
        d = jnp.sqrt(
            (rho * xi * 1j * u - b) ** 2
            - xi ** 2 * (2.0 * uj * 1j * u - u ** 2)
        )

        g = (b - rho * xi * 1j * u - d) / (b - rho * xi * 1j * u + d)
        exp_neg_dT = jnp.exp(-d * T)

        C = (r - q) * 1j * u * T + (kappa * theta / xi ** 2) * (
            (b - rho * xi * 1j * u - d) * T
            - 2.0 * jnp.log((1.0 - g * exp_neg_dT) / (1.0 - g))
        )
        D = ((b - rho * xi * 1j * u - d) / xi ** 2) * (
            (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
        )

        return jnp.exp(C + D * v0 + 1j * u * log_moneyness)

    # Integration grid (trapezoidal)
    du = 0.25
    u_vals = jnp.arange(1, n_points + 1, dtype=jnp.float64) * du

    # P1: b1 = kappa - rho*xi, u1 = 0.5
    b1 = kappa - rho * xi
    cf1_vals = jax.vmap(lambda u: _char_func(u, b1, 0.5))(u_vals)
    int1 = jnp.sum(jnp.real(cf1_vals / (1j * u_vals))) * du

    # P2: b2 = kappa, u2 = -0.5
    b2 = kappa
    cf2_vals = jax.vmap(lambda u: _char_func(u, b2, -0.5))(u_vals)
    int2 = jnp.sum(jnp.real(cf2_vals / (1j * u_vals))) * du

    P1 = 0.5 + int1 / jnp.pi
    P2 = 0.5 + int2 / jnp.pi

    call_price = S * jnp.exp(-q * T) * P1 - K * df * P2

    price = jnp.where(
        option_type == OptionType.Call,
        call_price,
        call_price - S * jnp.exp(-q * T) + K * df,
    )
    return price


def heston_price_cos(
    S, K, T, r, q,
    v0, kappa, theta, xi, rho,
    option_type: int,
    N: int = 128,
    L: float = 12.0,
):
    """Heston price via COS method (Fang-Oosterlee).

    Fast and accurate for European options.
    """
    S, K, T, r, q = (jnp.asarray(x, dtype=jnp.float64) for x in (S, K, T, r, q))
    v0, kappa, theta, xi, rho = (jnp.asarray(x, dtype=jnp.float64)
                                  for x in (v0, kappa, theta, xi, rho))

    x = jnp.log(S / K) + (r - q) * T

    # Truncation range
    c1 = (r - q) * T + (1.0 - jnp.exp(-kappa * T)) * (theta - v0) / (2.0 * kappa) - 0.5 * theta * T
    c2 = (1.0 / (8.0 * kappa**3)) * (
        xi * T * kappa * jnp.exp(-kappa * T) * (v0 - theta) * (8.0 * kappa * rho - 4.0 * xi)
        + kappa * rho * xi * (1.0 - jnp.exp(-kappa * T)) * (16.0 * theta - 8.0 * v0)
        + 2.0 * theta * kappa * T * (-4.0 * kappa * rho * xi + xi**2 + 4.0 * kappa**2)
        + xi**2 * ((theta - 2.0 * v0) * jnp.exp(-2.0 * kappa * T)
                    + theta * (6.0 * jnp.exp(-kappa * T) - 7.0) + 2.0 * v0)
        + 8.0 * kappa**2 * (v0 - theta) * (1.0 - jnp.exp(-kappa * T))
    )
    c2 = jnp.abs(c2)

    a = x + c1 - L * jnp.sqrt(c2)
    b = x + c1 + L * jnp.sqrt(c2)

    k_arr = jnp.arange(0, N, dtype=jnp.float64)

    def heston_cf(u):
        """Heston characteristic function of log(S_T/K)."""
        d = jnp.sqrt((kappa - 1j * rho * xi * u)**2 + xi**2 * (u**2 + 1j * u))
        g = (kappa - 1j * rho * xi * u - d) / (kappa - 1j * rho * xi * u + d)
        exp_dT = jnp.exp(-d * T)

        C = (kappa * theta / xi**2) * (
            (kappa - 1j * rho * xi * u - d) * T
            - 2.0 * jnp.log((1.0 - g * exp_dT) / (1.0 - g))
        )
        D = ((kappa - 1j * rho * xi * u - d) / xi**2) * (
            (1.0 - exp_dT) / (1.0 - g * exp_dT)
        )
        return jnp.exp(C + D * v0 + 1j * u * x)

    # COS coefficients for call payoff
    def chi(k, c, d):
        u = k * jnp.pi / (b - a)
        return (1.0 / (1.0 + u**2)) * (
            jnp.cos(u * (d - a)) * jnp.exp(d)
            - jnp.cos(u * (c - a)) * jnp.exp(c)
            + u * jnp.sin(u * (d - a)) * jnp.exp(d)
            - u * jnp.sin(u * (c - a)) * jnp.exp(c)
        )

    def psi(k, c, d):
        u = k * jnp.pi / (b - a)
        return jnp.where(
            k == 0,
            d - c,
            (jnp.sin(u * (d - a)) - jnp.sin(u * (c - a))) / u,
        )

    if option_type == OptionType.Call:
        V_k = 2.0 / (b - a) * (chi(k_arr, 0.0, b) - psi(k_arr, 0.0, b))
    else:
        V_k = 2.0 / (b - a) * (-chi(k_arr, a, 0.0) + psi(k_arr, a, 0.0))

    # Characteristic function at COS frequencies
    cf_vals = jnp.array([heston_cf(k * jnp.pi / (b - a)) for k in k_arr])
    re_cf = jnp.real(cf_vals * jnp.exp(-1j * k_arr * jnp.pi * a / (b - a)))

    # First term has weight 0.5
    weights = jnp.ones(N)
    weights = weights.at[0].set(0.5)

    price = K * jnp.exp(-r * T) * jnp.sum(weights * re_cf * V_k)
    return price
