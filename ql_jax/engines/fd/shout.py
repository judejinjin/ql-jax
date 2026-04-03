"""Finite-difference engine for shout options."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

from ql_jax.methods.finitedifferences.operators import (
    TridiagonalOperator, d_zero, d_plus_d_minus,
)
from ql_jax.methods.finitedifferences.schemes import theta_step


def _bs_european(S, K_strike, tau, r, q, sigma, phi):
    """Vectorised Black-Scholes price (phi=+1 call, -1 put)."""
    sqrt_tau = jnp.sqrt(jnp.maximum(tau, 1e-14))
    F = S * jnp.exp((r - q) * tau)
    df = jnp.exp(-r * tau)
    d1 = (jnp.log(F / K_strike) + 0.5 * sigma**2 * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return df * phi * (F * jnorm.cdf(phi * d1) - K_strike * jnorm.cdf(phi * d2))


def fd_shout_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: int = 1,
    *,
    n_x: int = 200,
    n_t: int = 200,
    n_sd: float = 4.0,
) -> float:
    """Price a shout call/put via finite differences.

    A shout option allows the holder to "shout" once during the life,
    locking in the intrinsic value at that point while keeping the
    option alive for further upside.  At maturity, the payoff is
    ``max(locked_intrinsic, terminal_payoff)``.

    The shout value when in-the-money is::

        locked_intrinsic * df + BS_European(S, strike=S, tau)

    (lock in intrinsic paid at T *plus* the ATM-forward European option
    on the remaining life.)  When out-of-the-money, shouting gives
    the plain European value so the constraint is never binding.

    Parameters
    ----------
    option_type : +1 call, -1 put
    """
    dt = T / n_t
    x_min = jnp.log(S0) - n_sd * sigma * jnp.sqrt(T)
    x_max = jnp.log(S0) + n_sd * sigma * jnp.sqrt(T)
    x = jnp.linspace(x_min, x_max, n_x)
    S = jnp.exp(x)
    dx = x[1] - x[0]
    phi = jnp.float64(option_type)

    # BS operator in log-spot
    nu = r - q - 0.5 * sigma**2
    diff_coef = 0.5 * sigma**2
    D1 = d_zero(x)
    D2 = d_plus_d_minus(x)
    lower = diff_coef * D2.lower + nu * D1.lower
    diag_v = diff_coef * D2.diag + nu * D1.diag - r
    upper = diff_coef * D2.upper + nu * D1.upper
    L = TridiagonalOperator(lower, diag_v, upper)

    # Terminal payoff
    V = jnp.maximum(phi * (S - K), 0.0)

    for step in range(n_t):
        tau = (step + 1) * dt  # time remaining from current step to T

        # PDE continuation
        V_cont = theta_step(V, dt, L, theta=0.5)

        # Shout value at each grid point.
        # ITM: lock intrinsic at T + ATM European on remaining life
        intrinsic = jnp.maximum(phi * (S - K), 0.0)
        df = jnp.exp(-r * tau)
        # ATM European with strike = S_i (same d1/d2 for every S_i)
        atm_euro = _bs_european(S, S, tau, r, q, sigma, phi)
        shout_itm = intrinsic * df + atm_euro

        # OTM: shout gives European(S, K, tau) — never better than V_cont
        shout_otm = _bs_european(S, K, tau, r, q, sigma, phi)

        itm_mask = phi * (S - K) > 0
        shout_val = jnp.where(itm_mask, shout_itm, shout_otm)

        V = jnp.maximum(V_cont, shout_val)

    # Interpolate at S0
    idx = jnp.searchsorted(x, jnp.log(S0))
    idx = jnp.clip(idx, 1, n_x - 1)
    w = (jnp.log(S0) - x[idx - 1]) / dx
    price = V[idx - 1] * (1.0 - w) + V[idx] * w
    return float(price)
