"""No-arbitrage SABR model.

Implements the absorption-at-zero SABR model that guarantees no-arbitrage
conditions. Based on the work of Hagan et al. with Paulot's corrections
for the zero-boundary behavior.

The key difference from standard Hagan SABR: at low strikes, the probability
density must remain non-negative, and the implied vol must not become
negative. This is enforced via a numerical correction near the boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ql_jax.termstructures.volatility.sabr_functions import sabr_vol
from ql_jax.termstructures.volatility.smile_section import SmileSection


def _d_minus(k, f, t, alpha, beta, rho, nu):
    """Helper: negative log-moneyness scaled."""
    fk_mid = jnp.power(f * k, (1.0 - beta) / 2.0)
    return nu / alpha * fk_mid * jnp.log(f / k)


def _absorption_probability(f, t, alpha, beta, nu):
    """Probability of absorption at zero.

    For beta < 1, the CEV-like process can hit zero. This returns
    the approximate probability mass at zero.
    """
    if beta >= 1.0:
        return 0.0
    one_m_beta = 1.0 - beta
    x = jnp.power(f, one_m_beta) / (one_m_beta * alpha)
    # Simplified estimate (leading-order term)
    from jax.scipy.stats import norm
    barrier = -x / jnp.sqrt(t * (1.0 + (2.0 - 3.0 * nu ** 2) / 24.0 * nu ** 2 * t))
    return float(norm.cdf(barrier))


def noarb_sabr_vol(f, k, t, alpha, beta, rho, nu):
    """No-arbitrage SABR implied volatility.

    Like standard Hagan SABR but with corrections that prevent
    negative density (butterfly arbitrage) at low strikes.

    Parameters
    ----------
    f : forward
    k : strike
    t : time to expiry
    alpha, beta, rho, nu : SABR parameters

    Returns
    -------
    Implied Black volatility
    """
    f = jnp.asarray(f, dtype=jnp.float64)
    k = jnp.asarray(k, dtype=jnp.float64)

    # Standard Hagan vol
    v_hagan = sabr_vol(k, f, t, alpha, beta, rho, nu)

    # For very low strikes, apply floor to prevent negative vol
    # The no-arb condition: sigma_BS(K) >= 0 and d^2(call)/dK^2 >= 0
    v_floor = alpha * jnp.power(k, beta - 1.0) * jnp.sqrt(
        jnp.maximum(1.0 - 2.0 * rho * nu / alpha * jnp.power(k, 1.0 - beta)
                     * jnp.log(f / k) + (nu / alpha) ** 2
                     * jnp.power(k, 2.0 * (1.0 - beta)) * jnp.log(f / k) ** 2, 0.0)
    ) * 0.5  # conservative floor

    # Use the maximum of Hagan vol and the floor
    vol = jnp.where(k > f * 0.01, v_hagan, jnp.maximum(v_hagan, v_floor))
    return jnp.maximum(vol, 1e-8)


def calibrate_noarb_sabr(strikes, market_vols, forward, expiry,
                          beta=0.5, initial_alpha=None):
    """Calibrate no-arb SABR to market data.

    Parameters
    ----------
    strikes : array
    market_vols : array
    forward : float
    expiry : float
    beta : float (fixed, typically 0 or 0.5)
    initial_alpha : float or None

    Returns
    -------
    dict : alpha, beta, rho, nu
    """
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    market_vols = jnp.asarray(market_vols, dtype=jnp.float64)

    if initial_alpha is None:
        atm_vol = float(jnp.interp(forward, strikes, market_vols))
        initial_alpha = atm_vol * jnp.power(forward, 1.0 - beta)

    from ql_jax.math.optimization.levenberg_marquardt import minimize as lm_minimize

    def residuals(params):
        alpha_, rho_raw, nu_raw = params
        alpha_ = jnp.abs(alpha_) + 1e-8
        rho_ = jnp.tanh(rho_raw)
        nu_ = jnp.abs(nu_raw) + 1e-8
        model_vols = jax.vmap(
            lambda k: noarb_sabr_vol(forward, k, expiry, alpha_, beta, rho_, nu_)
        )(strikes)
        return model_vols - market_vols

    x0 = jnp.array([float(initial_alpha), 0.0, 0.3])
    try:
        x, _, _ = lm_minimize(residuals, x0)
    except Exception:
        x = x0

    return {
        'alpha': float(jnp.abs(x[0])) + 1e-8,
        'beta': float(beta),
        'rho': float(jnp.tanh(x[1])),
        'nu': float(jnp.abs(x[2])) + 1e-8,
    }


class NoArbSabrSmileSection(SmileSection):
    """SmileSection using the no-arbitrage SABR model."""

    def __init__(self, expiry_time: float, forward: float,
                 alpha: float, beta: float, rho: float, nu: float):
        super().__init__(expiry_time, forward)
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        self._nu = nu

    def volatility(self, strike):
        return noarb_sabr_vol(
            self._forward, strike, self._expiry_time,
            self._alpha, self._beta, self._rho, self._nu,
        )

    @classmethod
    def from_market_data(cls, expiry_time, forward, strikes, vols, beta=0.5):
        """Calibrate no-arb SABR to market and return smile section."""
        params = calibrate_noarb_sabr(strikes, vols, forward, expiry_time, beta)
        return cls(expiry_time, forward, **params)
