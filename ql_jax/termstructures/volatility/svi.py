"""SVI (Stochastic Volatility Inspired) smile interpolation.

The raw SVI parameterization:
    w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

where k = log(K/F) is log-moneyness and w = sigma_BS^2 * T is total variance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from ql_jax.termstructures.volatility.smile_section import SmileSection


def svi_total_variance(k, a: float, b: float, rho: float, m: float, sigma: float) -> float:
    """Raw SVI total implied variance w(k).

    Parameters
    ----------
    k : log-moneyness ln(K/F)
    a, b, rho, m, sigma : SVI parameters
        a : overall variance level
        b : slope parameter (b >= 0)
        rho : correlation (-1 < rho < 1)
        m : centering parameter
        sigma : curvature (sigma > 0)
    """
    k = jnp.asarray(k, dtype=jnp.float64)
    diff = k - m
    return a + b * (rho * diff + jnp.sqrt(diff ** 2 + sigma ** 2))


def svi_implied_vol(k, t: float, a: float, b: float, rho: float,
                    m: float, sigma: float):
    """SVI implied Black volatility: sigma_BS = sqrt(w(k) / T)."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    w = jnp.maximum(w, 0.0)
    return jnp.sqrt(w / t)


def calibrate_svi(strikes, market_vols, forward: float, expiry: float,
                  initial_params=None):
    """Calibrate raw SVI parameters to market quotes.

    Parameters
    ----------
    strikes : array of strikes
    market_vols : array of corresponding Black implied vols
    forward : forward price
    expiry : time to expiry
    initial_params : dict with keys a, b, rho, m, sigma (optional)

    Returns
    -------
    dict with calibrated parameters: a, b, rho, m, sigma
    """
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    market_vols = jnp.asarray(market_vols, dtype=jnp.float64)
    k = jnp.log(strikes / forward)
    target_w = market_vols ** 2 * expiry

    if initial_params is None:
        # Reasonable defaults
        atm_var = float(jnp.interp(0.0, k, target_w))
        initial_params = {
            'a': atm_var * 0.5, 'b': 0.1, 'rho': -0.3,
            'm': 0.0, 'sigma': 0.3,
        }

    # Simple Levenberg-Marquardt via JAX optimization
    from ql_jax.math.optimization.levenberg_marquardt import minimize as lm_minimize

    def residuals(params_vec):
        a_, b_, rho_, m_, sigma_ = params_vec
        b_ = jnp.abs(b_)
        sigma_ = jnp.abs(sigma_) + 1e-8
        rho_ = jnp.tanh(rho_)  # constrain to (-1, 1)
        w_model = svi_total_variance(k, a_, b_, rho_, m_, sigma_)
        return w_model - target_w

    x0 = jnp.array([
        initial_params['a'], initial_params['b'],
        jnp.arctanh(jnp.clip(initial_params['rho'], -0.99, 0.99)),
        initial_params['m'], initial_params['sigma'],
    ])

    try:
        x, _, _ = lm_minimize(residuals, x0)
    except Exception:
        x = x0

    return {
        'a': float(x[0]),
        'b': float(jnp.abs(x[1])),
        'rho': float(jnp.tanh(x[2])),
        'm': float(x[3]),
        'sigma': float(jnp.abs(x[4])) + 1e-8,
    }


class SviSmileSection(SmileSection):
    """SmileSection implementation using SVI parameterization."""

    def __init__(self, expiry_time: float, forward: float,
                 a: float, b: float, rho: float, m: float, sigma: float):
        super().__init__(expiry_time, forward)
        self._a = a
        self._b = b
        self._rho = rho
        self._m = m
        self._sigma = sigma

    def volatility(self, strike):
        strike = jnp.asarray(strike, dtype=jnp.float64)
        k = jnp.log(strike / self._forward)
        return svi_implied_vol(k, self._expiry_time,
                               self._a, self._b, self._rho,
                               self._m, self._sigma)

    @classmethod
    def from_market_data(cls, expiry_time, forward, strikes, vols):
        """Create SviSmileSection by calibrating to market data."""
        params = calibrate_svi(strikes, vols, forward, expiry_time)
        return cls(expiry_time, forward, **params)
