"""Heston stochastic local volatility (SLV) process.

dS = (r - q) S dt + sqrt(V) * L(t, S) * S dW_S
dV = kappa*(theta - V) dt + xi*sqrt(V) dW_V
<dW_S, dW_V> = rho dt

L(t, S) is the leverage function (local vol / Heston vol).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class HestonSLVProcess:
    """Heston stochastic local volatility process.

    Parameters
    ----------
    spot : initial spot price
    rate : risk-free rate
    dividend : dividend yield
    v0 : initial variance
    kappa : mean reversion speed
    theta : long-run variance
    xi : vol-of-vol
    rho : correlation
    leverage_fn : callable(t, s) -> leverage multiplier L(t,s),
                  or None for pure Heston (L=1)
    """
    spot: float
    rate: float
    dividend: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    leverage_fn: object = None

    def _leverage(self, t, s):
        if self.leverage_fn is not None:
            return self.leverage_fn(t, s)
        return 1.0

    def evolve(self, t0, state, dt, dw):
        """Euler step for (log_s, v).

        Parameters
        ----------
        state : (log_s, v)
        dw : (dw_s, dw_v) independent standard normals * sqrt(dt)

        Returns
        -------
        (log_s_new, v_new)
        """
        log_s, v = state
        dw_s, dw_v = dw

        # Correlated Brownians
        dw_s_corr = dw_s
        dw_v_corr = self.rho * dw_s + jnp.sqrt(1.0 - self.rho**2) * dw_v

        # Positive variance
        v_pos = jnp.maximum(v, 0.0)
        sqrt_v = jnp.sqrt(v_pos)

        # Leverage function
        s = jnp.exp(log_s)
        lev = self._leverage(t0, s)

        # Log-spot evolution with local vol
        log_s_new = (log_s
                     + (self.rate - self.dividend - 0.5 * v_pos * lev**2) * dt
                     + sqrt_v * lev * dw_s_corr)

        # Variance evolution (same as Heston)
        v_new = v + self.kappa * (self.theta - v_pos) * dt + self.xi * sqrt_v * dw_v_corr

        return (log_s_new, v_new)
