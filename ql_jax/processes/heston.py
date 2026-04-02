"""Heston stochastic volatility process.

dS = (r - q)*S*dt + sqrt(V)*S*dW_S
dV = kappa*(theta - V)*dt + xi*sqrt(V)*dW_V
<dW_S, dW_V> = rho*dt
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class HestonProcess:
    """Heston stochastic volatility process.

    Parameters
    ----------
    spot : initial spot price
    rate : risk-free rate
    dividend : dividend yield
    v0 : initial variance
    kappa : mean reversion speed
    theta : long-run variance
    xi : vol-of-vol (sigma in some notations)
    rho : correlation between spot and variance Brownian motions
    """
    spot: float
    rate: float
    dividend: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def evolve(self, t0, state, dt, dw):
        """Euler step for (log_s, v).

        Parameters
        ----------
        state : (log_s, v)
        dw : (dw_s, dw_v) — independent standard normals * sqrt(dt)

        Returns
        -------
        (log_s_new, v_new)
        """
        log_s, v = state
        dw_s, dw_v = dw

        # Correlated Brownians
        dw_s_corr = dw_s
        dw_v_corr = self.rho * dw_s + jnp.sqrt(1.0 - self.rho**2) * dw_v

        # Ensure positive variance (full truncation)
        v_pos = jnp.maximum(v, 0.0)
        sqrt_v = jnp.sqrt(v_pos)

        # Log-spot evolution
        log_s_new = (log_s
                     + (self.rate - self.dividend - 0.5 * v_pos) * dt
                     + sqrt_v * dw_s_corr)

        # Variance evolution
        v_new = (v
                 + self.kappa * (self.theta - v_pos) * dt
                 + self.xi * sqrt_v * dw_v_corr)

        return (log_s_new, v_new)
