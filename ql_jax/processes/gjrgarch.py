"""GJR-GARCH(1,1) stochastic volatility process.

dS/S = (r - q) dt + sqrt(h) dW
h_{t+1} = omega + (beta + alpha + gamma*I_{t}) * h_t
          where I_t = 1 if return < 0

Annualized from daily parameters via daysPerYear.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class GJRGARCHProcess:
    """GJR-GARCH(1,1) stochastic volatility process.

    Parameters
    ----------
    spot : initial spot price
    rate : risk-free rate
    dividend : dividend yield
    v0 : initial daily variance
    omega : constant in variance equation
    alpha : coefficient on lagged squared return
    beta : coefficient on lagged variance
    gamma : leverage effect coefficient
    lambda_ : risk premium parameter
    days_per_year : annualization factor (default 252)
    """
    spot: float
    rate: float
    dividend: float
    v0: float
    omega: float
    alpha: float
    beta: float
    gamma: float
    lambda_: float
    days_per_year: float = 252.0

    def evolve(self, t0, state, dt, dw):
        """Euler step for (log_s, h).

        Parameters
        ----------
        state : (log_s, h) where h is the daily variance
        dw : (dw_s, dw_v) independent standard normals * sqrt(dt)

        Returns
        -------
        (log_s_new, h_new)
        """
        log_s, h = state
        dw_s = dw[0]

        h_pos = jnp.maximum(h, 0.0)
        sqrt_h = jnp.sqrt(h_pos)

        # Annualized variance
        h_annual = h_pos * self.days_per_year

        # Log-spot evolution
        log_s_new = (log_s
                     + (self.rate - self.dividend - 0.5 * h_annual) * dt
                     + jnp.sqrt(h_annual) * dw_s)

        # Daily return (standardized)
        z = dw_s / jnp.sqrt(jnp.maximum(dt, 1e-10))
        indicator = jnp.where(z < 0.0, 1.0, 0.0)

        # GARCH variance update
        h_new = (self.omega
                 + self.beta * h_pos
                 + self.alpha * h_pos * z**2
                 + self.gamma * h_pos * z**2 * indicator)

        return (log_s_new, h_new)
