"""Merton-76 jump-diffusion process.

dS/S = (r - q - lambda*m) dt + sigma dW + (e^J - 1) dN
J ~ Normal(nu, delta^2),  N ~ Poisson(lambda*dt)
m = exp(nu + 0.5*delta^2) - 1
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.special import ndtri


@dataclass(frozen=True)
class Merton76Process:
    """Merton-76 jump-diffusion process.

    Parameters
    ----------
    spot : initial spot price
    rate : risk-free rate
    dividend : dividend yield
    volatility : diffusion volatility
    lambda_ : jump intensity (per year)
    nu : mean log-jump size
    delta : std dev of log-jump size
    """
    spot: float
    rate: float
    dividend: float
    volatility: float
    lambda_: float
    nu: float
    delta: float

    @property
    def m(self):
        """Jump compensator."""
        return jnp.exp(self.nu + 0.5 * self.delta**2) - 1.0

    def drift(self, t, x):
        return self.rate - self.dividend - 0.5 * self.volatility**2 - self.lambda_ * self.m

    def diffusion(self, t, x):
        return self.volatility

    def evolve(self, t0, x0, dt, dw):
        """Euler step in log-space.

        Parameters
        ----------
        x0 : log-spot
        dw : (dw_diffusion, dw_jump) — dw_diffusion is normal*sqrt(dt),
             dw_jump is uniform [0,1] for jump

        Returns
        -------
        log-spot at t0+dt
        """
        dw_diff, dw_jump = dw

        jump_prob = self.lambda_ * dt
        has_jump = (dw_jump < jump_prob).astype(jnp.float64)
        jump_size = self.nu + self.delta * ndtri(
            jnp.clip(dw_jump / jnp.maximum(jump_prob, 1e-10), 0.01, 0.99)
        )

        drift = self.drift(t0, x0) * dt
        diffusion = self.volatility * dw_diff
        jump = has_jump * jump_size

        return x0 + drift + diffusion + jump
