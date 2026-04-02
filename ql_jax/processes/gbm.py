"""Geometric Brownian Motion process.

dS = mu*S dt + sigma*S dW

Standard GBM with constant drift and volatility.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class GeometricBrownianMotionProcess:
    """Geometric Brownian motion.

    Parameters
    ----------
    x0 : initial value
    mu : drift rate
    sigma : volatility
    """
    x0: float
    mu: float
    sigma: float

    def drift(self, t, x):
        return self.mu * x

    def diffusion(self, t, x):
        return self.sigma * x

    def evolve(self, t0, x0, dt, dw):
        """Exact step using log-normal formula.

        S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
        """
        z = dw / jnp.sqrt(dt)
        return x0 * jnp.exp((self.mu - 0.5 * self.sigma**2) * dt + self.sigma * jnp.sqrt(dt) * z)

    def expectation(self, t0, x0, dt):
        return x0 * jnp.exp(self.mu * dt)

    def variance(self, t0, x0, dt):
        return x0**2 * jnp.exp(2.0 * self.mu * dt) * (jnp.exp(self.sigma**2 * dt) - 1.0)

    def std_deviation(self, t0, x0, dt):
        return jnp.sqrt(self.variance(t0, x0, dt))
