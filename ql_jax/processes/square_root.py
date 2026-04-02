"""Square-root process (general CIR-like).

dx = speed*(level - x) dt + volatility*sqrt(x) dW

Alias for CIR process with clearer naming for use in Heston variance, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class SquareRootProcess:
    """Square-root diffusion process.

    Parameters
    ----------
    speed : mean reversion speed (kappa)
    volatility : diffusion coefficient (sigma)
    x0 : initial value
    level : long-run mean (theta)
    """
    speed: float
    volatility: float
    x0: float = 0.0
    level: float = 0.0

    def drift(self, t, x):
        return self.speed * (self.level - x)

    def diffusion(self, t, x):
        return self.volatility * jnp.sqrt(jnp.maximum(x, 0.0))

    def expectation(self, t0, x0, dt):
        ex = jnp.exp(-self.speed * dt)
        return self.level + (x0 - self.level) * ex

    def variance(self, t0, x0, dt):
        ex = jnp.exp(-self.speed * dt)
        k, s, l = self.speed, self.volatility, self.level
        return (x0 * s**2 * ex / k * (1.0 - ex)
                + l * s**2 / (2.0 * k) * (1.0 - ex)**2)

    def evolve(self, t0, x0, dt, dw):
        """Euler step with full truncation to ensure positivity."""
        x_pos = jnp.maximum(x0, 0.0)
        sqrt_x = jnp.sqrt(x_pos)
        x_new = x0 + self.speed * (self.level - x_pos) * dt + self.volatility * sqrt_x * dw
        return jnp.maximum(x_new, 0.0)
