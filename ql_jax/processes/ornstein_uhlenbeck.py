"""Ornstein-Uhlenbeck process.

dx = speed*(level - x) dt + volatility dW

Exact discretization:
  E[x(t+dt)|x(t)] = level + (x(t) - level) * exp(-speed*dt)
  Var[x(t+dt)|x(t)] = vol^2/(2*speed) * (1 - exp(-2*speed*dt))
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class OrnsteinUhlenbeckProcess:
    """Ornstein-Uhlenbeck (mean-reverting) process.

    Parameters
    ----------
    speed : mean reversion speed
    volatility : diffusion coefficient
    x0 : initial value
    level : long-run mean level
    """
    speed: float
    volatility: float
    x0: float = 0.0
    level: float = 0.0

    def drift(self, t, x):
        return self.speed * (self.level - x)

    def diffusion(self, t, x):
        return self.volatility

    def expectation(self, t0, x0, dt):
        return self.level + (x0 - self.level) * jnp.exp(-self.speed * dt)

    def variance(self, t0, x0, dt):
        return (self.volatility**2 / (2.0 * self.speed)
                * (1.0 - jnp.exp(-2.0 * self.speed * dt)))

    def std_deviation(self, t0, x0, dt):
        return jnp.sqrt(self.variance(t0, x0, dt))

    def evolve(self, t0, x0, dt, dw):
        """Exact step: x(t+dt) = E[x] + stddev * Z."""
        return self.expectation(t0, x0, dt) + self.std_deviation(t0, x0, dt) * dw / jnp.sqrt(dt)
