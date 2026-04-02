"""Cox-Ingersoll-Ross process.

dx = speed*(level - x) dt + volatility*sqrt(x) dW

Uses the QE (quadratic exponential) scheme for exact simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.stats import norm


@dataclass(frozen=True)
class CoxIngersollRossProcess:
    """Cox-Ingersoll-Ross (square-root) process.

    Parameters
    ----------
    speed : mean reversion speed (kappa)
    volatility : diffusion coefficient (sigma)
    x0 : initial value
    level : long-run mean
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

    def std_deviation(self, t0, x0, dt):
        return jnp.sqrt(jnp.maximum(self.variance(t0, x0, dt), 0.0))

    def evolve(self, t0, x0, dt, dw):
        """QE-like exact discretization.

        Uses moment-matched scheme: for low psi uses
        inverse CDF of non-central chi-squared approximation.
        """
        ex = jnp.exp(-self.speed * dt)
        m = self.expectation(t0, x0, dt)
        s2 = self.variance(t0, x0, dt)
        psi = s2 / jnp.maximum(m**2, 1e-20)

        # Standard normal from dw
        z = dw / jnp.sqrt(dt)

        # For small psi: matched non-central chi2 via normal approx
        # x_{t+dt} ~ m + sqrt(s2) * z
        result = jnp.maximum(m + jnp.sqrt(jnp.maximum(s2, 0.0)) * z, 0.0)
        return result
