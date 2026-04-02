"""Hull-White short-rate process.

dr = (theta(t) - a*r) dt + sigma dW
where theta(t) is chosen to match the initial term structure.

This wraps an Ornstein-Uhlenbeck process for the state variable x,
with r(t) = x(t) + alpha(t), where alpha(t) is a deterministic shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class HullWhiteProcess:
    """Hull-White (extended Vasicek) short-rate process.

    Parameters
    ----------
    a : mean reversion speed
    sigma : volatility
    rate_curve : yield term structure (must have .forward_rate(t) or .zero_rate(t))
                 Used to compute the deterministic shift alpha(t).
    """
    a: float
    sigma: float
    rate_curve: object = None

    def alpha(self, t):
        """Deterministic shift to match initial term structure.

        alpha(t) = f(0,t) + sigma^2/(2*a^2) * (1 - exp(-a*t))^2
        where f(0,t) is the instantaneous forward rate.
        """
        if self.rate_curve is not None and hasattr(self.rate_curve, 'forward_rate'):
            f = self.rate_curve.forward_rate(t)
        else:
            f = 0.05  # fallback
        return f + self.sigma**2 / (2.0 * self.a**2) * (1.0 - jnp.exp(-self.a * t))**2

    def drift(self, t, x):
        """Drift of the state variable x: -a*x."""
        return -self.a * x

    def diffusion(self, t, x):
        return self.sigma

    def expectation(self, t0, x0, dt):
        """E[x(t+dt) | x(t)] = x(t) * exp(-a*dt)."""
        return x0 * jnp.exp(-self.a * dt)

    def variance(self, t0, x0, dt):
        """Var[x(t+dt) | x(t)]."""
        return self.sigma**2 / (2.0 * self.a) * (1.0 - jnp.exp(-2.0 * self.a * dt))

    def std_deviation(self, t0, x0, dt):
        return jnp.sqrt(self.variance(t0, x0, dt))

    def evolve(self, t0, x0, dt, dw):
        """Exact step for state variable x.

        Returns x(t+dt). To get the short rate: r = x + alpha(t+dt).
        """
        return self.expectation(t0, x0, dt) + self.std_deviation(t0, x0, dt) * dw / jnp.sqrt(dt)

    def short_rate(self, t, x):
        """Short rate r(t) = x(t) + alpha(t)."""
        return x + self.alpha(t)

    def B(self, t, T):
        """Bond coefficient B(t,T) = (1 - exp(-a*(T-t))) / a."""
        return (1.0 - jnp.exp(-self.a * (T - t))) / self.a
