"""Black-Scholes process — geometric Brownian motion."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class BlackScholesProcess:
    """Geometric Brownian motion: dS = (r-q)*S*dt + sigma*S*dW.

    All methods are pure functions suitable for JAX tracing.
    """
    spot: float
    rate: float       # risk-free rate (cc)
    dividend: float   # dividend yield (cc)
    volatility: float

    def drift(self, t, x):
        """Drift coefficient: (r - q - 0.5*sigma^2)."""
        _ = t
        return (self.rate - self.dividend - 0.5 * self.volatility**2)

    def diffusion(self, t, x):
        """Diffusion coefficient: sigma."""
        _ = t, x
        return self.volatility

    def evolve(self, t0, x0, dt, dw):
        """Euler step in log-space: log(S_{t+dt}) = log(S_t) + drift*dt + sigma*dW.

        Parameters
        ----------
        t0 : current time
        x0 : current log-spot
        dt : time step
        dw : standard normal random variable * sqrt(dt)

        Returns
        -------
        log-spot at t0+dt
        """
        return x0 + self.drift(t0, x0) * dt + self.diffusion(t0, x0) * dw

    def spot_at(self, t0, s0, dt, dw):
        """Evolve spot price directly."""
        log_s = jnp.log(s0)
        log_s_new = self.evolve(t0, log_s, dt, dw)
        return jnp.exp(log_s_new)


@dataclass(frozen=True)
class GeneralizedBlackScholesProcess:
    """Black-Scholes with term structure handles.

    Attributes
    ----------
    spot : float
    rate_curve : YieldTermStructure
    dividend_curve : YieldTermStructure
    vol_surface : BlackVolTermStructure (or just a float)
    """
    spot: float
    rate_curve: object = None
    dividend_curve: object = None
    vol_surface: object = None

    def local_drift(self, t, x):
        r = self.rate_curve.zero_rate(t) if self.rate_curve else 0.0
        q = self.dividend_curve.zero_rate(t) if self.dividend_curve else 0.0
        v = self._vol_at(t, jnp.exp(x))
        return r - q - 0.5 * v**2

    def local_diffusion(self, t, x):
        return self._vol_at(t, jnp.exp(x))

    def _vol_at(self, t, s):
        if self.vol_surface is None:
            return 0.2  # default
        if hasattr(self.vol_surface, "black_vol"):
            return self.vol_surface.black_vol(t, s)
        if isinstance(self.vol_surface, (int, float)):
            return float(self.vol_surface)
        return 0.2
