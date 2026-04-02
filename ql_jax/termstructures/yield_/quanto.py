"""Quanto-adjusted yield term structure."""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.termstructures.yield_.base import YieldTermStructure


class QuantoTermStructure(YieldTermStructure):
    """Yield term structure adjusted for quanto effects.

    The adjustment accounts for the covariance between the
    exchange rate and the foreign short rate:
    r_quanto(t) = r_foreign(t) - sigma_fx * sigma_r * rho * t

    Parameters
    ----------
    underlying : YieldTermStructure — foreign yield curve
    sigma_fx : exchange rate volatility
    sigma_short_rate : foreign short rate volatility
    correlation : correlation between FX and short rate
    """

    def __init__(self, underlying, sigma_fx, sigma_short_rate, correlation):
        self._underlying = underlying
        self._sigma_fx = sigma_fx
        self._sigma_r = sigma_short_rate
        self._rho = correlation

    def discount(self, t):
        t = jnp.float64(t)
        base_df = self._underlying.discount(t)
        adjustment = self._sigma_fx * self._sigma_r * self._rho * t
        return base_df * jnp.exp(adjustment * t * 0.5)

    def zero_rate(self, t):
        t = jnp.maximum(jnp.float64(t), 1e-10)
        return -jnp.log(self.discount(t)) / t

    def forward_rate(self, t1, t2):
        t1, t2 = jnp.float64(t1), jnp.float64(t2)
        dt = jnp.maximum(t2 - t1, 1e-10)
        return -(jnp.log(self.discount(t2)) - jnp.log(self.discount(t1))) / dt
