"""Smile section base and implementations.

A SmileSection describes the volatility smile at a particular expiry.
"""

import jax.numpy as jnp

from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval
from ql_jax.termstructures.volatility.sabr_functions import sabr_vol


class SmileSection:
    """Base class for smile sections."""

    def __init__(self, expiry_time: float, forward: float = None):
        self._expiry_time = expiry_time
        self._forward = forward

    @property
    def expiry_time(self):
        return self._expiry_time

    def volatility(self, strike):
        """Return implied volatility at strike. Override in subclass."""
        raise NotImplementedError

    def variance(self, strike):
        vol = self.volatility(strike)
        return vol * vol * self._expiry_time


class FlatSmileSection(SmileSection):
    """Constant volatility across all strikes."""

    def __init__(self, expiry_time: float, vol: float, forward: float = None):
        super().__init__(expiry_time, forward)
        self._vol = jnp.asarray(vol, dtype=jnp.float64)

    def volatility(self, strike):
        return self._vol


class InterpolatedSmileSection(SmileSection):
    """Smile section from interpolated volatilities at given strikes."""

    def __init__(self, expiry_time: float, strikes, vols, forward: float = None):
        super().__init__(expiry_time, forward)
        self._strikes = jnp.array(strikes, dtype=jnp.float64)
        self._vols = jnp.array(vols, dtype=jnp.float64)
        self._interp_state = linear_build(self._strikes, self._vols)

    def volatility(self, strike):
        strike = jnp.asarray(strike, dtype=jnp.float64)
        return linear_eval(self._interp_state, strike)


class SABRSmileSection(SmileSection):
    """SABR-parameterized smile section."""

    def __init__(self, expiry_time: float, forward: float,
                 alpha: float, beta: float, rho: float, nu: float):
        super().__init__(expiry_time, forward)
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        self._nu = nu

    def volatility(self, strike):
        return sabr_vol(
            strike, self._forward, self._expiry_time,
            self._alpha, self._beta, self._rho, self._nu,
        )
