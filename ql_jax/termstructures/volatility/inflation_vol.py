"""Inflation volatility structures: CPI vol, YoY optionlet vol."""

from __future__ import annotations

import jax.numpy as jnp


class CPIVolatilityStructure:
    """Base class for CPI (zero-coupon inflation) volatility.

    Interface: vol(option_time, strike) -> Black vol for a CPI cap/floor.
    """

    def vol(self, t, strike=None):
        raise NotImplementedError


class ConstantCPIVolatility(CPIVolatilityStructure):
    """Constant CPI volatility.

    Parameters
    ----------
    vol_level : float
        Constant Black volatility for CPI options.
    """

    def __init__(self, vol_level: float):
        self._vol = jnp.float64(vol_level)

    def vol(self, t, strike=None):
        return self._vol


class YoYInflationOptionletVolatilityStructure:
    """Year-over-year inflation optionlet volatility surface.

    Stores vols at (expiry, strike) and interpolates.

    Parameters
    ----------
    expiries : array of expiry times
    strikes : array of strikes (YoY rates)
    vols : 2D array [n_expiry x n_strikes]
    """

    def __init__(self, expiries, strikes, vols):
        self._expiries = jnp.asarray(expiries, dtype=jnp.float64)
        self._strikes = jnp.asarray(strikes, dtype=jnp.float64)
        self._vols = jnp.asarray(vols, dtype=jnp.float64)

    def vol(self, t, strike=None):
        """YoY optionlet vol at (t, strike)."""
        t = jnp.float64(t)
        if strike is None:
            mid = len(self._strikes) // 2
            return jnp.interp(t, self._expiries, self._vols[:, mid])
        from ql_jax.termstructures.volatility.equityfx_extended import _bilinear_interp
        return _bilinear_interp(t, jnp.float64(strike), self._expiries, self._strikes, self._vols)
