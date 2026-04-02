"""Optionlet stripping methods (Stripper1, Stripper2) and stripped optionlet adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.termstructures.volatility.capfloor_vol import CapFloorTermVolSurface


class StrippedOptionlet:
    """Container for stripped optionlet volatilities.

    Parameters
    ----------
    expiries : array of optionlet expiry times (year fracs)
    strikes : array of strikes (same for all expiries, or list of arrays)
    optionlet_vols : 2D array [n_expiries x n_strikes]
    """

    def __init__(self, expiries, strikes, optionlet_vols):
        self.expiries = jnp.asarray(expiries, dtype=jnp.float64)
        self.strikes = jnp.asarray(strikes, dtype=jnp.float64)
        self.optionlet_vols = jnp.asarray(optionlet_vols, dtype=jnp.float64)


class OptionletStripper1:
    """Strips optionlet volatilities from a cap/floor term vol surface.

    Uses flat-vol bootstrapping: given cap term vols at different strikes,
    iteratively solve for optionlet vols from shortest maturity outward.

    Parameters
    ----------
    cap_vol_surface : CapFloorTermVolSurface
        Cap/floor ATM or strike-parameterized term vol surface.
    discount_curve : callable
        t -> discount factor.
    """

    def __init__(self, cap_vol_surface, discount_curve):
        self._surface = cap_vol_surface
        self._discount = discount_curve

    def strip(self, expiries, strikes, dt=0.25):
        """Bootstrap optionlet vols.

        Parameters
        ----------
        expiries : array of cap/floor expiry times
        strikes : array of strikes
        dt : caplet period length (default quarterly)

        Returns
        -------
        StrippedOptionlet
        """
        expiries = jnp.asarray(expiries, dtype=jnp.float64)
        strikes = jnp.asarray(strikes, dtype=jnp.float64)
        n_exp = len(expiries)
        n_k = len(strikes)
        result = jnp.zeros((n_exp, n_k), dtype=jnp.float64)

        for j in range(n_k):
            K = strikes[j]
            prev_var_sum = 0.0
            for i in range(n_exp):
                T = expiries[i]
                # Cap vol for this maturity and strike
                cap_vol = self._surface.vol(T, K)
                # Total variance from cap vol
                total_var = cap_vol ** 2 * T
                # Number of caplets
                n_caplets = jnp.maximum(jnp.round(T / dt), 1.0)
                # Optionlet variance = (total_var * n_caplets - prev_var_sum) / 1
                optionlet_var = jnp.maximum(total_var - prev_var_sum * (n_caplets - 1) / n_caplets, 1e-10)
                caplet_vol = jnp.sqrt(optionlet_var / jnp.maximum(dt, 1e-10))
                result = result.at[i, j].set(caplet_vol)
                prev_var_sum = total_var / n_caplets

        return StrippedOptionlet(expiries, strikes, result)


class OptionletStripper2:
    """Strips ATM optionlet vols and adjusts for smile.

    Uses OptionletStripper1 result plus ATM vol levels to rescale.

    Parameters
    ----------
    stripper1 : OptionletStripper1
        Already-stripped optionlet data.
    atm_cap_vols : 1D array
        ATM cap vols at each maturity.
    """

    def __init__(self, stripper1_result: StrippedOptionlet, atm_cap_vols):
        self._base = stripper1_result
        self._atm_vols = jnp.asarray(atm_cap_vols, dtype=jnp.float64)

    def adjust(self):
        """Return adjusted StrippedOptionlet."""
        # ATM column index (mid-strike)
        mid = len(self._base.strikes) // 2
        atm_stripped = self._base.optionlet_vols[:, mid]
        # Ratio adjustment
        ratio = self._atm_vols / jnp.maximum(atm_stripped, 1e-10)
        adjusted = self._base.optionlet_vols * ratio[:, None]
        return StrippedOptionlet(self._base.expiries, self._base.strikes, adjusted)


class StrippedOptionletAdapter:
    """Adapts StrippedOptionlet to a BlackVolTermStructure-like interface.

    Parameters
    ----------
    stripped : StrippedOptionlet
    """

    def __init__(self, stripped: StrippedOptionlet):
        self._stripped = stripped

    def vol(self, t, strike=None):
        """Interpolated optionlet vol at (t, strike)."""
        t = jnp.float64(t)
        if strike is None:
            mid = len(self._stripped.strikes) // 2
            return jnp.interp(t, self._stripped.expiries, self._stripped.optionlet_vols[:, mid])

        strike = jnp.float64(strike)
        from ql_jax.termstructures.volatility.equityfx_extended import _bilinear_interp
        return _bilinear_interp(
            t, strike,
            self._stripped.expiries, self._stripped.strikes,
            self._stripped.optionlet_vols,
        )
