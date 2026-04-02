"""Swaption volatility structures.

Base class and implementations for swaption volatility surfaces.
"""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval


class SwaptionVolStructure:
    """Base class for swaption volatility term structures."""

    def __init__(self, reference_date: Date, day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._day_counter = day_counter

    @property
    def reference_date(self):
        return self._reference_date

    def time_from_reference(self, d: Date):
        return year_fraction(self._reference_date, d, self._day_counter)

    def volatility(self, option_time, swap_length, strike=None):
        """Swaption volatility. Override in subclass."""
        raise NotImplementedError

    def black_variance(self, option_time, swap_length, strike=None):
        vol = self.volatility(option_time, swap_length, strike)
        return vol * vol * option_time


class SwaptionConstantVol(SwaptionVolStructure):
    """Constant swaption volatility."""

    def __init__(self, reference_date: Date, vol, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._vol = jnp.asarray(float(vol), dtype=jnp.float64)

    def volatility(self, option_time, swap_length, strike=None):
        return self._vol


class SwaptionVolMatrix(SwaptionVolStructure):
    """Swaption volatility matrix.

    A 2D grid of ATM swaption vols indexed by option tenor and swap length.
    Bilinear interpolation.
    """

    def __init__(self, reference_date: Date,
                 option_tenors, swap_lengths, vol_matrix,
                 day_counter: str = 'Actual365Fixed'):
        """
        option_tenors : array of year fractions for option expiry
        swap_lengths : array of year fractions for swap tenors
        vol_matrix : 2D array (n_option x n_swap)
        """
        super().__init__(reference_date, day_counter)
        self._option_tenors = jnp.array(option_tenors, dtype=jnp.float64)
        self._swap_lengths = jnp.array(swap_lengths, dtype=jnp.float64)
        self._vols = jnp.array(vol_matrix, dtype=jnp.float64)

    def volatility(self, option_time, swap_length, strike=None):
        option_time = jnp.asarray(option_time, dtype=jnp.float64)
        swap_length = jnp.asarray(swap_length, dtype=jnp.float64)

        # Interpolate along option tenor axis
        ot = self._option_tenors
        sl = self._swap_lengths

        # Find bracketing option tenor
        i = jnp.searchsorted(ot, option_time) - 1
        i = jnp.clip(i, 0, len(ot) - 2)
        w_opt = (option_time - ot[i]) / (ot[i + 1] - ot[i] + 1e-20)
        w_opt = jnp.clip(w_opt, 0.0, 1.0)

        # Interpolate across swap lengths for both bracketing option tenors
        row_lo = self._vols[i, :]
        row_hi = self._vols[i + 1, :]

        # Find bracketing swap length
        j = jnp.searchsorted(sl, swap_length) - 1
        j = jnp.clip(j, 0, len(sl) - 2)
        w_swap = (swap_length - sl[j]) / (sl[j + 1] - sl[j] + 1e-20)
        w_swap = jnp.clip(w_swap, 0.0, 1.0)

        # Bilinear interpolation
        v00 = row_lo[j]
        v01 = row_lo[j + 1]
        v10 = row_hi[j]
        v11 = row_hi[j + 1]

        return (v00 * (1 - w_opt) * (1 - w_swap) +
                v01 * (1 - w_opt) * w_swap +
                v10 * w_opt * (1 - w_swap) +
                v11 * w_opt * w_swap)


class SwaptionVolCube(SwaptionVolStructure):
    """Swaption volatility cube — ATM matrix + smile in strike dimension."""

    def __init__(self, reference_date: Date,
                 atm_matrix: SwaptionVolMatrix,
                 option_tenors, swap_lengths, strikes, vol_spreads,
                 day_counter: str = 'Actual365Fixed'):
        """
        vol_spreads : 3D array (n_option x n_swap x n_strikes) of volatility spreads above ATM
        """
        super().__init__(reference_date, day_counter)
        self._atm = atm_matrix
        self._option_tenors = jnp.array(option_tenors, dtype=jnp.float64)
        self._swap_lengths = jnp.array(swap_lengths, dtype=jnp.float64)
        self._strikes = jnp.array(strikes, dtype=jnp.float64)
        self._spreads = jnp.array(vol_spreads, dtype=jnp.float64)

    def volatility(self, option_time, swap_length, strike=None):
        atm_vol = self._atm.volatility(option_time, swap_length)
        if strike is None:
            return atm_vol
        # TODO: interpolate spread from cube
        return atm_vol
