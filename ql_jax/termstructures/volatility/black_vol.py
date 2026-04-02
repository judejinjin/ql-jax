"""Black volatility term structures.

BlackVolTermStructure: base for Black volatilities.
BlackConstantVol: constant vol surface.
BlackVarianceCurve: term structure of ATM vols.
BlackVarianceSurface: full (strike, time) surface.
"""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval


class BlackVolTermStructure:
    """Base class for Black volatility term structures."""

    def __init__(self, reference_date: Date, day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._day_counter = day_counter

    @property
    def reference_date(self):
        return self._reference_date

    def time_from_reference(self, d: Date):
        return year_fraction(self._reference_date, d, self._day_counter)

    def black_vol(self, t, strike=None):
        """Black volatility at time t and strike. Override in subclass."""
        raise NotImplementedError

    def black_variance(self, t, strike=None):
        vol = self.black_vol(t, strike)
        return vol * vol * t

    def black_forward_vol(self, t1, t2, strike=None):
        """Forward Black volatility over [t1, t2]."""
        var1 = self.black_variance(t1, strike)
        var2 = self.black_variance(t2, strike)
        dt = t2 - t1
        return jnp.sqrt(jnp.maximum(var2 - var1, 0.0) / jnp.maximum(dt, 1e-10))


class BlackConstantVol(BlackVolTermStructure):
    """Constant Black volatility."""

    def __init__(self, reference_date: Date, vol, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._vol = jnp.asarray(float(vol), dtype=jnp.float64)

    def black_vol(self, t, strike=None):
        return self._vol


class BlackVarianceCurve(BlackVolTermStructure):
    """Black variance term structure — ATM vol curve.

    Interpolates total variance as a function of time.
    """

    def __init__(self, reference_date: Date, dates, vols, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        times = [year_fraction(reference_date, d, day_counter) for d in dates]
        variances = [v * v * t for v, t in zip(vols, times)]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._variances = jnp.array(variances, dtype=jnp.float64)
        self._interp_state = linear_build(self._times, self._variances)

    def black_vol(self, t, strike=None):
        t = jnp.asarray(t, dtype=jnp.float64)
        var = linear_eval(self._interp_state, t)
        return jnp.sqrt(jnp.maximum(var, 0.0) / jnp.maximum(t, 1e-10))

    def black_variance(self, t, strike=None):
        t = jnp.asarray(t, dtype=jnp.float64)
        return linear_eval(self._interp_state, t)


class BlackVarianceSurface(BlackVolTermStructure):
    """Full Black variance surface — bilinear interpolation on (strike, time).

    Stores total variances on a grid and interpolates bilinearly.
    """

    def __init__(self, reference_date: Date, dates, strikes, vol_matrix,
                 day_counter: str = 'Actual365Fixed'):
        """
        Parameters
        ----------
        dates : list of Date — expiry dates (columns)
        strikes : list of float — strike levels (rows)
        vol_matrix : 2D array shape (n_strikes, n_dates)
        """
        super().__init__(reference_date, day_counter)
        times = jnp.array([year_fraction(reference_date, d, day_counter) for d in dates], dtype=jnp.float64)
        strikes_arr = jnp.array(strikes, dtype=jnp.float64)
        vol_mat = jnp.array(vol_matrix, dtype=jnp.float64)
        # Store total variances
        self._times = times
        self._strikes = strikes_arr
        self._variances = vol_mat ** 2 * times[None, :]  # (n_strikes, n_dates)

    def black_vol(self, t, strike=None):
        t = jnp.asarray(t, dtype=jnp.float64)
        if strike is None:
            # ATM: take middle strike
            idx = len(self._strikes) // 2
            var_at_strikes = self._variances[idx, :]
        else:
            strike = jnp.asarray(strike, dtype=jnp.float64)
            # Interpolate across strikes for each time
            # Simple: find bracketing strikes and linearly interpolate
            i = jnp.searchsorted(self._strikes, strike) - 1
            i = jnp.clip(i, 0, len(self._strikes) - 2)
            w = (strike - self._strikes[i]) / (self._strikes[i + 1] - self._strikes[i] + 1e-20)
            w = jnp.clip(w, 0.0, 1.0)
            var_at_strikes = (1.0 - w) * self._variances[i, :] + w * self._variances[i + 1, :]

        # Interpolate across time
        interp_state = linear_build(self._times, var_at_strikes)
        var = linear_eval(interp_state, t)
        return jnp.sqrt(jnp.maximum(var, 0.0) / jnp.maximum(t, 1e-10))


class LocalVolTermStructure:
    """Base class for local volatility surfaces."""

    def __init__(self, reference_date: Date, day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._day_counter = day_counter

    def local_vol(self, t, strike):
        raise NotImplementedError


class LocalConstantVol(LocalVolTermStructure):
    """Constant local volatility."""

    def __init__(self, reference_date: Date, vol, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._vol = jnp.asarray(float(vol), dtype=jnp.float64)

    def local_vol(self, t, strike):
        return self._vol


class ImpliedVolTermStructure(BlackVolTermStructure):
    """Implied vol surface shifted to a new reference date."""

    def __init__(self, original: BlackVolTermStructure, reference_date: Date):
        super().__init__(reference_date, original._day_counter)
        self._original = original
        self._t_offset = year_fraction(original._reference_date, reference_date, original._day_counter)

    def black_vol(self, t, strike=None):
        return self._original.black_vol(self._t_offset + t, strike)
