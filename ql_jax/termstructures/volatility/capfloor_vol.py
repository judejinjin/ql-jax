"""Cap/Floor volatility term structures."""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval


class CapFloorTermVolatilityStructure:
    """Base class for cap/floor volatility term structures."""

    def __init__(self, reference_date: Date, day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._day_counter = day_counter

    def time_from_reference(self, d: Date):
        return year_fraction(self._reference_date, d, self._day_counter)

    def volatility(self, t, strike=None):
        raise NotImplementedError


class ConstantCapFloorTermVol(CapFloorTermVolatilityStructure):
    """Constant cap/floor volatility."""

    def __init__(self, reference_date: Date, vol, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._vol = jnp.asarray(float(vol), dtype=jnp.float64)

    def volatility(self, t, strike=None):
        return self._vol


class CapFloorTermVolCurve(CapFloorTermVolatilityStructure):
    """Cap/floor ATM vol curve interpolated over time."""

    def __init__(self, reference_date: Date, dates, vols, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        times = jnp.array([year_fraction(reference_date, d, day_counter) for d in dates], dtype=jnp.float64)
        vols = jnp.array(vols, dtype=jnp.float64)
        self._times = times
        self._vols = vols
        self._interp_state = linear_build(times, vols)

    def volatility(self, t, strike=None):
        t = jnp.asarray(t, dtype=jnp.float64)
        return linear_eval(self._interp_state, t)


class CapFloorTermVolSurface(CapFloorTermVolatilityStructure):
    """Cap/floor vol surface: time × strike grid."""

    def __init__(self, reference_date: Date, dates, strikes, vol_matrix,
                 day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._times = jnp.array(
            [year_fraction(reference_date, d, day_counter) for d in dates], dtype=jnp.float64
        )
        self._strikes = jnp.array(strikes, dtype=jnp.float64)
        self._vols = jnp.array(vol_matrix, dtype=jnp.float64)  # (n_strikes, n_times)

    def volatility(self, t, strike=None):
        t = jnp.asarray(t, dtype=jnp.float64)
        if strike is None:
            idx = len(self._strikes) // 2
            vols_at_strike = self._vols[idx, :]
        else:
            strike = jnp.asarray(strike, dtype=jnp.float64)
            i = jnp.searchsorted(self._strikes, strike) - 1
            i = jnp.clip(i, 0, len(self._strikes) - 2)
            w = (strike - self._strikes[i]) / (self._strikes[i + 1] - self._strikes[i] + 1e-20)
            w = jnp.clip(w, 0.0, 1.0)
            vols_at_strike = (1 - w) * self._vols[i, :] + w * self._vols[i + 1, :]

        interp_state = linear_build(self._times, vols_at_strike)
        return linear_eval(interp_state, t)


class OptionletVolatilityStructure:
    """Base class for optionlet (caplet/floorlet) volatility."""

    def __init__(self, reference_date: Date, day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._day_counter = day_counter

    def volatility(self, t, strike=None):
        raise NotImplementedError


class ConstantOptionletVol(OptionletVolatilityStructure):
    """Constant optionlet volatility."""

    def __init__(self, reference_date: Date, vol, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._vol = jnp.asarray(float(vol), dtype=jnp.float64)

    def volatility(self, t, strike=None):
        return self._vol
