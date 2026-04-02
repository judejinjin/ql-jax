"""Extended volatility surfaces: delta-based, local vol curve, piecewise."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.termstructures.volatility.black_vol import BlackVolTermStructure


class BlackVolSurfaceDelta(BlackVolTermStructure):
    """Black volatility surface parameterized in delta space.

    Stores volatilities at (expiry, delta) nodes and interpolates.

    Parameters
    ----------
    expiries : array of expiry times (year fracs)
    deltas : array of delta values (e.g. [0.1, 0.25, 0.5, 0.75, 0.9])
    vols : 2D array of vols [n_expiry x n_delta]
    spot : spot price (for delta-to-strike conversion)
    r : risk-free rate
    q : dividend yield
    """

    def __init__(self, expiries, deltas, vols, spot=100.0, r=0.05, q=0.0):
        self._expiries = jnp.asarray(expiries, dtype=jnp.float64)
        self._deltas = jnp.asarray(deltas, dtype=jnp.float64)
        self._vols = jnp.asarray(vols, dtype=jnp.float64)
        self._spot = spot
        self._r = r
        self._q = q

    def black_vol(self, t, strike=None):
        """Return vol at (t, strike). If strike is None, return ATM."""
        from jax.scipy.stats import norm as jnorm

        t = jnp.float64(t)
        if strike is None:
            # ATM = 50-delta
            delta = 0.5
        else:
            # Approximate delta from BS
            # Start with ATM vol guess
            fwd = self._spot * jnp.exp((self._r - self._q) * t)
            sigma_guess = jnp.interp(t, self._expiries, self._vols[:, len(self._deltas) // 2])
            d1 = (jnp.log(fwd / strike) + 0.5 * sigma_guess ** 2 * t) / (sigma_guess * jnp.sqrt(t) + 1e-10)
            delta = jnorm.cdf(d1)

        # Bilinear interpolation in (t, delta) space
        vol = _bilinear_interp(t, delta, self._expiries, self._deltas, self._vols)
        return jnp.maximum(vol, 1e-6)


class LocalVolCurve:
    """1D local volatility as function of time only.

    Parameters
    ----------
    times : array of times
    local_vols : array of local vol values
    """

    def __init__(self, times, local_vols):
        self._times = jnp.asarray(times, dtype=jnp.float64)
        self._vols = jnp.asarray(local_vols, dtype=jnp.float64)

    def local_vol(self, t, S=None):
        """Local vol at time t (independent of S)."""
        return jnp.interp(jnp.float64(t), self._times, self._vols)


class FixedLocalVolSurface:
    """Local volatility from a fixed grid of (time, spot, vol) data.

    Parameters
    ----------
    times : array of times
    spots : array of spot levels
    local_vols : 2D array [n_times x n_spots]
    """

    def __init__(self, times, spots, local_vols):
        self._times = jnp.asarray(times, dtype=jnp.float64)
        self._spots = jnp.asarray(spots, dtype=jnp.float64)
        self._vols = jnp.asarray(local_vols, dtype=jnp.float64)

    def local_vol(self, t, S):
        """Interpolated local vol at (t, S)."""
        return _bilinear_interp(
            jnp.float64(t), jnp.float64(S),
            self._times, self._spots, self._vols,
        )


class PiecewiseBlackVarianceSurface(BlackVolTermStructure):
    """Piecewise-constant total variance surface.

    Variance is constant between breakpoints in time.

    Parameters
    ----------
    times : breakpoint times
    strikes : array of strikes
    total_variances : 2D array [n_times x n_strikes]
    """

    def __init__(self, times, strikes, total_variances):
        self._times = jnp.asarray(times, dtype=jnp.float64)
        self._strikes = jnp.asarray(strikes, dtype=jnp.float64)
        self._tvars = jnp.asarray(total_variances, dtype=jnp.float64)

    def black_vol(self, t, strike=None):
        t = jnp.float64(t)
        if strike is None:
            # ATM: use middle strike
            idx = len(self._strikes) // 2
            tv = jnp.interp(t, self._times, self._tvars[:, idx])
        else:
            tv = _bilinear_interp(t, jnp.float64(strike),
                                  self._times, self._strikes, self._tvars)
        return jnp.sqrt(jnp.maximum(tv / jnp.maximum(t, 1e-10), 1e-12))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _bilinear_interp(t, s, ts, ss, values):
    """Simple bilinear interpolation on a grid.

    Parameters
    ----------
    t, s : query point
    ts : 1D array of first dimension nodes
    ss : 1D array of second dimension nodes
    values : 2D array [len(ts) x len(ss)]
    """
    # Clamp to bounds
    t = jnp.clip(t, ts[0], ts[-1])
    s = jnp.clip(s, ss[0], ss[-1])

    # Find indices
    i = jnp.searchsorted(ts, t, side='right') - 1
    i = jnp.clip(i, 0, len(ts) - 2)
    j = jnp.searchsorted(ss, s, side='right') - 1
    j = jnp.clip(j, 0, len(ss) - 2)

    # Weights
    wt = (t - ts[i]) / jnp.maximum(ts[i + 1] - ts[i], 1e-15)
    ws = (s - ss[j]) / jnp.maximum(ss[j + 1] - ss[j], 1e-15)

    v00 = values[i, j]
    v01 = values[i, j + 1]
    v10 = values[i + 1, j]
    v11 = values[i + 1, j + 1]

    return (1 - wt) * (1 - ws) * v00 + (1 - wt) * ws * v01 + wt * (1 - ws) * v10 + wt * ws * v11
