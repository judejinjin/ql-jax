"""Fitted bond discount curve.

Fits a parametric discount function to a set of bond prices using
non-linear optimization (Levenberg-Marquardt).
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.base import YieldTermStructure


class FittedBondDiscountCurve(YieldTermStructure):
    """Discount curve fitted to bond prices.

    Uses a parametric form for the discount function and calibrates
    parameters to match observed bond prices.
    """

    def __init__(
        self,
        reference_date: Date,
        fitting_method,  # A FittingMethod instance
        day_counter: str = 'Actual365Fixed',
    ):
        super().__init__(reference_date, day_counter)
        self._fitting_method = fitting_method

    def discount_impl(self, t):
        return self._fitting_method.discount(t)


# ---------------------------------------------------------------------------
# Fitting methods
# ---------------------------------------------------------------------------

class NelsonSiegel:
    """Nelson-Siegel parametric form for zero rates.

    z(t) = b0 + (b1 + b2) * [1 - exp(-t/tau)] / (t/tau) - b2 * exp(-t/tau)
    """

    def __init__(self, params=None):
        if params is None:
            self.params = jnp.array([0.05, -0.02, 0.01, 1.0], dtype=jnp.float64)
        else:
            self.params = jnp.array(params, dtype=jnp.float64)

    def zero_rate(self, t):
        b0, b1, b2, tau = self.params
        t = jnp.maximum(t, 1e-10)
        x = t / tau
        factor = (1.0 - jnp.exp(-x)) / x
        return b0 + b1 * factor + b2 * (factor - jnp.exp(-x))

    def discount(self, t):
        r = self.zero_rate(t)
        return jnp.exp(-r * t)


class Svensson:
    """Svensson extension of Nelson-Siegel.

    z(t) = b0 + b1 * f1(t/tau1) + b2 * [f1(t/tau1) - exp(-t/tau1)]
           + b3 * [f1(t/tau2) - exp(-t/tau2)]
    where f1(x) = [1 - exp(-x)] / x
    """

    def __init__(self, params=None):
        if params is None:
            self.params = jnp.array([0.05, -0.02, 0.01, 0.005, 1.0, 3.0], dtype=jnp.float64)
        else:
            self.params = jnp.array(params, dtype=jnp.float64)

    def zero_rate(self, t):
        b0, b1, b2, b3, tau1, tau2 = self.params
        t = jnp.maximum(t, 1e-10)
        x1 = t / tau1
        x2 = t / tau2
        f1 = (1.0 - jnp.exp(-x1)) / x1
        f2 = (1.0 - jnp.exp(-x2)) / x2
        return b0 + b1 * f1 + b2 * (f1 - jnp.exp(-x1)) + b3 * (f2 - jnp.exp(-x2))

    def discount(self, t):
        r = self.zero_rate(t)
        return jnp.exp(-r * t)


class ExponentialSplines:
    """Exponential spline discount function."""

    def __init__(self, params=None, kappa=0.05):
        self.kappa = kappa
        if params is None:
            self.params = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        else:
            self.params = jnp.array(params, dtype=jnp.float64)

    def discount(self, t):
        a = self.params
        x = self.kappa * t
        return (a[0] + a[1] * t + a[2] * t**2 +
                a[3] * jnp.exp(-x) + a[4] * jnp.exp(-2 * x))

    def zero_rate(self, t):
        df = self.discount(t)
        t = jnp.maximum(t, 1e-10)
        return -jnp.log(jnp.maximum(df, 1e-20)) / t
