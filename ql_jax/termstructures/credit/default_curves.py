"""Credit default term structures.

Provides hazard rate, default density, and survival probability curves.
All differentiable via JAX.
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval


class DefaultProbabilityTermStructure:
    """Base class for credit term structures.

    Subclasses implement one of:
    - hazard_rate(t)
    - default_density(t)
    - survival_probability(t)
    """

    def __init__(self, reference_date: Date, day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._day_counter = day_counter

    @property
    def reference_date(self):
        return self._reference_date

    def time_from_reference(self, d: Date):
        return year_fraction(self._reference_date, d, self._day_counter)

    def survival_probability(self, t):
        """P(tau > t) — probability of surviving to time t."""
        raise NotImplementedError

    def default_probability(self, t1, t2=None):
        """Probability of default in (t1, t2]. If t2 is None, returns P(tau <= t1)."""
        if t2 is None:
            return 1.0 - self.survival_probability(t1)
        return self.survival_probability(t1) - self.survival_probability(t2)

    def hazard_rate(self, t):
        """Instantaneous hazard rate h(t) = -d/dt ln S(t)."""
        dt = 1e-4
        s1 = self.survival_probability(t)
        s2 = self.survival_probability(t + dt)
        return -jnp.log(s2 / s1) / dt

    def default_density(self, t):
        """Default density f(t) = -dS/dt = h(t) * S(t)."""
        return self.hazard_rate(t) * self.survival_probability(t)


class FlatHazardRate(DefaultProbabilityTermStructure):
    """Constant hazard rate: S(t) = exp(-h*t)."""

    def __init__(self, reference_date: Date, hazard_rate, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._hazard_rate = jnp.asarray(float(hazard_rate), dtype=jnp.float64)

    def survival_probability(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return jnp.exp(-self._hazard_rate * t)

    def hazard_rate(self, t):
        return self._hazard_rate

    def default_density(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return self._hazard_rate * jnp.exp(-self._hazard_rate * t)


class InterpolatedHazardRateCurve(DefaultProbabilityTermStructure):
    """Piecewise linear hazard rate curve.

    S(t) = exp(-∫₀ᵗ h(s) ds) with linearly interpolated h(s).
    """

    def __init__(self, reference_date: Date, dates, hazard_rates,
                 day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        times = [year_fraction(reference_date, d, day_counter) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._hazard_rates = jnp.array(hazard_rates, dtype=jnp.float64)
        self._interp_state = linear_build(self._times, self._hazard_rates)

    def hazard_rate(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return linear_eval(self._interp_state, t)

    def survival_probability(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        # Integrate hazard rate using trapezoidal rule
        n = 100
        ts = jnp.linspace(0.0, float(t), n + 1)
        hs = jnp.array([float(self.hazard_rate(ti)) for ti in ts])
        integral = jnp.trapezoid(hs, ts)
        return jnp.exp(-integral)


class InterpolatedSurvivalProbabilityCurve(DefaultProbabilityTermStructure):
    """Interpolated survival probability curve."""

    def __init__(self, reference_date: Date, dates, survival_probs,
                 day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        times = [year_fraction(reference_date, d, day_counter) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        # Interpolate log survival for guaranteed positive probabilities
        self._log_surv = jnp.log(jnp.array(survival_probs, dtype=jnp.float64))
        self._interp_state = linear_build(self._times, self._log_surv)

    def survival_probability(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        log_s = linear_eval(self._interp_state, t)
        return jnp.exp(log_s)


class PiecewiseDefaultCurve(DefaultProbabilityTermStructure):
    """Bootstrapped default probability curve from CDS spreads.

    Similar to PiecewiseYieldCurve but for credit curves.
    """

    def __init__(self, reference_date: Date, dates, survival_probs,
                 day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        times = [year_fraction(reference_date, d, day_counter) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._log_surv = jnp.log(jnp.array(survival_probs, dtype=jnp.float64))
        self._interp_state = linear_build(self._times, self._log_surv)

    def survival_probability(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        log_s = linear_eval(self._interp_state, t)
        return jnp.exp(log_s)
