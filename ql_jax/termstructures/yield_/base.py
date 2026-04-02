"""Yield term structure base class.

Provides the core interface: discount(t), zero_rate(t), forward_rate(t1, t2).
All implementations must provide at least one of these; the rest are derived.
All operations are differentiable via JAX.
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction


class YieldTermStructure:
    """Base class for yield term structures.

    All times t are year fractions from the reference date.
    Subclasses must implement at least one of:
      - discount_impl(t) -> float
      - zero_rate_impl(t) -> float
      - forward_rate_impl(t) -> float (instantaneous forward)
    """

    def __init__(self, reference_date: Date, day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._day_counter = day_counter

    @property
    def reference_date(self) -> Date:
        return self._reference_date

    @property
    def day_counter(self) -> str:
        return self._day_counter

    def time_from_reference(self, d: Date) -> jnp.ndarray:
        """Year fraction from reference date to d."""
        return jnp.asarray(
            year_fraction(self._reference_date, d, self._day_counter),
            dtype=jnp.float64,
        )

    # ----- Core interface (override at least one) -----

    def discount_impl(self, t):
        """Discount factor at time t. Override in subclass."""
        r = self.zero_rate_impl(t)
        return jnp.exp(-r * t)

    def zero_rate_impl(self, t):
        """Continuously compounded zero rate at time t. Override in subclass."""
        df = self.discount_impl(t)
        return jnp.where(t > 0, -jnp.log(df) / t, 0.0)

    def forward_rate_impl(self, t):
        """Instantaneous forward rate at time t. Override in subclass."""
        # Numerical derivative of -log(discount(t))
        dt = 1e-4
        df1 = self.discount_impl(t)
        df2 = self.discount_impl(t + dt)
        return -jnp.log(df2 / df1) / dt

    # ----- Public interface -----

    def discount(self, t):
        """Discount factor P(0, t)."""
        t = jnp.asarray(t, dtype=jnp.float64)
        return self.discount_impl(t)

    def zero_rate(self, t):
        """Continuously compounded zero rate r(t) such that P(0,t) = exp(-r*t)."""
        t = jnp.asarray(t, dtype=jnp.float64)
        return self.zero_rate_impl(t)

    def forward_rate(self, t1, t2=None):
        """Forward rate.

        If t2 is None, returns instantaneous forward rate at t1.
        If t2 is given, returns the simple forward rate over [t1, t2]:
            F(t1,t2) = (P(0,t1)/P(0,t2) - 1) / (t2 - t1)
        """
        t1 = jnp.asarray(t1, dtype=jnp.float64)
        if t2 is None:
            return self.forward_rate_impl(t1)
        t2 = jnp.asarray(t2, dtype=jnp.float64)
        df1 = self.discount(t1)
        df2 = self.discount(t2)
        tau = t2 - t1
        return jnp.where(tau > 0, (df1 / df2 - 1.0) / tau, self.forward_rate_impl(t1))

    def discount_date(self, d: Date):
        """Discount factor to date d."""
        return self.discount(self.time_from_reference(d))

    def zero_rate_date(self, d: Date):
        """Zero rate to date d."""
        return self.zero_rate(self.time_from_reference(d))
