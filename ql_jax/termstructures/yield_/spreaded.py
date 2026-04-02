"""Implied and spread term structures.

ImpliedTermStructure: shifts reference date of an existing curve.
ForwardSpreadedTermStructure: adds spread to forward rates.
ZeroSpreadedTermStructure: adds spread to zero rates.
"""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.base import YieldTermStructure


class ImpliedTermStructure(YieldTermStructure):
    """Term structure implied from another with a different reference date.

    discount(t) = original.discount(t_offset + t) / original.discount(t_offset)
    where t_offset is the year fraction from original ref to this ref.
    """

    def __init__(self, original: YieldTermStructure, reference_date: Date):
        super().__init__(reference_date, original.day_counter)
        self._original = original
        self._t_offset = float(original.time_from_reference(reference_date))

    def discount_impl(self, t):
        df_ref = self._original.discount(self._t_offset)
        df_target = self._original.discount(self._t_offset + t)
        return df_target / df_ref


class ForwardSpreadedTermStructure(YieldTermStructure):
    """Term structure with a spread added to forward rates."""

    def __init__(self, original: YieldTermStructure, spread):
        super().__init__(original.reference_date, original.day_counter)
        self._original = original
        self._spread = jnp.asarray(float(spread), dtype=jnp.float64)

    def forward_rate_impl(self, t):
        return self._original.forward_rate_impl(t) + self._spread

    def discount_impl(self, t):
        return self._original.discount(t) * jnp.exp(-self._spread * t)

    def zero_rate_impl(self, t):
        return self._original.zero_rate(t) + self._spread


class ZeroSpreadedTermStructure(YieldTermStructure):
    """Term structure with a spread added to zero rates.

    z_new(t) = z_original(t) + spread
    df_new(t) = exp(-(z_original(t) + spread) * t) = df_original(t) * exp(-spread * t)
    """

    def __init__(self, original: YieldTermStructure, spread):
        super().__init__(original.reference_date, original.day_counter)
        self._original = original
        self._spread = jnp.asarray(float(spread), dtype=jnp.float64)

    def zero_rate_impl(self, t):
        return self._original.zero_rate(t) + self._spread

    def discount_impl(self, t):
        return self._original.discount(t) * jnp.exp(-self._spread * t)
