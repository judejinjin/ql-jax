"""Simple quote — a mutable scalar value.

In QuantLib, SimpleQuote is Observable + holds a single Real.
In QL-JAX, quotes are just jnp scalar arrays for differentiability.
We provide a thin wrapper for API compatibility.
"""

import jax.numpy as jnp


class SimpleQuote:
    """A simple market quote holding a single value."""

    __slots__ = ('_value',)

    def __init__(self, value=None):
        if value is None:
            self._value = jnp.nan
        else:
            self._value = jnp.asarray(float(value), dtype=jnp.float64)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = jnp.asarray(float(v), dtype=jnp.float64)

    def is_valid(self):
        return not jnp.isnan(self._value)

    def __repr__(self):
        return f"SimpleQuote({float(self._value):.6f})"


def composite_quote(q1, q2, f):
    """Composite quote: f(q1, q2).

    In QL-JAX, quotes are arrays, so this is just function application.
    """
    return f(q1, q2)


def derived_quote(q, f):
    """Derived quote: f(q).

    In QL-JAX, this is just function application on an array value.
    """
    return f(q)


def delta_vol_quote(option_type, delta, spot, t, vol):
    """Delta-vol quote for FX options."""
    return jnp.asarray(vol, dtype=jnp.float64)


def forward_value_quote(index_value, discount):
    """Forward value: index_value / discount."""
    return index_value / discount


def implied_stddev_quote(option_type, strike, forward, price, displacement=0.0):
    """Implied standard deviation from an option price.

    Returns sigma * sqrt(T) from Black formula inversion.
    Actual computation requires Black formula — placeholder identity.
    """
    return price


def futures_conv_adjustment_quote(futures_price, convexity_adj):
    """Futures convexity-adjusted quote."""
    return futures_price - convexity_adj
