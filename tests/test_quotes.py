"""Tests for quote types."""

import jax.numpy as jnp
from ql_jax.quotes.simple import (
    SimpleQuote, composite_quote, derived_quote,
    forward_value_quote, futures_conv_adjustment_quote,
)


class TestSimpleQuote:
    def test_construction(self):
        q = SimpleQuote(1.5)
        assert abs(float(q.value) - 1.5) < 1e-12

    def test_null_construction(self):
        q = SimpleQuote()
        assert not q.is_valid()

    def test_set_value(self):
        q = SimpleQuote(1.0)
        q.value = 2.0
        assert abs(float(q.value) - 2.0) < 1e-12

    def test_is_valid(self):
        q = SimpleQuote(3.14)
        assert q.is_valid()

    def test_repr(self):
        q = SimpleQuote(1.5)
        assert "1.5" in repr(q)


class TestCompositeQuote:
    def test_sum(self):
        q1 = jnp.array(1.0)
        q2 = jnp.array(2.0)
        result = composite_quote(q1, q2, lambda a, b: a + b)
        assert abs(float(result) - 3.0) < 1e-12

    def test_product(self):
        q1 = jnp.array(3.0)
        q2 = jnp.array(4.0)
        result = composite_quote(q1, q2, lambda a, b: a * b)
        assert abs(float(result) - 12.0) < 1e-12


class TestDerivedQuote:
    def test_square(self):
        q = jnp.array(5.0)
        result = derived_quote(q, lambda x: x * x)
        assert abs(float(result) - 25.0) < 1e-12


class TestForwardValueQuote:
    def test_basic(self):
        result = forward_value_quote(
            jnp.array(100.0), jnp.array(0.95)
        )
        assert abs(float(result) - 100.0 / 0.95) < 1e-10


class TestFuturesConvAdjustment:
    def test_basic(self):
        result = futures_conv_adjustment_quote(
            jnp.array(99.0), jnp.array(0.01)
        )
        assert abs(float(result) - 98.99) < 1e-10
