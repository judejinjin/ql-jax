"""Tests for Phase 13 quote types."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestDerivedQuote:
    def test_transform(self):
        from ql_jax.quotes.composite import DerivedQuote

        class SimpleQuote:
            def __init__(self, v):
                self._v = v
            @property
            def value(self):
                return self._v

        base = SimpleQuote(100.0)
        dq = DerivedQuote(base_quote=base, transform=lambda x: x * 2)
        assert float(dq.value) == pytest.approx(200.0)


class TestCompositeQuote:
    def test_compose(self):
        from ql_jax.quotes.composite import CompositeQuote

        class SimpleQuote:
            def __init__(self, v):
                self._v = v
            @property
            def value(self):
                return self._v

        q1 = SimpleQuote(3.0)
        q2 = SimpleQuote(4.0)
        cq = CompositeQuote(quote1=q1, quote2=q2, compose=lambda a, b: a + b)
        assert float(cq.value) == pytest.approx(7.0)


class TestDeltaVolQuote:
    def test_construction(self):
        from ql_jax.quotes.composite import DeltaVolQuote
        dvq = DeltaVolQuote(delta=0.25, vol=0.12, maturity=1.0)
        assert float(dvq.value) == pytest.approx(0.12)
        assert dvq.delta == pytest.approx(0.25)
        assert dvq.maturity == pytest.approx(1.0)


class TestEurodollarFuturesQuote:
    def test_rate_to_price(self):
        from ql_jax.quotes.composite import EurodollarFuturesQuote

        class SimpleQuote:
            def __init__(self, v):
                self._v = v
            @property
            def value(self):
                return self._v

        rate_quote = SimpleQuote(0.05)  # 5% rate
        eq = EurodollarFuturesQuote(rate_quote=rate_quote)
        assert float(eq.value) == pytest.approx(95.0)


class TestFuturesConvAdjustmentQuote:
    def test_adjustment(self):
        from ql_jax.quotes.composite import FuturesConvAdjustmentQuote

        class SimpleQuote:
            def __init__(self, v):
                self._v = v
            @property
            def value(self):
                return self._v

        fq = SimpleQuote(95.5)
        adj = FuturesConvAdjustmentQuote(futures_quote=fq, adjustment=0.002)
        assert float(adj.value) == pytest.approx(95.498)


class TestImpliedStdDevQuote:
    def test_construction(self):
        from ql_jax.quotes.composite import ImpliedStdDevQuote

        class SimpleQuote:
            def __init__(self, v):
                self._v = v
            @property
            def value(self):
                return self._v

        price_q = SimpleQuote(10.0)
        isdq = ImpliedStdDevQuote(option_type='call', forward=100.0, strike=100.0,
                                   price_quote=price_q, discount=1.0)
        vol = isdq.value
        assert float(vol) > 0
