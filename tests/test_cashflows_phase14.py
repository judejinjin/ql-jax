"""Tests for Phase 14 cashflow extensions."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestCashflowVectorsFixed:
    def test_fixed_leg(self):
        from ql_jax.cashflows.vectors import cashflow_vectors_fixed_leg
        notionals = jnp.array([1e6, 1e6, 1e6, 1e6])
        accrual_times = jnp.array([0.25, 0.5, 0.75, 1.0])
        rates = jnp.array([0.05, 0.05, 0.05, 0.05])
        result = cashflow_vectors_fixed_leg(notionals, accrual_times, rates)
        assert 'amounts' in result
        assert 'payment_times' in result
        # amounts = 1e6 * 0.05 * 0.25 = 12500 per period
        assert float(result['amounts'][0]) == pytest.approx(12500.0, rel=0.01)

    def test_with_day_count(self):
        from ql_jax.cashflows.vectors import cashflow_vectors_fixed_leg
        notionals = jnp.array([1e6, 1e6])
        accrual_times = jnp.array([0.5, 1.0])
        rates = jnp.array([0.04, 0.04])
        dcf = jnp.array([0.5, 0.5])
        result = cashflow_vectors_fixed_leg(notionals, accrual_times, rates, day_count_fractions=dcf)
        # 1e6 * 0.04 * 0.5 = 20000
        assert float(result['amounts'][0]) == pytest.approx(20000.0, rel=0.01)


class TestCashflowVectorsFloating:
    def test_floating_leg(self):
        from ql_jax.cashflows.vectors import cashflow_vectors_floating_leg
        notionals = jnp.array([1e6, 1e6, 1e6])
        accrual_times = jnp.array([0.25, 0.5, 0.75])
        forward_rates = jnp.array([0.05, 0.052, 0.054])
        result = cashflow_vectors_floating_leg(notionals, accrual_times, forward_rates)
        assert result['amounts'].shape == (3,)

    def test_with_spread(self):
        from ql_jax.cashflows.vectors import cashflow_vectors_floating_leg
        notionals = jnp.array([1e6])
        accrual_times = jnp.array([0.5])
        forward_rates = jnp.array([0.05])
        spreads = jnp.array([0.01])
        result = cashflow_vectors_floating_leg(notionals, accrual_times, forward_rates,
                                                spreads=spreads)
        # amount = 1e6 * (1.0*0.05 + 0.01) * 0.5 = 30000
        assert float(result['amounts'][0]) == pytest.approx(30000.0, rel=0.01)

    def test_with_gearing(self):
        from ql_jax.cashflows.vectors import cashflow_vectors_floating_leg
        notionals = jnp.array([1e6])
        accrual_times = jnp.array([1.0])
        forward_rates = jnp.array([0.05])
        gearings = jnp.array([2.0])
        result = cashflow_vectors_floating_leg(notionals, accrual_times, forward_rates,
                                                gearings=gearings)
        # amount = 1e6 * (2.0 * 0.05) * 1.0 = 100000
        assert float(result['amounts'][0]) == pytest.approx(100000.0, rel=0.01)


class TestRateAveraging:
    def test_compound(self):
        from ql_jax.cashflows.vectors import RateAveraging
        rates = jnp.array([0.05, 0.05, 0.05, 0.05])
        accrual = jnp.array([0.25, 0.25, 0.25, 0.25])
        compound = RateAveraging.compound(rates, accrual)
        # prod(1 + 0.05*0.25) - 1 = (1.0125)^4 - 1 ~ 0.05095
        expected = (1.0125) ** 4 - 1
        assert float(compound) == pytest.approx(expected, rel=0.001)

    def test_simple(self):
        from ql_jax.cashflows.vectors import RateAveraging
        rates = jnp.array([0.04, 0.05, 0.06])
        accrual = jnp.array([0.25, 0.25, 0.5])
        simple = RateAveraging.simple(rates, accrual)
        # Weighted avg: (0.04*0.25 + 0.05*0.25 + 0.06*0.5) / (0.25+0.25+0.5)
        expected = (0.04 * 0.25 + 0.05 * 0.25 + 0.06 * 0.5) / 1.0
        assert float(simple) == pytest.approx(expected, rel=0.001)
