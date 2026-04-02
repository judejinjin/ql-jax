"""Tests for new instruments (Phase 9)."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestLookbackOption:
    def test_fixed_call_payoff(self):
        from ql_jax.instruments.lookback import ContinuousFixedLookbackOption
        opt = ContinuousFixedLookbackOption(strike=100, is_call=True, current_minmax=110, maturity=1.0)
        assert float(opt.payoff(115.0)) == pytest.approx(15.0)
        assert float(opt.payoff(90.0)) == 0.0

    def test_fixed_put_payoff(self):
        from ql_jax.instruments.lookback import ContinuousFixedLookbackOption
        opt = ContinuousFixedLookbackOption(strike=100, is_call=False, current_minmax=90, maturity=1.0)
        assert float(opt.payoff(85.0)) == pytest.approx(15.0)

    def test_floating_call_payoff(self):
        from ql_jax.instruments.lookback import ContinuousFloatingLookbackOption
        opt = ContinuousFloatingLookbackOption(is_call=True, current_minmax=90, maturity=1.0)
        assert float(opt.payoff(105.0, 90.0)) == pytest.approx(15.0)

    def test_floating_put_payoff(self):
        from ql_jax.instruments.lookback import ContinuousFloatingLookbackOption
        opt = ContinuousFloatingLookbackOption(is_call=False, current_minmax=110, maturity=1.0)
        assert float(opt.payoff(100.0, 115.0)) == pytest.approx(15.0)


class TestBasketOption:
    def test_min_basket_call(self):
        from ql_jax.instruments.basket import BasketPayoff, BasketType
        p = BasketPayoff(basket_type=BasketType.MIN, strike=95, is_call=True)
        prices = jnp.array([100.0, 90.0, 105.0])
        assert float(p(prices)) == 0.0  # min=90 < 95

    def test_max_basket_call(self):
        from ql_jax.instruments.basket import BasketPayoff, BasketType
        p = BasketPayoff(basket_type=BasketType.MAX, strike=95, is_call=True)
        prices = jnp.array([100.0, 90.0, 105.0])
        assert float(p(prices)) == pytest.approx(10.0)  # max=105

    def test_average_basket(self):
        from ql_jax.instruments.basket import BasketPayoff, BasketType
        p = BasketPayoff(basket_type=BasketType.AVERAGE, strike=95, is_call=True)
        prices = jnp.array([100.0, 90.0, 105.0])
        expected = (100 + 90 + 105) / 3.0 - 95
        assert float(p(prices)) == pytest.approx(expected, abs=1e-10)

    def test_spread_option(self):
        from ql_jax.instruments.basket import SpreadOption
        s = SpreadOption(strike=5.0, is_call=True, maturity=1.0)
        assert float(s.payoff_value(110.0, 100.0)) == pytest.approx(5.0)
        assert float(s.payoff_value(100.0, 100.0)) == 0.0


class TestCliquetOption:
    def test_payoff_no_cap_floor(self):
        from ql_jax.instruments.cliquet import CliquetOption
        opt = CliquetOption(maturity=1.0, is_call=True)
        returns = jnp.array([0.05, -0.02, 0.03, 0.01])
        assert float(opt.payoff(returns)) == pytest.approx(0.07, abs=1e-10)

    def test_payoff_with_local_cap_floor(self):
        from ql_jax.instruments.cliquet import CliquetOption
        opt = CliquetOption(local_cap=0.03, local_floor=-0.01, maturity=1.0, is_call=True)
        returns = jnp.array([0.05, -0.02, 0.03, 0.01])
        # Capped: [0.03, -0.01, 0.03, 0.01] -> sum = 0.06
        assert float(opt.payoff(returns)) == pytest.approx(0.06, abs=1e-10)

    def test_payoff_with_global_floor(self):
        from ql_jax.instruments.cliquet import CliquetOption
        opt = CliquetOption(global_floor=0.0, maturity=1.0, is_call=True)
        returns = jnp.array([-0.05, -0.02, -0.03, -0.01])
        assert float(opt.payoff(returns)) == 0.0


class TestVarianceSwap:
    def test_payoff(self):
        from ql_jax.instruments.variance_swap import VarianceSwap
        vs = VarianceSwap(strike_variance=0.04, notional=1e6, maturity=1.0)
        payoff = vs.payoff(0.05)
        assert float(payoff) == pytest.approx(1e6 * 0.01)

    def test_npv(self):
        from ql_jax.instruments.variance_swap import VarianceSwap
        vs = VarianceSwap(strike_variance=0.04, notional=1e6, maturity=1.0)
        npv = vs.npv(0.05, 0.95)
        assert float(npv) == pytest.approx(1e6 * 0.01 * 0.95)

    def test_realized_variance(self):
        from ql_jax.instruments.variance_swap import realized_variance_from_prices
        prices = jnp.array([100.0, 101.0, 99.5, 100.5, 102.0])
        rv = realized_variance_from_prices(prices, 1.0 / 252.0)
        assert float(rv) > 0


class TestDoubleBarrier:
    def test_alive_in_range(self):
        from ql_jax.instruments.double_barrier import DoubleBarrierOption, DoubleBarrierType
        opt = DoubleBarrierOption(
            barrier_lo=90, barrier_hi=110, barrier_type=DoubleBarrierType.KNOCK_OUT,
            strike=100, is_call=True, maturity=1.0
        )
        path = jnp.array([100.0, 102.0, 98.0, 105.0])
        assert bool(opt.is_alive(path))

    def test_knocked_out(self):
        from ql_jax.instruments.double_barrier import DoubleBarrierOption, DoubleBarrierType
        opt = DoubleBarrierOption(
            barrier_lo=90, barrier_hi=110, barrier_type=DoubleBarrierType.KNOCK_OUT,
            strike=100, is_call=True, maturity=1.0
        )
        path = jnp.array([100.0, 102.0, 115.0, 105.0])
        assert not bool(opt.is_alive(path))


class TestMargrabeOption:
    def test_payoff(self):
        from ql_jax.instruments.margrabe import MargrabeOption
        opt = MargrabeOption(maturity=1.0)
        assert float(opt.payoff(110.0, 100.0)) == pytest.approx(10.0)
        assert float(opt.payoff(90.0, 100.0)) == 0.0


class TestForwardStartOption:
    def test_payoff(self):
        from ql_jax.instruments.forward_start import ForwardVanillaOption
        opt = ForwardVanillaOption(moneyness=1.0, reset_date=0.5, maturity=1.0, is_call=True)
        # Strike set at spot_at_reset (100), final spot 110
        assert float(opt.payoff(100.0, 110.0)) == pytest.approx(10.0)
        assert float(opt.payoff(100.0, 90.0)) == 0.0


class TestInflationSwap:
    def test_zcis_npv(self):
        from ql_jax.instruments.inflation_swap import ZeroCouponInflationSwap
        swap = ZeroCouponInflationSwap(
            notional=1e6, fixed_rate=0.02, maturity=5.0,
            base_cpi=250.0, is_payer=True
        )
        # 10% inflation over 5 years
        npv = swap.npv(275.0, 0.9)
        assert float(npv) != 0.0  # should have non-zero value


class TestConvertibleBond:
    def test_conversion_value(self):
        from ql_jax.instruments.convertible_bond import ConvertibleBond
        cb = ConvertibleBond(face_value=1000, coupon_rate=0.05,
                             conversion_ratio=20, maturity=5.0)
        assert float(cb.conversion_value(60.0)) == pytest.approx(1200.0)
        assert float(cb.conversion_price) == pytest.approx(50.0)
        assert float(cb.parity(60.0)) == pytest.approx(1.2)


class TestTwoAssetCorrelation:
    def test_payoff(self):
        from ql_jax.instruments.two_asset import TwoAssetCorrelationOption
        opt = TwoAssetCorrelationOption(strike1=100, strike2=50, is_call=True, maturity=1.0)
        # S1=110 > K1=100, S2=60 > K2=50 -> payoff = 10
        assert float(opt.payoff(110.0, 60.0)) == pytest.approx(10.0)
        # S2=40 < K2=50 -> payoff = 0
        assert float(opt.payoff(110.0, 40.0)) == pytest.approx(0.0)
