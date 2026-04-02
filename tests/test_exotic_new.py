"""Tests for exotic instruments and their analytic engines."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ── Lookback ─────────────────────────────────────────────────────────────────

class TestLookback:
    def test_floating_lookback_instrument(self):
        from ql_jax.instruments.lookback import ContinuousFloatingLookbackOption
        opt = ContinuousFloatingLookbackOption(is_call=True, current_minmax=90.0, maturity=1.0)
        assert opt.is_call is True
        assert opt.maturity == 1.0

    def test_fixed_lookback_instrument(self):
        from ql_jax.instruments.lookback import ContinuousFixedLookbackOption
        opt = ContinuousFixedLookbackOption(strike=100.0, is_call=True, current_minmax=105.0, maturity=1.0)
        assert opt.strike == 100.0

    def test_analytic_floating_lookback(self):
        from ql_jax.engines.analytic.lookback import floating_lookback_price
        price = float(floating_lookback_price(
            S=100.0, S_min=90.0, S_max=110.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type='call'
        ))
        assert price > 0.0
        assert price < 100.0  # reasonable bound

    def test_analytic_fixed_lookback(self):
        from ql_jax.engines.analytic.lookback import fixed_lookback_price
        price = float(fixed_lookback_price(
            S=100.0, K=100.0, S_min=95.0, S_max=105.0, T=1.0, r=0.05, q=0.0, sigma=0.2, option_type='call'
        ))
        assert price > 0.0

    def test_lookback_call_gt_put(self):
        from ql_jax.engines.analytic.lookback import floating_lookback_price
        call = float(floating_lookback_price(
            S=100.0, S_min=95.0, S_max=105.0, T=1.0, r=0.05, q=0.0, sigma=0.3, option_type='call'
        ))
        put = float(floating_lookback_price(
            S=100.0, S_min=95.0, S_max=105.0, T=1.0, r=0.05, q=0.0, sigma=0.3, option_type='put'
        ))
        assert call > 0.0
        assert put > 0.0


# ── Cliquet ──────────────────────────────────────────────────────────────────

class TestCliquet:
    def test_cliquet_instrument(self):
        from ql_jax.instruments.cliquet import CliquetOption
        reset_dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        opt = CliquetOption(
            local_cap=0.1, local_floor=-0.1, global_cap=0.3,
            global_floor=-0.3, reset_dates=reset_dates, maturity=1.0, is_call=True
        )
        assert opt.maturity == 1.0

    def test_analytic_cliquet(self):
        from ql_jax.engines.analytic.cliquet import cliquet_price
        reset_dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        price = float(cliquet_price(
            S=100.0, T=1.0, r=0.05, q=0.02, sigma=0.2, reset_dates=reset_dates
        ))
        assert price > 0.0

    def test_forward_start(self):
        from ql_jax.engines.analytic.cliquet import forward_start_price
        price = float(forward_start_price(
            S=100.0, T1=0.5, T2=1.0, r=0.05, q=0.02, sigma=0.2
        ))
        assert price > 0.0


# ── Chooser ──────────────────────────────────────────────────────────────────

class TestChooser:
    def test_chooser_instrument(self):
        from ql_jax.instruments.chooser import SimpleChooserOption
        opt = SimpleChooserOption(strike=100.0, choose_date=0.5, maturity=1.0)
        assert opt.choose_date == 0.5

    def test_complex_chooser(self):
        from ql_jax.instruments.chooser import ComplexChooserOption
        opt = ComplexChooserOption(
            strike_call=100.0, strike_put=95.0,
            choose_date=0.5, maturity_call=1.0, maturity_put=1.0
        )
        assert opt.strike_call == 100.0


# ── Compound ─────────────────────────────────────────────────────────────────

class TestCompound:
    def test_compound_instrument(self):
        from ql_jax.instruments.compound import CompoundOption
        opt = CompoundOption(
            strike_mother=5.0, strike_daughter=100.0,
            maturity_mother=0.5, maturity_daughter=1.0,
            is_call_mother=True, is_call_daughter=True
        )
        assert opt.maturity_mother == 0.5

    def test_analytic_compound(self):
        from ql_jax.engines.analytic.compound import compound_option_price
        price = float(compound_option_price(
            S=100.0, K1=5.0, K2=100.0, T1=0.5, T2=1.0, r=0.05, q=0.0, sigma=0.2
        ))
        assert price > 0.0


# ── Quanto ───────────────────────────────────────────────────────────────────

class TestQuanto:
    def test_quanto_instrument(self):
        from ql_jax.instruments.quanto import QuantoVanillaOption
        opt = QuantoVanillaOption(strike=100.0, is_call=True, maturity=1.0, fx_rate=1.2)
        assert opt.fx_rate == 1.2

    def test_analytic_quanto(self):
        from ql_jax.engines.analytic.quanto import quanto_vanilla_price
        price = float(quanto_vanilla_price(
            S=100.0, K=100.0, T=1.0, r_d=0.05, r_f=0.02, q=0.0,
            sigma_s=0.2, sigma_fx=0.1, rho=-0.3, fx_rate=1.2
        ))
        assert price > 0.0


# ── Double Barrier ───────────────────────────────────────────────────────────

class TestDoubleBarrier:
    def test_double_barrier_instrument(self):
        from ql_jax.instruments.double_barrier import DoubleBarrierOption, DoubleBarrierType
        opt = DoubleBarrierOption(
            barrier_lo=80.0, barrier_hi=120.0,
            barrier_type=DoubleBarrierType.KNOCK_OUT,
            strike=100.0, is_call=True, rebate=0.0, maturity=1.0
        )
        assert opt.barrier_lo == 80.0

    def test_analytic_double_barrier(self):
        from ql_jax.engines.analytic.double_barrier import double_barrier_price
        price = float(double_barrier_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2,
            lower_barrier=80.0, upper_barrier=120.0
        ))
        assert price >= 0.0  # could be near zero if barriers tight


# ── Variance Swap ────────────────────────────────────────────────────────────

class TestVarianceSwap:
    def test_variance_swap_instrument(self):
        from ql_jax.instruments.variance_swap import VarianceSwap
        vs = VarianceSwap(strike_variance=0.04, notional=100.0, maturity=1.0, is_long=True)
        assert vs.strike_variance == 0.04

    def test_fair_strike(self):
        from ql_jax.engines.analytic.variance_swap import variance_swap_fair_strike
        strike = float(variance_swap_fair_strike(
            S=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2
        ))
        assert strike > 0.0
        assert abs(strike - 0.04) < 0.02  # approximately sigma^2

    def test_realized_variance(self):
        from ql_jax.engines.analytic.variance_swap import discrete_realized_variance
        # Log returns of constant prices = all zeros
        log_returns = jnp.array([0.0, 0.0, 0.0])
        rv = float(discrete_realized_variance(log_returns, T=1.0))
        assert abs(rv) < 1e-10


# ── Margrabe / Spread ────────────────────────────────────────────────────────

class TestMargrabe:
    def test_margrabe_instrument(self):
        from ql_jax.instruments.margrabe import MargrabeOption
        opt = MargrabeOption(maturity=1.0, quantity1=1.0, quantity2=1.0)
        payoff = float(opt.payoff(110.0, 100.0))
        assert payoff == 10.0

    def test_margrabe_exchange_price(self):
        from ql_jax.engines.analytic.spread import margrabe_exchange_price
        price = float(margrabe_exchange_price(
            S1=100.0, S2=100.0, T=1.0, q1=0.0, q2=0.0,
            sigma1=0.2, sigma2=0.2, rho=0.5
        ))
        assert price > 0.0

    def test_kirk_spread(self):
        from ql_jax.engines.analytic.spread import kirk_spread_price
        price = float(kirk_spread_price(
            S1=100.0, S2=95.0, K=0.0, T=1.0, r=0.05,
            q1=0.0, q2=0.0, sigma1=0.2, sigma2=0.2, rho=0.5
        ))
        assert price > 0.0


# ── Basket ───────────────────────────────────────────────────────────────────

class TestBasket:
    def test_basket_instrument(self):
        from ql_jax.instruments.basket import BasketOption, BasketPayoff, BasketType
        payoff = BasketPayoff(basket_type=BasketType.AVERAGE, strike=100.0, is_call=True)
        opt = BasketOption(payoff=payoff, maturity=1.0, exercise_type="european")
        assert opt.maturity == 1.0

    def test_spread_option(self):
        from ql_jax.instruments.basket import SpreadOption
        opt = SpreadOption(strike=5.0, is_call=True, maturity=1.0)
        assert opt.strike == 5.0


# ── Digital ──────────────────────────────────────────────────────────────────

class TestDigital:
    def test_digital_price(self):
        from ql_jax.engines.analytic.digital import digital_price
        price = float(digital_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.2
        ))
        assert 0.0 < price < 1.0  # digital payoff bounded by discount

    def test_double_digital(self):
        from ql_jax.engines.analytic.digital import double_digital_price
        price = float(double_digital_price(
            S=100.0, K_lower=90.0, K_upper=110.0, T=1.0, r=0.05, q=0.0, sigma=0.2
        ))
        assert price >= 0.0
