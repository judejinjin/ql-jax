"""Tests for additional_gaps Phases 7-10.

Phase 7: Vanilla option engines (CEV, integral, QD+, cash dividend, BSM-HW, Heston-HW, Vasicek)
Phase 8: Asian engines (Choi, Lévy)
Phase 9: Barrier engines (vanna-volga, perturbative)
Phase 10: Basket/spread engines (operator splitting, single-factor BSM basket)
"""

import jax
import jax.numpy as jnp
import pytest
import math

jax.config.update("jax_enable_x64", True)


# =========================================================================
# Phase 7: Vanilla option engines
# =========================================================================

class TestCEVEngine:
    def test_cev_call_positive(self):
        from ql_jax.engines.analytic.cev import cev_price
        price = cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 0.5, 1)
        assert float(price) > 0.0

    def test_cev_put_positive(self):
        from ql_jax.engines.analytic.cev import cev_price
        price = cev_price(100.0, 110.0, 1.0, 0.05, 0.0, 0.2, 0.5, -1)
        assert float(price) > 0.0

    def test_cev_put_call_parity(self):
        """CEV put-call parity check."""
        from ql_jax.engines.analytic.cev import cev_price
        call = cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 0.5, 1)
        put = cev_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 0.5, -1)
        F = 100.0 * math.exp(0.05)
        disc = math.exp(-0.05)
        assert float(call) - float(put) == pytest.approx(disc * (F - 100.0), rel=0.05)


class TestIntegralEngine:
    def test_matches_bs(self):
        from ql_jax.engines.analytic.integral import integral_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 1)
        integ = integral_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 1, n_points=500)
        assert float(integ) == pytest.approx(float(bs), rel=0.005)

    def test_put_matches_bs(self):
        from ql_jax.engines.analytic.integral import integral_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs = black_scholes_price(100.0, 110.0, 0.5, 0.03, 0.01, 0.3, -1)
        integ = integral_price(100.0, 110.0, 0.5, 0.03, 0.01, 0.3, -1, n_points=500)
        assert float(integ) == pytest.approx(float(bs), rel=0.005)


class TestQDPlusAmerican:
    def test_call_ge_european(self):
        from ql_jax.engines.analytic.qdplus_american import qdplus_american_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        euro = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 1)
        amer = qdplus_american_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 1)
        assert float(amer) >= float(euro) - 0.01

    def test_put_ge_european(self):
        from ql_jax.engines.analytic.qdplus_american import qdplus_american_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        euro = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, -1)
        amer = qdplus_american_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, -1)
        assert float(amer) >= float(euro) - 0.01

    def test_put_ge_intrinsic(self):
        from ql_jax.engines.analytic.qdplus_american import qdplus_american_price
        price = qdplus_american_price(80.0, 100.0, 1.0, 0.05, 0.0, 0.25, -1)
        intrinsic = 20.0
        assert float(price) >= intrinsic - 0.01

    def test_deep_itm_call_with_dividends(self):
        from ql_jax.engines.analytic.qdplus_american import qdplus_american_price
        # With high dividends, American call has exercise premium
        price = qdplus_american_price(120.0, 100.0, 1.0, 0.05, 0.10, 0.25, 1)
        assert float(price) > 20.0  # Should be worth at least intrinsic


class TestCashDividendEuropean:
    def test_dividend_reduces_call(self):
        from ql_jax.engines.analytic.cash_dividend_european import cash_dividend_european_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, 1)
        div = cash_dividend_european_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, 1,
                                           dividend_times=[0.5],
                                           dividend_amounts=[5.0])
        assert float(div) < float(bs)

    def test_no_dividends_matches_bs(self):
        from ql_jax.engines.analytic.cash_dividend_european import cash_dividend_european_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, 1)
        div = cash_dividend_european_price(100.0, 100.0, 1.0, 0.05, 0.0, 0.25, 1)
        assert float(div) == pytest.approx(float(bs))


class TestBSMHullWhite:
    def test_flat_rate_matches_bs(self):
        from ql_jax.engines.analytic.bsm_hull_white import bsm_hull_white_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        # With zero rate vol, should match BS
        hybrid = bsm_hull_white_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
                                       0.1, 0.0, 0.0, option_type=1)
        bs = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 1)
        assert float(hybrid) == pytest.approx(float(bs), rel=0.01)

    def test_positive_price(self):
        from ql_jax.engines.analytic.bsm_hull_white import bsm_hull_white_price
        price = bsm_hull_white_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
                                      0.1, 0.01, -0.3, option_type=1)
        assert float(price) > 0.0


class TestHestonHullWhite:
    def test_positive_price(self):
        from ql_jax.engines.analytic.heston_hull_white import heston_hull_white_price
        price = heston_hull_white_price(
            100.0, 100.0, 1.0, 0.05, 0.02,
            0.04, 1.5, 0.04, 0.3, -0.7,
            0.1, 0.01, -0.2,
            option_type=1,
        )
        assert float(price) > 0.0

    def test_put_call_parity(self):
        from ql_jax.engines.analytic.heston_hull_white import heston_hull_white_price
        args = dict(S=100.0, K=105.0, T=0.5, r0=0.05, q=0.02,
                    v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho_sv=-0.7,
                    a_hw=0.1, sigma_hw=0.005, rho_sr=-0.1, n_points=128)
        call = heston_hull_white_price(**args, option_type=1)
        put = heston_hull_white_price(**args, option_type=-1)
        F = 100.0 * jnp.exp((0.05 - 0.02) * 0.5)
        disc = jnp.exp(-0.05 * 0.5)
        parity = disc * (F - 105.0)
        assert float(call) - float(put) == pytest.approx(float(parity), rel=0.05)


class TestExpFittingHeston:
    def test_matches_standard_heston(self):
        from ql_jax.engines.analytic.exp_fitting_heston import exp_fitting_heston_price
        from ql_jax.engines.analytic.heston import heston_price
        args = (100.0, 100.0, 1.0, 0.05, 0.02, 0.04, 1.5, 0.04, 0.3, -0.7)
        std = heston_price(*args, option_type=1)
        exp_fit = exp_fitting_heston_price(*args, option_type=1)
        assert float(exp_fit) == pytest.approx(float(std), rel=0.01)


class TestVasicekEuropean:
    def test_positive_price(self):
        from ql_jax.engines.analytic.vasicek_european import vasicek_european_price
        price = vasicek_european_price(
            100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
            0.5, 0.05, 0.01, -0.3, option_type=1,
        )
        assert float(price) > 0.0

    def test_zero_rate_vol_matches_bs(self):
        from ql_jax.engines.analytic.vasicek_european import vasicek_european_price
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        vasicek = vasicek_european_price(
            100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
            0.5, 0.05, 0.0, 0.0, option_type=1,
        )
        bs = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 1)
        assert float(vasicek) == pytest.approx(float(bs), rel=0.01)


# =========================================================================
# Phase 8: Asian engines
# =========================================================================

class TestChoiAsian:
    def test_call_positive(self):
        from ql_jax.engines.analytic.asian_choi import choi_asian_price
        price = choi_asian_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 12)
        assert float(price) > 0.0

    def test_put_positive(self):
        from ql_jax.engines.analytic.asian_choi import choi_asian_price
        price = choi_asian_price(100.0, 110.0, 1.0, 0.05, 0.02, 0.25, 12, 'put')
        assert float(price) > 0.0

    def test_consistent_with_geometric(self):
        from ql_jax.engines.analytic.asian_choi import choi_asian_price
        from ql_jax.engines.analytic.asian import geometric_asian_price
        # Arithmetic >= geometric for calls
        arith = choi_asian_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 50)
        geom = geometric_asian_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25, 50)
        assert float(arith) >= float(geom) - 0.01


class TestLevyAsian:
    def test_call_positive(self):
        from ql_jax.engines.analytic.asian_choi import levy_asian_price
        price = levy_asian_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25)
        assert float(price) > 0.0

    def test_put_positive(self):
        from ql_jax.engines.analytic.asian_choi import levy_asian_price
        price = levy_asian_price(100.0, 110.0, 1.0, 0.05, 0.02, 0.25, 'put')
        assert float(price) > 0.0


# =========================================================================
# Phase 9: Barrier engines
# =========================================================================

class TestVannaVolgaBarrier:
    def test_flat_vol(self):
        """With flat vol, should match standard barrier."""
        from ql_jax.engines.analytic.barrier_vanna_volga import vanna_volga_barrier_price
        from ql_jax.engines.analytic.barrier import barrier_price
        std = barrier_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
                                    80.0, 0.0, 'call', 'down_and_out')
        vv = vanna_volga_barrier_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
                                        80.0, 0.0, 'call', 'down_and_out')
        assert float(vv) == pytest.approx(float(std), rel=1e-10)

    def test_smile_adjustment(self):
        from ql_jax.engines.analytic.barrier_vanna_volga import vanna_volga_barrier_price
        price = vanna_volga_barrier_price(
            100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
            80.0, 0.0, 'call', 'down_and_out',
            sigma_25d_put=0.28, sigma_25d_call=0.23,
        )
        assert float(price) > 0.0


class TestPerturbativeBarrier:
    def test_flat_vol(self):
        """With zero slope/curv, matches standard."""
        from ql_jax.engines.analytic.barrier_perturbative import perturbative_barrier_price
        from ql_jax.engines.analytic.barrier import barrier_price
        std = barrier_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
                                    80.0, 0.0, 'call', 'down_and_out')
        pert = perturbative_barrier_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.25,
                                           80.0, 0.0, 'call', 'down_and_out')
        assert float(pert) == pytest.approx(float(std), rel=1e-10)


# =========================================================================
# Phase 10: Basket/spread engines
# =========================================================================

class TestOperatorSplittingSpread:
    def test_call_positive(self):
        from ql_jax.engines.analytic.basket_single_factor import operator_splitting_spread_price
        price = operator_splitting_spread_price(
            100.0, 90.0, 1.0, 0.3, 0.25, 0.6, 5.0, 0.95, option_type=1,
        )
        assert float(price) > 0.0

    def test_atm_spread(self):
        from ql_jax.engines.analytic.basket_single_factor import operator_splitting_spread_price
        price = operator_splitting_spread_price(
            100.0, 100.0, 0.5, 0.2, 0.2, 0.5, 0.0, 0.98, option_type=1,
        )
        assert float(price) > 0.0

    def test_put_positive(self):
        from ql_jax.engines.analytic.basket_single_factor import operator_splitting_spread_price
        price = operator_splitting_spread_price(
            100.0, 110.0, 1.0, 0.3, 0.25, 0.7, 0.0, 0.95, option_type=-1,
        )
        assert float(price) > 0.0


class TestSingleFactorBasket:
    def test_two_asset_basket_call(self):
        from ql_jax.engines.analytic.basket_single_factor import single_factor_bsm_basket_price
        import jax.numpy as jnp
        price = single_factor_bsm_basket_price(
            forwards=jnp.array([100.0, 100.0]),
            vols=jnp.array([0.25, 0.30]),
            weights=jnp.array([0.5, 0.5]),
            corr_matrix=jnp.array([[1.0, 0.5], [0.5, 1.0]]),
            K=100.0, T=1.0, df=0.95, option_type=1,
        )
        assert float(price) > 0.0

    def test_single_asset_matches_black(self):
        from ql_jax.engines.analytic.basket_single_factor import single_factor_bsm_basket_price
        from ql_jax.engines.analytic.black_formula import black_price
        import jax.numpy as jnp
        basket = single_factor_bsm_basket_price(
            forwards=jnp.array([100.0]),
            vols=jnp.array([0.25]),
            weights=jnp.array([1.0]),
            corr_matrix=jnp.array([[1.0]]),
            K=100.0, T=1.0, df=0.95, option_type=1,
        )
        bk = black_price(100.0, 100.0, 1.0, 0.25, 0.95, 1)
        assert float(basket) == pytest.approx(float(bk), rel=0.01)

    def test_three_asset_basket(self):
        from ql_jax.engines.analytic.basket_single_factor import single_factor_bsm_basket_price
        import jax.numpy as jnp
        corr = jnp.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
        price = single_factor_bsm_basket_price(
            forwards=jnp.array([100.0, 110.0, 90.0]),
            vols=jnp.array([0.20, 0.25, 0.30]),
            weights=jnp.array([0.4, 0.3, 0.3]),
            corr_matrix=corr,
            K=100.0, T=0.5, df=0.97, option_type=1,
        )
        assert float(price) > 0.0
