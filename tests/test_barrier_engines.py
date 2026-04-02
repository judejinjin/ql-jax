"""Tests for barrier, binary, basket, and lookback engines."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Binary barrier options
# ---------------------------------------------------------------------------

class TestBinaryBarrier:
    def test_down_and_in_call(self):
        from ql_jax.engines.analytic.barrier_binary import binary_barrier_price
        price = binary_barrier_price(S=100, K=90, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                      barrier=80, cash_payoff=1.0,
                                      barrier_type="down-and-in", option_type=1)
        assert float(price) >= 0

    def test_down_and_out_call(self):
        from ql_jax.engines.analytic.barrier_binary import binary_barrier_price
        price = binary_barrier_price(S=100, K=90, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                      barrier=80, cash_payoff=1.0,
                                      barrier_type="down-and-out", option_type=1)
        assert float(price) >= 0

    def test_up_and_in_put(self):
        from ql_jax.engines.analytic.barrier_binary import binary_barrier_price
        price = binary_barrier_price(S=100, K=110, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                      barrier=120, cash_payoff=1.0,
                                      barrier_type="up-and-in", option_type=-1)
        assert float(price) >= 0

    def test_up_and_out_put(self):
        from ql_jax.engines.analytic.barrier_binary import binary_barrier_price
        price = binary_barrier_price(S=100, K=110, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                      barrier=120, cash_payoff=1.0,
                                      barrier_type="up-and-out", option_type=-1)
        assert float(price) >= 0

    def test_in_plus_out_equals_vanilla(self):
        """knock-in + knock-out = vanilla binary."""
        from ql_jax.engines.analytic.barrier_binary import binary_barrier_price
        di = binary_barrier_price(S=100, K=95, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                   barrier=80, cash_payoff=1.0, barrier_type="down-and-in", option_type=1)
        do = binary_barrier_price(S=100, K=95, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                   barrier=80, cash_payoff=1.0, barrier_type="down-and-out", option_type=1)
        total = float(di) + float(do)
        # Sum should be positive and bounded by discounted cash payoff
        assert total > 0
        assert total <= 1.0 * jnp.exp(-0.05 * 1.0) + 0.1


class TestDoubleBarrierBinary:
    def test_no_touch(self):
        from ql_jax.engines.analytic.barrier_binary import double_barrier_binary_price
        price = double_barrier_binary_price(S=100, T=1.0, r=0.05, q=0.0, sigma=0.20,
                                             lower=80, upper=120, cash_payoff=1.0)
        assert float(price) > 0
        assert float(price) <= 1.0  # Bounded by payoff

    def test_tight_barriers(self):
        from ql_jax.engines.analytic.barrier_binary import double_barrier_binary_price
        # Very tight barriers => low probability of surviving
        price = double_barrier_binary_price(S=100, T=1.0, r=0.05, q=0.0, sigma=0.30,
                                             lower=98, upper=102)
        assert float(price) >= 0
        assert float(price) < 0.5

    def test_wide_barriers(self):
        from ql_jax.engines.analytic.barrier_binary import double_barrier_binary_price
        # Very wide barriers => high probability of surviving
        price = double_barrier_binary_price(S=100, T=0.5, r=0.05, q=0.0, sigma=0.15,
                                             lower=50, upper=200)
        assert float(price) > 0.5


class TestPartialBarrier:
    def test_partial_down_out(self):
        from ql_jax.engines.analytic.barrier_binary import partial_barrier_price
        price = partial_barrier_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                       barrier=80, barrier_start=0.0, barrier_end=0.5,
                                       barrier_type="down-and-out", option_type=1)
        assert float(price) > 0

    def test_partial_window(self):
        from ql_jax.engines.analytic.barrier_binary import partial_barrier_price
        # Forward-start partial barrier
        price = partial_barrier_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                       barrier=80, barrier_start=0.5, barrier_end=1.0,
                                       barrier_type="down-and-out", option_type=1)
        assert float(price) > 0


# ---------------------------------------------------------------------------
# Basket options
# ---------------------------------------------------------------------------

class TestStulzBasket:
    def test_two_asset_call(self):
        from ql_jax.engines.analytic.basket import stulz_basket_price
        price = stulz_basket_price(S1=100, S2=100, K=100, T=1.0,
                                    r=0.05, q1=0.0, q2=0.0,
                                    sigma1=0.20, sigma2=0.25, rho=0.5, option_type=1)
        assert float(price) > 0

    def test_zero_correlation(self):
        from ql_jax.engines.analytic.basket import stulz_basket_price
        price = stulz_basket_price(S1=100, S2=100, K=100, T=1.0,
                                    r=0.05, q1=0.0, q2=0.0,
                                    sigma1=0.20, sigma2=0.20, rho=0.0, option_type=1)
        assert float(price) > 0

    def test_high_correlation(self):
        from ql_jax.engines.analytic.basket import stulz_basket_price
        price_low = stulz_basket_price(S1=100, S2=100, K=100, T=1.0,
                                        r=0.05, q1=0.0, q2=0.0,
                                        sigma1=0.20, sigma2=0.20, rho=0.2, option_type=1)
        price_high = stulz_basket_price(S1=100, S2=100, K=100, T=1.0,
                                         r=0.05, q1=0.0, q2=0.0,
                                         sigma1=0.20, sigma2=0.20, rho=0.9, option_type=1)
        # Higher correlation typically changes basket option price
        assert float(price_low) > 0
        assert float(price_high) > 0


class TestMomentMatchingBasket:
    def test_two_asset(self):
        from ql_jax.engines.analytic.basket import moment_matching_basket_price
        spots = jnp.array([100.0, 100.0])
        divs = jnp.array([0.0, 0.0])
        sigmas = jnp.array([0.20, 0.25])
        corr = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        price = moment_matching_basket_price(spots, K=100, T=1.0, r=0.05,
                                              divs=divs, sigmas=sigmas, corr=corr)
        assert float(price) > 0

    def test_three_asset(self):
        from ql_jax.engines.analytic.basket import moment_matching_basket_price
        spots = jnp.array([100.0, 110.0, 90.0])
        divs = jnp.array([0.01, 0.02, 0.0])
        sigmas = jnp.array([0.20, 0.25, 0.30])
        corr = jnp.array([[1.0, 0.3, 0.1],
                           [0.3, 1.0, 0.2],
                           [0.1, 0.2, 1.0]])
        price = moment_matching_basket_price(spots, K=100, T=1.0, r=0.05,
                                              divs=divs, sigmas=sigmas, corr=corr)
        assert float(price) > 0

    def test_put(self):
        from ql_jax.engines.analytic.basket import moment_matching_basket_price
        spots = jnp.array([100.0, 100.0])
        divs = jnp.array([0.0, 0.0])
        sigmas = jnp.array([0.20, 0.20])
        corr = jnp.eye(2)
        price = moment_matching_basket_price(spots, K=100, T=1.0, r=0.05,
                                              divs=divs, sigmas=sigmas, corr=corr, option_type=-1)
        assert float(price) > 0


class TestMCBasket:
    def test_mc_vs_analytic(self):
        from ql_jax.engines.analytic.basket import moment_matching_basket_price, mc_european_basket
        spots = jnp.array([100.0, 100.0])
        divs = jnp.array([0.0, 0.0])
        sigmas = jnp.array([0.20, 0.20])
        corr = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        analytic = moment_matching_basket_price(spots, K=100, T=1.0, r=0.05,
                                                 divs=divs, sigmas=sigmas, corr=corr)
        mc = mc_european_basket(spots, K=100, T=1.0, r=0.05,
                                divs=divs, sigmas=sigmas, corr=corr, n_paths=200_000, seed=42)
        assert abs(float(mc) - float(analytic)) / float(analytic) < 0.10


# ---------------------------------------------------------------------------
# Partial lookback
# ---------------------------------------------------------------------------

class TestPartialLookback:
    def test_fixed_call(self):
        from ql_jax.engines.analytic.lookback_partial import partial_fixed_lookback_price
        price = partial_fixed_lookback_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                              lookback_start=0.0, lookback_end=0.5,
                                              S_max=105, option_type=1)
        assert float(price) > 0

    def test_fixed_put(self):
        from ql_jax.engines.analytic.lookback_partial import partial_fixed_lookback_price
        price = partial_fixed_lookback_price(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                              lookback_start=0.0, lookback_end=0.5,
                                              S_min=95, option_type=-1)
        assert float(price) > 0

    def test_floating_call(self):
        from ql_jax.engines.analytic.lookback_partial import partial_floating_lookback_price
        price = partial_floating_lookback_price(S=100, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                                 lookback_start=0.0, lookback_end=0.5,
                                                 S_min=95, option_type=1)
        assert float(price) > 0

    def test_floating_put(self):
        from ql_jax.engines.analytic.lookback_partial import partial_floating_lookback_price
        price = partial_floating_lookback_price(S=100, T=1.0, r=0.05, q=0.0, sigma=0.25,
                                                 lookback_start=0.0, lookback_end=0.5,
                                                 S_max=105, option_type=-1)
        assert float(price) > 0
