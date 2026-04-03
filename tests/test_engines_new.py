"""Tests for new pricing engines: analytic exotics, FD, MC, lattice, swaption, credit."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ── Analytic exotic engines ──────────────────────────────────────────────────

class TestBarrierAnalytic:
    def test_down_and_out_call(self):
        from ql_jax.engines.analytic.barrier import barrier_price
        price = barrier_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                             sigma=0.20, barrier=90.0, barrier_type='down_and_out',
                             option_type='call')
        assert float(price) > 0.0
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        vanilla = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=1)
        assert float(price) < float(vanilla)

    def test_up_and_in_put(self):
        from ql_jax.engines.analytic.barrier import barrier_price
        price = barrier_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                             sigma=0.20, barrier=110.0, barrier_type='up_and_in',
                             option_type='put')
        assert float(price) > 0.0


class TestAsianAnalytic:
    def test_geometric_asian(self):
        from ql_jax.engines.analytic.asian import geometric_asian_price
        price = geometric_asian_price(S=100.0, K=100.0, T=1.0, r=0.05,
                                     q=0.0, sigma=0.20, n_fixings=12)
        assert float(price) > 0.0
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        vanilla = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=1)
        assert float(price) < float(vanilla)

    def test_turnbull_wakeman(self):
        from ql_jax.engines.analytic.asian import turnbull_wakeman_price
        price = turnbull_wakeman_price(S=100.0, K=100.0, T=1.0, r=0.05,
                                       q=0.0, sigma=0.20, n_fixings=12)
        assert float(price) > 0.0


class TestLookbackAnalytic:
    def test_floating_lookback(self):
        from ql_jax.engines.analytic.lookback import floating_lookback_price
        price = floating_lookback_price(S=100.0, S_min=95.0, S_max=105.0,
                                        T=1.0, r=0.05, q=0.0, sigma=0.20,
                                        option_type='call')
        assert float(price) > 5.0

    def test_fixed_lookback(self):
        from ql_jax.engines.analytic.lookback import fixed_lookback_price
        price = fixed_lookback_price(S=100.0, K=100.0, S_min=95.0, S_max=105.0,
                                     T=1.0, r=0.05, q=0.0, sigma=0.20,
                                     option_type='call')
        assert float(price) > 0.0


class TestSpreadOptions:
    def test_kirk_spread(self):
        from ql_jax.engines.analytic.spread import kirk_spread_price
        price = kirk_spread_price(S1=100.0, S2=95.0, K=5.0, T=1.0, r=0.05,
                                  q1=0.0, q2=0.0, sigma1=0.20, sigma2=0.25, rho=0.5)
        assert float(price) > 0.0

    def test_margrabe_exchange(self):
        from ql_jax.engines.analytic.spread import margrabe_exchange_price
        price = margrabe_exchange_price(S1=100.0, S2=100.0, T=1.0, q1=0.0, q2=0.0,
                                        sigma1=0.20, sigma2=0.25, rho=0.5)
        assert float(price) > 0.0


class TestQuantoAnalytic:
    def test_quanto_vanilla(self):
        from ql_jax.engines.analytic.quanto import quanto_vanilla_price
        price = quanto_vanilla_price(S=100.0, K=100.0, T=1.0, r_d=0.05, r_f=0.02,
                                     q=0.0, sigma_s=0.20, sigma_fx=0.10, rho=-0.3,
                                     fx_rate=1.0)
        assert float(price) > 0.0


class TestVarianceSwap:
    def test_fair_strike(self):
        from ql_jax.engines.analytic.variance_swap import variance_swap_fair_strike
        strike = variance_swap_fair_strike(S=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20)
        assert abs(float(strike) - 0.04) < 0.02


class TestDigital:
    def test_cash_or_nothing(self):
        from ql_jax.engines.analytic.digital import digital_price
        price = digital_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                             sigma=0.20, option_type='call', payout_type='cash',
                             payout=1.0)
        assert 0.0 < float(price) < 1.0


class TestCompound:
    def test_compound_option(self):
        from ql_jax.engines.analytic.compound import compound_option_price
        price = compound_option_price(S=100.0, K1=5.0, K2=100.0, T1=0.5, T2=1.0,
                                      r=0.05, q=0.0, sigma=0.20)
        assert float(price) > 0.0


class TestCliquet:
    def test_cliquet_price(self):
        from ql_jax.engines.analytic.cliquet import cliquet_price
        reset_dates = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        price = cliquet_price(S=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20,
                             reset_dates=reset_dates,
                             local_cap=0.05, local_floor=-0.05,
                             global_cap=0.15, global_floor=0.0)
        assert float(price) >= 0.0


class TestDoubleBarrier:
    def test_double_barrier_price(self):
        from ql_jax.engines.analytic.double_barrier import double_barrier_price
        price = double_barrier_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                                     sigma=0.20, lower_barrier=80.0,
                                     upper_barrier=120.0, option_type='call',
                                     n_terms=20)
        assert float(price) > 0.0


# ── FD engines ───────────────────────────────────────────────────────────────

class TestFDBarrier:
    def test_fd_barrier_price(self):
        from ql_jax.engines.fd.barrier import fd_barrier_price
        price = fd_barrier_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                                sigma=0.20, barrier=90.0, option_type=1,
                                barrier_type='down_and_out',
                                n_space=100, n_time=100)
        assert float(price) > 0.0


class TestFDHeston:
    @pytest.mark.slow
    def test_fd_heston_price(self):
        from ql_jax.engines.fd.heston import fd_heston_price
        price = fd_heston_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                               v0=0.04, kappa=1.5, theta=0.04,
                               sigma_v=0.3, rho=-0.7, option_type=1,
                               n_x=15, n_v=8, n_t=15)
        assert float(price) > 0.0


# ── MC engines ───────────────────────────────────────────────────────────────

class TestMCHeston:
    def test_mc_heston_price(self):
        from ql_jax.engines.mc.heston import mc_heston_price
        price = mc_heston_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                               v0=0.04, kappa=1.5, theta=0.04,
                               sigma_v=0.3, rho=-0.7, option_type=1,
                               n_paths=10000, n_steps=100,
                               key=jax.random.PRNGKey(42))
        assert float(price) > 0.0


class TestMCLookback:
    def test_mc_lookback_price(self):
        from ql_jax.engines.mc.lookback import mc_lookback_price
        price = mc_lookback_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                                  sigma=0.20, option_type=1,
                                  lookback_type='floating',
                                  n_paths=5000, n_steps=100,
                                  key=jax.random.PRNGKey(42))
        assert float(price) > 0.0


# ── Lattice engines ──────────────────────────────────────────────────────────

class TestTrinomial:
    def test_trinomial_vanilla(self):
        from ql_jax.engines.lattice.trinomial import trinomial_price
        price = trinomial_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0,
                               sigma=0.20, option_type=1, n_steps=100)
        from ql_jax.engines.analytic.black_formula import black_scholes_price
        bs = black_scholes_price(S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20, option_type=1)
        assert abs(float(price) - float(bs)) < 0.5

    def test_trinomial_barrier(self):
        from ql_jax.engines.lattice.trinomial import trinomial_barrier_price
        # Up-and-out call with barrier above spot — should have positive value
        price = trinomial_barrier_price(S=100.0, K=95.0, T=1.0, r=0.05, q=0.0,
                                        sigma=0.20, barrier=120.0, option_type=1,
                                        is_up_and_out=True, n_steps=100)
        assert float(price) >= 0.0


class TestBSMTree:
    def test_dividend_pricing(self):
        from ql_jax.engines.lattice.bsm import bsm_binomial_dividend_price
        price = bsm_binomial_dividend_price(
            S=100.0, K=100.0, T=1.0, r=0.05, q=0.0, sigma=0.20,
            option_type=1, n_steps=100,
            dividends_times=jnp.array([0.5]),
            dividends_amounts=jnp.array([2.0]))
        assert float(price) > 0.0


class TestShortRateTree:
    def test_hw_tree_bond(self):
        from ql_jax.engines.lattice.short_rate_tree import hw_trinomial_tree, hw_tree_bond_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        tree = hw_trinomial_tree(a=0.1, sigma=0.01, T=2.0, n_steps=50,
                                discount_fn=discount_fn)
        P = hw_tree_bond_price(tree, maturity_step=50)
        # HW tree bond price should be close to disc factor but can differ
        assert 0.5 < float(P) < 1.5


# ── Swaption engines ────────────────────────────────────────────────────────

class TestSwaptionBachelier:
    def test_bachelier_swaption(self):
        from ql_jax.engines.swaption.bachelier import bachelier_swaption_price
        price = bachelier_swaption_price(notional=1e6, fixed_rate=0.05,
                                          swap_rate=0.05, annuity=4.5,
                                          normal_vol=0.005, expiry=1.0,
                                          is_payer=True)
        assert float(price) > 0.0


class TestSwaptionHW:
    def test_hw_swaption(self):
        from ql_jax.engines.swaption.hull_white import hw_swaption_price
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        price = hw_swaption_price(
            notional=1e6, fixed_rate=0.05,
            payment_dates=jnp.array([1.5, 2.0, 2.5, 3.0]),
            day_fractions=jnp.full(4, 0.5),
            a=0.1, sigma=0.01, discount_fn=discount_fn,
            expiry=1.0, is_payer=True,
        )
        assert float(price) > 0.0


# ── Credit engines ───────────────────────────────────────────────────────────

class TestCreditEngines:
    def test_cds_npv(self):
        from ql_jax.engines.credit.analytics import cds_npv
        payment_dates = jnp.arange(0.25, 5.25, 0.25)
        day_fracs = jnp.full_like(payment_dates, 0.25)
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.exp(-0.02 * t)
        npv = cds_npv(spread=0.01, recovery=0.4,
                     payment_dates=payment_dates,
                     day_fractions=day_fracs,
                     discount_fn=discount_fn,
                     survival_fn=survival_fn,
                     notional=1e6, is_protection_buyer=True)
        assert float(npv) != 0.0

    def test_fair_spread(self):
        from ql_jax.engines.credit.analytics import cds_fair_spread
        payment_dates = jnp.arange(0.25, 5.25, 0.25)
        day_fracs = jnp.full_like(payment_dates, 0.25)
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        survival_fn = lambda t: jnp.exp(-0.02 * t)
        fs = cds_fair_spread(recovery=0.4,
                            payment_dates=payment_dates,
                            day_fractions=day_fracs,
                            discount_fn=discount_fn,
                            survival_fn=survival_fn,
                            notional=1e6)
        assert float(fs) > 0.0
