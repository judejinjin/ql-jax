"""Tests for equity instruments and pricing engines (Phase 4)."""

import pytest
import jax.numpy as jnp
import jax

from ql_jax._util.types import OptionType, BarrierType
from ql_jax.instruments.payoff import (
    plain_vanilla_payoff, cash_or_nothing_payoff,
    asset_or_nothing_payoff, straddle_payoff,
)
from ql_jax.instruments.exercise import EuropeanExercise, AmericanExercise
from ql_jax.instruments.vanilla_option import VanillaOption
from ql_jax.engines.analytic.black_formula import (
    black_scholes_price, black_price, bachelier_price,
    implied_volatility_black_scholes, bs_greeks,
)
from ql_jax.engines.analytic.heston import heston_price
from ql_jax.engines.lattice.binomial import binomial_price
from ql_jax.engines.fd.black_scholes import fd_black_scholes_price
from ql_jax.instruments.barrier import analytic_barrier_price
from ql_jax.instruments.asian import (
    analytic_continuous_geometric_asian_price,
    analytic_discrete_geometric_asian_price,
)


# ===== Test parameters =====
S0, K0, T0, R0, Q0, SIGMA0 = 100.0, 100.0, 1.0, 0.05, 0.02, 0.20


# ===== Payoffs =====

class TestPayoffs:
    def test_vanilla_call_itm(self):
        assert plain_vanilla_payoff(110.0, 100.0, OptionType.Call) == 10.0

    def test_vanilla_call_otm(self):
        assert plain_vanilla_payoff(90.0, 100.0, OptionType.Call) == 0.0

    def test_vanilla_put_itm(self):
        assert plain_vanilla_payoff(90.0, 100.0, OptionType.Put) == 10.0

    def test_vanilla_put_otm(self):
        assert plain_vanilla_payoff(110.0, 100.0, OptionType.Put) == 0.0

    def test_cash_or_nothing(self):
        p = cash_or_nothing_payoff(110.0, 100.0, 1.0, OptionType.Call)
        assert float(p) == 1.0
        p = cash_or_nothing_payoff(90.0, 100.0, 1.0, OptionType.Call)
        assert float(p) == 0.0

    def test_straddle(self):
        assert float(straddle_payoff(110.0, 100.0)) == 10.0
        assert float(straddle_payoff(90.0, 100.0)) == 10.0

    def test_payoff_vectorized(self):
        """vmap-friendly payoff."""
        spots = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
        payoffs = jax.vmap(lambda s: plain_vanilla_payoff(s, 100.0, OptionType.Call))(spots)
        expected = jnp.array([0.0, 0.0, 0.0, 10.0, 20.0])
        assert jnp.allclose(payoffs, expected)


# ===== Black-Scholes =====

class TestBlackScholes:
    def test_call_price_positive(self):
        p = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert float(p) > 0

    def test_put_price_positive(self):
        p = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        assert float(p) > 0

    def test_put_call_parity(self):
        """C - P = S*exp(-qT) - K*exp(-rT)."""
        c = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        p = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        parity = S0 * jnp.exp(-Q0 * T0) - K0 * jnp.exp(-R0 * T0)
        assert abs(float(c - p) - float(parity)) < 1e-10

    def test_deep_itm_call(self):
        """Deep ITM call ≈ S*exp(-qT) - K*exp(-rT)."""
        c = black_scholes_price(200.0, 100.0, 1.0, R0, Q0, SIGMA0, OptionType.Call)
        intrinsic = 200.0 * jnp.exp(-Q0) - 100.0 * jnp.exp(-R0)
        assert abs(float(c) - float(intrinsic)) < 1.0

    def test_at_expiry(self):
        """At T=0, value = intrinsic."""
        c = black_scholes_price(110.0, 100.0, 1e-10, R0, Q0, SIGMA0, OptionType.Call)
        assert abs(float(c) - 10.0) < 0.1

    def test_known_value(self):
        """Check against known BS price."""
        # S=100, K=100, T=1, r=5%, q=2%, sigma=20%
        # Expected call ≈ 9.23 (approx)
        c = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, OptionType.Call)
        assert 8.0 < float(c) < 11.0

    def test_vectorized(self):
        """Vectorized over strikes."""
        strikes = jnp.array([90.0, 95.0, 100.0, 105.0, 110.0])
        prices = jax.vmap(lambda k: black_scholes_price(
            S0, k, T0, R0, Q0, SIGMA0, OptionType.Call
        ))(strikes)
        # Prices should be monotonically decreasing in strike
        for i in range(len(prices) - 1):
            assert float(prices[i]) > float(prices[i + 1])


class TestBlackFormula:
    def test_black_price_vs_bsm(self):
        """Black formula with F = S*exp((r-q)*T), df = exp(-rT) should match BSM."""
        F = S0 * jnp.exp((R0 - Q0) * T0)
        df = jnp.exp(-R0 * T0)
        bp = black_price(F, K0, T0, SIGMA0, df, OptionType.Call)
        bsp = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert abs(float(bp) - float(bsp)) < 1e-8


class TestBachelier:
    def test_positive_price(self):
        p = bachelier_price(100.0, 100.0, 1.0, 20.0, 0.95, OptionType.Call)
        assert float(p) > 0

    def test_put_call_parity(self):
        df = jnp.exp(-R0 * T0)
        F = S0
        c = bachelier_price(F, K0, T0, 20.0, df, OptionType.Call)
        p = bachelier_price(F, K0, T0, 20.0, df, OptionType.Put)
        assert abs(float(c - p) - float(df * (F - K0))) < 1e-8


# ===== Implied Volatility =====

class TestImpliedVol:
    def test_round_trip(self):
        """IV from BS price should recover the original vol."""
        price = black_scholes_price(S0, K0, T0, R0, Q0, 0.25, OptionType.Call)
        iv = implied_volatility_black_scholes(
            float(price), S0, K0, T0, R0, Q0, OptionType.Call, guess=0.2
        )
        assert abs(iv - 0.25) < 0.001


# ===== Greeks via AD =====

class TestGreeks:
    def test_delta_call_between_0_1(self):
        g = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert 0.0 < g["delta"] < 1.0

    def test_delta_put_between_neg1_0(self):
        g = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        assert -1.0 < g["delta"] < 0.0

    def test_gamma_positive(self):
        g = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert g["gamma"] > 0

    def test_vega_positive(self):
        g = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert g["vega"] > 0

    def test_theta_negative_for_call(self):
        g = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert g["theta"] < 0  # time decay

    def test_put_call_delta_parity(self):
        gc = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        gp = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        # delta_call - delta_put ≈ exp(-qT)
        expected = float(jnp.exp(-Q0 * T0))
        assert abs(gc["delta"] - gp["delta"] - expected) < 0.01

    def test_gamma_same_for_call_put(self):
        gc = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        gp = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        assert abs(gc["gamma"] - gp["gamma"]) < 1e-6


# ===== Heston =====

class TestHeston:
    def test_reduces_to_bs(self):
        """Heston with very small vol-of-vol should approach BS."""
        # Small xi → nearly constant vol → BS-like
        hp = heston_price(
            S0, K0, T0, R0, Q0,
            v0=SIGMA0**2, kappa=5.0, theta=SIGMA0**2,
            xi=0.01, rho=0.0,
            option_type=OptionType.Call,
        )
        bsp = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        # Should be reasonably close
        assert abs(float(hp) - float(bsp)) < 1.0

    def test_positive_price(self):
        hp = heston_price(
            S0, K0, T0, R0, Q0,
            v0=0.04, kappa=2.0, theta=0.04,
            xi=0.5, rho=-0.7,
            option_type=OptionType.Call,
        )
        assert float(hp) > 0

    def test_put_positive(self):
        hp = heston_price(
            S0, K0, T0, R0, Q0,
            v0=0.04, kappa=2.0, theta=0.04,
            xi=0.5, rho=-0.7,
            option_type=OptionType.Put,
        )
        assert float(hp) > 0


# ===== Binomial Tree =====

class TestBinomial:
    def test_european_converges_to_bs(self):
        """European binomial should converge to BS with many steps."""
        bp = binomial_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
                            n_steps=500, american=False)
        bsp = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert abs(float(bp) - float(bsp)) < 0.15

    def test_american_call_no_dividend(self):
        """American call with no dividend = European call."""
        am = binomial_price(S0, K0, T0, R0, 0.0, SIGMA0, OptionType.Call,
                            n_steps=300, american=True)
        eu = binomial_price(S0, K0, T0, R0, 0.0, SIGMA0, OptionType.Call,
                            n_steps=300, american=False)
        assert abs(float(am) - float(eu)) < 0.1

    def test_american_put_ge_european(self):
        """American put should be >= European put."""
        am = binomial_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
                            n_steps=300, american=True)
        eu = binomial_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
                            n_steps=300, american=False)
        assert float(am) >= float(eu) - 0.01

    def test_different_tree_types(self):
        """All tree types produce reasonable prices."""
        for tree in ["crr", "jr", "tian", "trigeorgis"]:
            p = binomial_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
                               n_steps=200, tree_type=tree)
            bsp = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
            assert abs(float(p) - float(bsp)) < 0.5, f"Tree type {tree} failed"


# ===== Finite Differences =====

class TestFD:
    def test_european_converges_to_bs(self):
        """FD European should converge to BS."""
        fd = fd_black_scholes_price(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
            n_space=200, n_time=200,
        )
        bsp = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert abs(float(fd) - float(bsp)) < 0.5

    def test_american_put_ge_european(self):
        """American FD put >= European BS."""
        am = fd_black_scholes_price(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
            n_space=200, n_time=200, american=True,
        )
        eu = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        assert float(am) >= float(eu) - 0.1

    def test_put_price(self):
        """FD European put should converge to BS."""
        fd = fd_black_scholes_price(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
            n_space=200, n_time=200,
        )
        bsp = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        assert abs(float(fd) - float(bsp)) < 0.5


# ===== Barrier Options =====

class TestBarrier:
    def test_down_out_call(self):
        """Down-and-out call with barrier below spot."""
        p = analytic_barrier_price(
            S0, K0, T0, R0, Q0, SIGMA0,
            OptionType.Call, BarrierType.DownOut,
            barrier=80.0,
        )
        # Should be less than vanilla
        vanilla = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert 0 < float(p) < float(vanilla) + 0.01

    def test_knock_in_out_parity(self):
        """KI + KO = vanilla (with zero rebate)."""
        vanilla = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        ki = analytic_barrier_price(
            S0, K0, T0, R0, Q0, SIGMA0,
            OptionType.Call, BarrierType.DownIn, barrier=80.0,
        )
        ko = analytic_barrier_price(
            S0, K0, T0, R0, Q0, SIGMA0,
            OptionType.Call, BarrierType.DownOut, barrier=80.0,
        )
        assert abs(float(ki + ko) - float(vanilla)) < 0.5


# ===== Asian Options =====

class TestAsian:
    def test_geometric_less_than_vanilla(self):
        """Geometric average Asian call < vanilla call."""
        asian = analytic_continuous_geometric_asian_price(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
        )
        vanilla = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert 0 < float(asian) < float(vanilla)

    def test_discrete_positive(self):
        p = analytic_discrete_geometric_asian_price(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call, n_fixings=12,
        )
        assert float(p) > 0

    def test_continuous_put_call_parity(self):
        """Geometric Asian put-call parity check."""
        c = analytic_continuous_geometric_asian_price(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
        )
        p = analytic_continuous_geometric_asian_price(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
        )
        # C - P should be positive when S > K*exp(-rT)
        assert float(c) > float(p)


# ===== Stochastic Processes =====

class TestProcesses:
    def test_bs_process_evolve(self):
        from ql_jax.processes.black_scholes import BlackScholesProcess
        proc = BlackScholesProcess(spot=100.0, rate=0.05, dividend=0.02, volatility=0.2)
        log_s = jnp.log(100.0)
        dt = 1.0 / 252.0
        dw = 0.01 * jnp.sqrt(dt)
        new_log_s = proc.evolve(0.0, log_s, dt, dw)
        assert jnp.isfinite(new_log_s)

    def test_heston_process_evolve(self):
        from ql_jax.processes.heston import HestonProcess
        proc = HestonProcess(
            spot=100.0, rate=0.05, dividend=0.02,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
        )
        state = (jnp.log(100.0), 0.04)
        dt = 1.0 / 252.0
        dw = (0.01 * jnp.sqrt(dt), -0.005 * jnp.sqrt(dt))
        new_state = proc.evolve(0.0, state, dt, dw)
        assert jnp.isfinite(new_state[0])
        assert jnp.isfinite(new_state[1])


# ===== AD through pricing =====

class TestPricingAD:
    def test_bs_delta_via_grad(self):
        """Delta via jax.grad should match bs_greeks."""
        def price_fn(s):
            return black_scholes_price(s, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        delta_ad = jax.grad(price_fn)(jnp.float64(S0))
        g = bs_greeks(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert abs(float(delta_ad) - g["delta"]) < 1e-6

    def test_heston_differentiable(self):
        """Heston price should be differentiable w.r.t. spot."""
        def price_fn(s):
            return heston_price(s, K0, T0, R0, Q0,
                                0.04, 2.0, 0.04, 0.5, -0.7,
                                OptionType.Call)
        delta = jax.grad(price_fn)(jnp.float64(S0))
        assert jnp.isfinite(delta)
        assert float(delta) > 0  # call delta > 0

    def test_binomial_differentiable(self):
        """Binomial tree should be differentiable."""
        def price_fn(s):
            return binomial_price(s, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
                                  n_steps=50)
        delta = jax.grad(price_fn)(jnp.float64(S0))
        assert jnp.isfinite(delta)
        assert 0.0 < float(delta) < 1.0
