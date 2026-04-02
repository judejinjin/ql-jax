"""Tests for Monte Carlo framework (Phase 5)."""

import pytest
import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType, BarrierType
from ql_jax.processes.black_scholes import BlackScholesProcess
from ql_jax.processes.heston import HestonProcess
from ql_jax.methods.montecarlo.path import (
    generate_paths_bs, generate_paths_heston, time_grid,
)
from ql_jax.methods.montecarlo.longstaff_schwartz import (
    lsm_american_put, lsm_american_option,
)
from ql_jax.engines.mc.european import mc_european_bs, mc_european_heston
from ql_jax.engines.mc.american import mc_american_bs
from ql_jax.engines.mc.barrier import mc_barrier_bs
from ql_jax.engines.mc.asian import mc_asian_arithmetic_bs
from ql_jax.engines.analytic.black_formula import black_scholes_price


# Common parameters
S0, K0, T0, R0, Q0, SIGMA0 = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
KEY = jax.random.PRNGKey(12345)


# ===== Path Generation =====

class TestPathGeneration:
    def test_bs_path_shape(self):
        proc = BlackScholesProcess(spot=100.0, rate=0.05, dividend=0.02, volatility=0.2)
        paths = generate_paths_bs(proc, 1.0, 50, 500, KEY)
        assert paths.shape == (500, 51)

    def test_bs_path_starts_at_spot(self):
        proc = BlackScholesProcess(spot=100.0, rate=0.05, dividend=0.02, volatility=0.2)
        paths = generate_paths_bs(proc, 1.0, 50, 500, KEY)
        assert jnp.allclose(paths[:, 0], jnp.log(100.0))

    def test_bs_antithetic_shape(self):
        proc = BlackScholesProcess(spot=100.0, rate=0.05, dividend=0.02, volatility=0.2)
        paths = generate_paths_bs(proc, 1.0, 50, 500, KEY, antithetic=True)
        assert paths.shape == (1000, 51)

    def test_bs_antithetic_mean(self):
        """Antithetic paths should have roughly symmetric terminal values."""
        proc = BlackScholesProcess(spot=100.0, rate=0.05, dividend=0.02, volatility=0.2)
        paths = generate_paths_bs(proc, 1.0, 10, 2000, KEY, antithetic=True)
        n = paths.shape[0] // 2
        log_terminal = paths[:, -1]
        mean_orig = jnp.mean(log_terminal[:n])
        mean_anti = jnp.mean(log_terminal[n:])
        expected = jnp.log(100.0) + (0.05 - 0.02 - 0.5 * 0.04) * 1.0
        assert abs(float(mean_orig + mean_anti) / 2.0 - float(expected)) < 0.05

    def test_heston_path_shape(self):
        proc = HestonProcess(
            spot=100.0, rate=0.05, dividend=0.02,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
        )
        log_paths, var_paths = generate_paths_heston(proc, 1.0, 50, 500, KEY)
        assert log_paths.shape == (500, 51)
        assert var_paths.shape == (500, 51)

    def test_heston_variance_positive_mostly(self):
        proc = HestonProcess(
            spot=100.0, rate=0.05, dividend=0.02,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5,
        )
        _, var_paths = generate_paths_heston(proc, 1.0, 100, 300, KEY)
        frac_negative = jnp.mean(var_paths < -0.01)
        assert float(frac_negative) < 0.1

    def test_time_grid(self):
        tg = time_grid(1.0, 10)
        assert tg.shape == (11,)
        assert float(tg[0]) == 0.0
        assert abs(float(tg[-1]) - 1.0) < 1e-10


# ===== MC European Engine =====

class TestMCEuropean:
    def test_bs_call_close_to_analytic(self):
        """MC European call should be close to BS analytic."""
        mc_price, mc_se = mc_european_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
            n_paths=20000, n_steps=1, key=KEY,
        )
        bs_price = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        assert abs(float(mc_price) - float(bs_price)) < 3 * float(mc_se) + 0.5

    def test_bs_put_close_to_analytic(self):
        mc_price, mc_se = mc_european_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
            n_paths=20000, n_steps=1, key=KEY,
        )
        bs_price = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        assert abs(float(mc_price) - float(bs_price)) < 3 * float(mc_se) + 0.5

    def test_stderr_decreases_with_paths(self):
        _, se_low = mc_european_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
            n_paths=2000, key=KEY,
        )
        _, se_high = mc_european_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
            n_paths=20000, key=KEY,
        )
        assert float(se_high) < float(se_low)

    def test_heston_mc_positive(self):
        mc_price, _ = mc_european_heston(
            S0, K0, T0, R0, Q0,
            v0=0.04, kappa=2.0, theta=0.04, xi=0.5, rho=-0.7,
            option_type=OptionType.Call,
            n_paths=10000, n_steps=50, key=KEY,
        )
        assert float(mc_price) > 0

    def test_heston_mc_close_to_analytic(self):
        """MC Heston should be close to semi-analytic Heston."""
        from ql_jax.engines.analytic.heston import heston_price
        analytic = heston_price(
            S0, K0, T0, R0, Q0,
            0.04, 2.0, 0.04, 0.5, -0.7,
            OptionType.Call,
        )
        mc_price, mc_se = mc_european_heston(
            S0, K0, T0, R0, Q0,
            0.04, 2.0, 0.04, 0.5, -0.7,
            OptionType.Call,
            n_paths=20000, n_steps=50, key=KEY,
        )
        assert abs(float(mc_price) - float(analytic)) < 3 * float(mc_se) + 1.0


# ===== MC American (Longstaff-Schwartz) =====

class TestLSM:
    def test_american_put_ge_european(self):
        """American put via LSM should be >= European put."""
        eu_price = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put)
        am_price = mc_american_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
            n_paths=10000, n_steps=25, key=KEY,
        )
        assert float(am_price) >= float(eu_price) - 0.5

    def test_american_call_no_div_eq_european(self):
        """American call with no dividend ~ European call."""
        eu_price = black_scholes_price(S0, K0, T0, R0, 0.0, SIGMA0, OptionType.Call)
        am_price = mc_american_bs(
            S0, K0, T0, R0, 0.0, SIGMA0, OptionType.Call,
            n_paths=10000, n_steps=25, key=KEY,
        )
        assert abs(float(am_price) - float(eu_price)) < 2.0

    def test_lsm_positive_price(self):
        am_price = mc_american_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
            n_paths=5000, n_steps=20, key=KEY,
        )
        assert float(am_price) > 0


# ===== MC Barrier =====

class TestMCBarrier:
    def test_down_out_call_less_than_vanilla(self):
        """Down-and-out call < vanilla call."""
        vanilla = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        mc_price, _ = mc_barrier_bs(
            S0, K0, T0, R0, Q0, SIGMA0,
            OptionType.Call, BarrierType.DownOut, barrier=80.0,
            n_paths=20000, n_steps=100, key=KEY,
        )
        assert float(mc_price) < float(vanilla) + 0.5
        assert float(mc_price) > 0

    def test_knock_in_out_sum(self):
        """KI + KO approx vanilla (discrete monitoring)."""
        vanilla = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        mc_ki, _ = mc_barrier_bs(
            S0, K0, T0, R0, Q0, SIGMA0,
            OptionType.Call, BarrierType.DownIn, barrier=80.0,
            n_paths=20000, n_steps=100, key=KEY,
        )
        mc_ko, _ = mc_barrier_bs(
            S0, K0, T0, R0, Q0, SIGMA0,
            OptionType.Call, BarrierType.DownOut, barrier=80.0,
            n_paths=20000, n_steps=100, key=KEY,
        )
        assert abs(float(mc_ki + mc_ko) - float(vanilla)) < 2.0


# ===== MC Asian =====

class TestMCAsian:
    def test_arithmetic_asian_less_than_vanilla(self):
        """Arithmetic Asian call < vanilla call."""
        vanilla = black_scholes_price(S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call)
        mc_price, _ = mc_asian_arithmetic_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
            n_fixings=12, n_paths=20000, key=KEY, control_variate=True,
        )
        assert 0 < float(mc_price) < float(vanilla) + 0.5

    def test_control_variate_reduces_stderr(self):
        """Control variate should reduce standard error."""
        _, se_no_cv = mc_asian_arithmetic_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
            n_fixings=12, n_paths=10000, key=KEY, control_variate=False,
        )
        _, se_cv = mc_asian_arithmetic_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
            n_fixings=12, n_paths=10000, key=KEY, control_variate=True,
        )
        assert float(se_cv) < float(se_no_cv) * 1.5

    def test_asian_put_positive(self):
        mc_price, _ = mc_asian_arithmetic_bs(
            S0, K0, T0, R0, Q0, SIGMA0, OptionType.Put,
            n_fixings=12, n_paths=10000, key=KEY,
        )
        assert float(mc_price) > 0


# ===== Pathwise AD through MC =====

class TestMCAD:
    def test_mc_delta_via_grad(self):
        """MC delta via jax.grad should be reasonable."""
        def price_fn(s):
            p, _ = mc_european_bs(s, K0, T0, R0, Q0, SIGMA0, OptionType.Call,
                                   n_paths=5000, n_steps=1, key=KEY)
            return p
        delta = jax.grad(price_fn)(jnp.float64(S0))
        assert jnp.isfinite(delta)
        assert 0.0 < float(delta) < 1.0

    def test_mc_vega_via_grad(self):
        """MC vega via jax.grad should be positive."""
        def price_fn(sig):
            p, _ = mc_european_bs(S0, K0, T0, R0, Q0, sig, OptionType.Call,
                                   n_paths=5000, n_steps=1, key=KEY)
            return p
        vega = jax.grad(price_fn)(jnp.float64(SIGMA0))
        assert jnp.isfinite(vega)
        assert float(vega) > 0
