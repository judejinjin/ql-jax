"""Tests for all stochastic processes."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestOrnsteinUhlenbeck:
    def test_expectation(self):
        from ql_jax.processes.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
        p = OrnsteinUhlenbeckProcess(speed=0.5, volatility=0.1, x0=0.05, level=0.03)
        e = p.expectation(0.0, 0.05, 1.0)
        # E[x(1)] = 0.03 + (0.05 - 0.03)*exp(-0.5)
        expected = 0.03 + 0.02 * jnp.exp(-0.5)
        assert abs(float(e) - float(expected)) < 1e-12

    def test_variance(self):
        from ql_jax.processes.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
        p = OrnsteinUhlenbeckProcess(speed=0.5, volatility=0.1, x0=0.05, level=0.03)
        v = p.variance(0.0, 0.05, 1.0)
        expected = 0.01 / 1.0 * (1.0 - jnp.exp(-1.0))
        assert abs(float(v) - float(expected)) < 1e-12

    def test_drift_diffusion(self):
        from ql_jax.processes.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess
        p = OrnsteinUhlenbeckProcess(speed=2.0, volatility=0.3, x0=0.0, level=0.05)
        assert abs(p.drift(0.0, 0.1) - 2.0 * (0.05 - 0.1)) < 1e-12
        assert abs(p.diffusion(0.0, 0.1) - 0.3) < 1e-12


class TestCoxIngersollRoss:
    def test_expectation(self):
        from ql_jax.processes.cir import CoxIngersollRossProcess
        p = CoxIngersollRossProcess(speed=1.0, volatility=0.2, x0=0.05, level=0.04)
        e = p.expectation(0.0, 0.05, 1.0)
        expected = 0.04 + (0.05 - 0.04) * jnp.exp(-1.0)
        assert abs(float(e) - float(expected)) < 1e-12

    def test_positivity(self):
        from ql_jax.processes.cir import CoxIngersollRossProcess
        p = CoxIngersollRossProcess(speed=1.0, volatility=0.2, x0=0.01, level=0.04)
        key = jax.random.PRNGKey(42)
        dt = 0.01
        x = 0.01
        for _ in range(100):
            key, subkey = jax.random.split(key)
            dw = jax.random.normal(subkey) * jnp.sqrt(dt)
            x = p.evolve(0.0, x, dt, dw)
            assert float(x) >= 0.0

    def test_drift(self):
        from ql_jax.processes.cir import CoxIngersollRossProcess
        p = CoxIngersollRossProcess(speed=2.0, volatility=0.3, x0=0.05, level=0.04)
        d = p.drift(0.0, 0.05)
        assert abs(float(d) - 2.0 * (0.04 - 0.05)) < 1e-12


class TestGBM:
    def test_exact_evolution(self):
        from ql_jax.processes.gbm import GeometricBrownianMotionProcess
        p = GeometricBrownianMotionProcess(x0=100.0, mu=0.05, sigma=0.2)
        # With zero noise, should get deterministic drift
        dt = 1.0
        dw = jnp.array(0.0) * jnp.sqrt(dt)  # zero noise
        result = p.evolve(0.0, 100.0, dt, dw)
        expected = 100.0 * jnp.exp((0.05 - 0.02) * 1.0)
        assert abs(float(result) - float(expected)) < 1e-10

    def test_expectation(self):
        from ql_jax.processes.gbm import GeometricBrownianMotionProcess
        p = GeometricBrownianMotionProcess(x0=100.0, mu=0.05, sigma=0.2)
        e = p.expectation(0.0, 100.0, 1.0)
        assert abs(float(e) - 100.0 * jnp.exp(0.05)) < 1e-10


class TestSquareRoot:
    def test_expectation(self):
        from ql_jax.processes.square_root import SquareRootProcess
        p = SquareRootProcess(speed=1.0, volatility=0.3, x0=0.04, level=0.05)
        e = p.expectation(0.0, 0.04, 1.0)
        expected = 0.05 + (0.04 - 0.05) * jnp.exp(-1.0)
        assert abs(float(e) - float(expected)) < 1e-12

    def test_evolve_positive(self):
        from ql_jax.processes.square_root import SquareRootProcess
        p = SquareRootProcess(speed=1.0, volatility=0.3, x0=0.04, level=0.05)
        result = p.evolve(0.0, 0.04, 0.01, jnp.array(-5.0) * jnp.sqrt(0.01))
        assert float(result) >= 0.0


class TestHullWhiteProcess:
    def test_B_function(self):
        from ql_jax.processes.hull_white import HullWhiteProcess
        p = HullWhiteProcess(a=0.1, sigma=0.01)
        b = p.B(0.0, 1.0)
        expected = (1.0 - jnp.exp(-0.1)) / 0.1
        assert abs(float(b) - float(expected)) < 1e-12

    def test_drift(self):
        from ql_jax.processes.hull_white import HullWhiteProcess
        p = HullWhiteProcess(a=0.1, sigma=0.01)
        d = p.drift(0.0, 0.05)
        assert abs(float(d) - (-0.1 * 0.05)) < 1e-12

    def test_expectation_decays(self):
        from ql_jax.processes.hull_white import HullWhiteProcess
        p = HullWhiteProcess(a=0.5, sigma=0.01)
        e = p.expectation(0.0, 1.0, 10.0)
        assert abs(float(e)) < 0.01  # should decay to near zero


class TestG2Process:
    def test_expectation(self):
        from ql_jax.processes.g2 import G2Process
        p = G2Process(a=0.1, sigma=0.01, b=0.2, eta=0.02, rho=0.5)
        e = p.expectation(0.0, (0.01, 0.02), 1.0)
        assert abs(float(e[0]) - 0.01 * jnp.exp(-0.1)) < 1e-12
        assert abs(float(e[1]) - 0.02 * jnp.exp(-0.2)) < 1e-12

    def test_covariance_symmetric(self):
        from ql_jax.processes.g2 import G2Process
        p = G2Process(a=0.1, sigma=0.01, b=0.2, eta=0.02, rho=0.5)
        cov = p.covariance(0.0, (0.0, 0.0), 1.0)
        assert abs(float(cov[0, 1]) - float(cov[1, 0])) < 1e-15

    def test_covariance_positive(self):
        from ql_jax.processes.g2 import G2Process
        p = G2Process(a=0.1, sigma=0.01, b=0.2, eta=0.02, rho=0.5)
        cov = p.covariance(0.0, (0.0, 0.0), 1.0)
        assert float(cov[0, 0]) > 0
        assert float(cov[1, 1]) > 0


class TestBatesProcess:
    def test_jump_compensator(self):
        from ql_jax.processes.bates import BatesProcess
        p = BatesProcess(
            spot=100, rate=0.05, dividend=0.02, v0=0.04,
            kappa=1.0, theta=0.04, xi=0.5, rho=-0.7,
            lambda_=1.0, nu=-0.1, delta=0.2
        )
        m = p.m
        expected = jnp.exp(-0.1 + 0.5 * 0.04) - 1.0
        assert abs(float(m) - float(expected)) < 1e-12

    def test_evolve_shape(self):
        from ql_jax.processes.bates import BatesProcess
        p = BatesProcess(
            spot=100, rate=0.05, dividend=0.02, v0=0.04,
            kappa=1.0, theta=0.04, xi=0.5, rho=-0.7,
            lambda_=1.0, nu=-0.1, delta=0.2
        )
        dt = 0.01
        state = (jnp.log(100.0), jnp.array(0.04))
        dw = (jnp.array(0.1), jnp.array(-0.05), jnp.array(0.5))
        log_s, v = p.evolve(0.0, state, dt, dw)
        assert jnp.isfinite(log_s)
        assert jnp.isfinite(v)


class TestMerton76Process:
    def test_compensator(self):
        from ql_jax.processes.merton76 import Merton76Process
        p = Merton76Process(
            spot=100, rate=0.05, dividend=0.0, volatility=0.2,
            lambda_=1.0, nu=0.0, delta=0.1
        )
        m = p.m
        expected = jnp.exp(0.5 * 0.01) - 1.0
        assert abs(float(m) - float(expected)) < 1e-12

    def test_evolve_no_jump(self):
        from ql_jax.processes.merton76 import Merton76Process
        p = Merton76Process(
            spot=100, rate=0.05, dividend=0.0, volatility=0.2,
            lambda_=0.0, nu=0.0, delta=0.0
        )
        # With zero jump intensity, should behave like BS
        dt = 0.01
        dw = (jnp.array(0.01), jnp.array(0.5))
        result = p.evolve(0.0, jnp.log(100.0), dt, dw)
        assert jnp.isfinite(result)


class TestGJRGARCH:
    def test_evolve(self):
        from ql_jax.processes.gjrgarch import GJRGARCHProcess
        p = GJRGARCHProcess(
            spot=100, rate=0.05, dividend=0.02, v0=0.0001,
            omega=1e-6, alpha=0.1, beta=0.85, gamma=0.05, lambda_=0.0
        )
        state = (jnp.log(100.0), jnp.array(0.0001))
        dw = jnp.array([0.01, 0.0])
        log_s, h = p.evolve(0.0, state, 1.0 / 252.0, dw)
        assert jnp.isfinite(log_s)
        assert jnp.isfinite(h)


class TestHestonSLV:
    def test_pure_heston_mode(self):
        from ql_jax.processes.heston_slv import HestonSLVProcess
        p = HestonSLVProcess(
            spot=100, rate=0.05, dividend=0.02, v0=0.04,
            kappa=1.0, theta=0.04, xi=0.5, rho=-0.7,
            leverage_fn=None
        )
        state = (jnp.log(100.0), jnp.array(0.04))
        dw = (jnp.array(0.01), jnp.array(-0.01))
        log_s, v = p.evolve(0.0, state, 0.01, dw)
        assert jnp.isfinite(log_s)
        assert jnp.isfinite(v)

    def test_with_leverage(self):
        from ql_jax.processes.heston_slv import HestonSLVProcess
        lev_fn = lambda t, s: 1.5  # constant leverage
        p = HestonSLVProcess(
            spot=100, rate=0.05, dividend=0.02, v0=0.04,
            kappa=1.0, theta=0.04, xi=0.5, rho=-0.7,
            leverage_fn=lev_fn
        )
        state = (jnp.log(100.0), jnp.array(0.04))
        dw = (jnp.array(0.01), jnp.array(-0.01))
        log_s, v = p.evolve(0.0, state, 0.01, dw)
        assert jnp.isfinite(log_s)


class TestProcessArray:
    def test_independent(self):
        from ql_jax.processes.gbm import GeometricBrownianMotionProcess
        from ql_jax.processes.process_array import StochasticProcessArray

        p1 = GeometricBrownianMotionProcess(x0=100.0, mu=0.05, sigma=0.2)
        p2 = GeometricBrownianMotionProcess(x0=50.0, mu=0.03, sigma=0.15)
        pa = StochasticProcessArray.create([p1, p2])

        assert pa.size == 2
        x0 = jnp.array([100.0, 50.0])
        dt = 0.01
        dw = jnp.array([0.01, -0.01])
        x_new = pa.evolve(0.0, x0, dt, dw)
        assert x_new.shape == (2,)
        assert jnp.all(jnp.isfinite(x_new))

    def test_correlated(self):
        from ql_jax.processes.gbm import GeometricBrownianMotionProcess
        from ql_jax.processes.process_array import StochasticProcessArray

        p1 = GeometricBrownianMotionProcess(x0=100.0, mu=0.05, sigma=0.2)
        p2 = GeometricBrownianMotionProcess(x0=50.0, mu=0.03, sigma=0.15)
        corr = jnp.array([[1.0, 0.5], [0.5, 1.0]])
        pa = StochasticProcessArray.create([p1, p2], correlation=corr)

        x0 = jnp.array([100.0, 50.0])
        dt = 0.01
        dw = jnp.array([0.01, -0.01])
        x_new = pa.evolve(0.0, x0, dt, dw)
        assert x_new.shape == (2,)


class TestDiscretization:
    def test_euler(self):
        from ql_jax.processes.discretization import EulerDiscretization
        from ql_jax.processes.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess

        p = OrnsteinUhlenbeckProcess(speed=1.0, volatility=0.1, x0=0.0, level=0.05)
        euler = EulerDiscretization()
        x_new = euler.evolve(p, 0.0, 0.0, 0.01, jnp.array(0.001))
        expected = 0.0 + 1.0 * 0.05 * 0.01 + 0.1 * 0.001
        assert abs(float(x_new) - expected) < 1e-12

    def test_end_euler(self):
        from ql_jax.processes.discretization import EndEulerDiscretization
        from ql_jax.processes.ornstein_uhlenbeck import OrnsteinUhlenbeckProcess

        p = OrnsteinUhlenbeckProcess(speed=1.0, volatility=0.1, x0=0.0, level=0.05)
        end_euler = EndEulerDiscretization()
        x_new = end_euler.evolve(p, 0.0, 0.0, 0.01, jnp.array(0.001))
        assert jnp.isfinite(x_new)
